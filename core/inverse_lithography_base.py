import time
import logging
import numpy as np
import cv2
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.sparse.linalg import svds
from tqdm import tqdm
from scipy.sparse import lil_matrix, csr_matrix
from config.parameters import *
from utils.optimization_logger import OptimizationLogger

logger = logging.getLogger(__name__)


class EdgeConstrainedInverseLithographyOptimizer:
    """
    边缘约束逆光刻优化器 (Edge-Constrained PE Base)
    执行基于像素误差 (Pixel Error) 的梯度下降优化，但只在边缘区域更新像素
    """

    def __init__(self, lambda_=LAMBDA, na=NA, sigma=SIGMA, dx=DX, dy=DY,
                 lx=LX, ly=LY, k_svd=ILT_K_SVD, a=A, tr=TR,
                 optimizer_type=OPTIMIZER_TYPE, edge_pixel_range=10):
        self.lambda_ = lambda_
        self.na = na
        self.sigma = sigma
        self.dx = dx
        self.dy = dy
        self.lx = lx
        self.ly = ly
        self.k_svd = k_svd
        self.a = a
        self.tr = tr
        self.optimizer_type = optimizer_type
        self.edge_pixel_range = edge_pixel_range
        self.optimizer_state = {}

        # 边缘更新掩膜
        self.update_mask = None

        self.singular_values = None
        self.eigen_functions = None
        self._precompute_tcc_svd()

        logger.info(
            f"EdgeConstrainedInverseLithographyOptimizer initialized with {optimizer_type}, edge_range={edge_pixel_range}")

    def pupil_response_function(self, fx, fy):
        r = np.sqrt(fx ** 2 + fy ** 2)
        r_max = self.na / self.lambda_
        P = np.where(r < r_max, self.lambda_ ** 2 / (np.pi * (self.na) ** 2), 0)
        return P

    def light_source_function(self, fx, fy):
        r = np.sqrt(fx ** 2 + fy ** 2)
        r_max = self.sigma * self.na / self.lambda_
        J = np.where(r <= r_max, self.lambda_ ** 2 / (np.pi * (self.sigma * self.na) ** 2), 0.0)
        return J

    def _compute_full_tcc_matrix(self, fx, fy, sparsity_threshold=0.001):
        Lx, Ly = len(fx), len(fy)
        FX, FY = np.meshgrid(fx, fy, indexing='xy')
        J = self.light_source_function(FX, FY)
        P = self.pupil_response_function(FX, FY)
        tcc_kernel = J * P

        TCC_sparse = lil_matrix((Lx * Ly, Lx * Ly), dtype=np.complex128)
        neighborhood_radius = 10
        for i in tqdm(range(Lx), desc="Base TCC Construction"):
            for j in range(Ly):
                if np.abs(tcc_kernel[i, j]) > sparsity_threshold:
                    for m in range(max(0, i - neighborhood_radius), min(Lx, i + neighborhood_radius + 1)):
                        for n in range(max(0, j - neighborhood_radius), min(Ly, j + neighborhood_radius + 1)):
                            if np.abs(tcc_kernel[m, n]) > sparsity_threshold:
                                idx1 = i * Ly + j
                                idx2 = m * Ly + n
                                TCC_sparse[idx1, idx2] = tcc_kernel[i, j] * np.conj(tcc_kernel[m, n])
        return csr_matrix(TCC_sparse)

    def _svd_of_tcc_matrix(self, TCC_csr, k, Lx=LX, Ly=LY):
        k_actual = min(k, min(TCC_csr.shape) - 1)
        print(f"TCC SVD precomputation completed with {k_actual} singular values")

        U, S, Vh = svds(TCC_csr, k=k_actual)
        significant_mask = S > (np.max(S) * 0.01)
        S, U = S[significant_mask], U[:, significant_mask]
        idx = np.argsort(S)[::-1]
        S, U = S[idx], U[:, idx]

        H_functions = [U[:, i].reshape(Lx, Ly) for i in range(len(S))]
        return S, H_functions

    def _precompute_tcc_svd(self):
        max_freq = self.na / self.lambda_
        freq = 2 * max_freq
        fx = np.linspace(-freq, freq, self.lx)
        fy = np.linspace(-freq, freq, self.ly)
        TCC_4d = self._compute_full_tcc_matrix(fx, fy)
        self.singular_values, self.eigen_functions = self._svd_of_tcc_matrix(TCC_4d, self.k_svd)

    def photoresist_model(self, intensity):
        return 1 / (1 + np.exp(-self.a * (intensity - self.tr)))

    def _initialize_optimizer_state(self, mask_shape):
        if self.optimizer_type == 'sgd':
            self.optimizer_state = {}
        elif self.optimizer_type == 'momentum':
            self.optimizer_state = {'velocity': np.zeros(mask_shape)}
        elif self.optimizer_type == 'rmsprop':
            self.optimizer_state = {'square_avg': np.zeros(mask_shape)}
        elif self.optimizer_type == 'adam':
            self.optimizer_state = {'m': np.zeros(mask_shape), 'v': np.zeros(mask_shape), 't': 0}
        elif self.optimizer_type == 'cg':
            self.optimizer_state = {'prev_grad': None, 'direction': None, 't': 0}

    def _update_with_optimizer(self, mask, gradient, learning_rate, **params):
        if self.optimizer_type == 'sgd':
            new_mask = mask - learning_rate * gradient
        elif self.optimizer_type == 'momentum':
            v = self.optimizer_state['velocity']
            v = params.get('momentum', 0.95) * v - learning_rate * gradient
            self.optimizer_state['velocity'] = v
            new_mask = mask + v
        elif self.optimizer_type == 'rmsprop':
            decay_rate = params.get('decay_rate', 0.99)
            epsilon = params.get('epsilon', 1e-8)
            square_avg = self.optimizer_state['square_avg']
            square_avg = decay_rate * square_avg + (1 - decay_rate) * (gradient ** 2)
            self.optimizer_state['square_avg'] = square_avg
            new_mask = mask - learning_rate * gradient / (np.sqrt(square_avg) + epsilon)
        elif self.optimizer_type == 'cg':
            grad_curr = gradient
            t = self.optimizer_state['t']
            if t == 0:
                direction = -grad_curr
            else:
                grad_prev = self.optimizer_state['prev_grad']
                direction_prev = self.optimizer_state['direction']
                y_k = grad_curr - grad_prev
                numerator = np.sum(grad_curr * y_k)
                denominator = np.sum(grad_prev ** 2)
                beta = numerator / (denominator + 1e-10)
                beta = max(0, beta)
                direction = -grad_curr + beta * direction_prev
            self.optimizer_state['prev_grad'] = grad_curr
            self.optimizer_state['direction'] = direction
            self.optimizer_state['t'] = t + 1
            new_mask = mask + learning_rate * direction
        elif self.optimizer_type == 'adam':
            t = self.optimizer_state['t'] + 1
            m, v = self.optimizer_state['m'], self.optimizer_state['v']
            m = params.get('beta1', 0.9) * m + (1 - 0.9) * gradient
            v = params.get('beta2', 0.999) * v + (1 - 0.999) * (gradient ** 2)
            m_hat = m / (1 - 0.9 ** t)
            v_hat = v / (1 - 0.999 ** t)
            new_mask = mask - learning_rate * m_hat / (np.sqrt(v_hat) + 1e-4)
            self.optimizer_state.update({'m': m, 'v': v, 't': t})

        return np.clip(new_mask, 0, 1)

    def _detect_edge_region(self, target, edge_pixel_range=10):
        """
        检测目标图像的边缘区域，生成更新区域掩膜
        """
        # 确保目标是二值图像
        binary_target = (target > 0.5).astype(np.uint8)

        # 方法1: 使用Sobel算子检测边缘
        sobelx = cv2.Sobel(binary_target, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(binary_target, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)

        # 方法2: 使用形态学梯度检测边缘 (膨胀-腐蚀)
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(binary_target, kernel, iterations=1)
        eroded = cv2.erode(binary_target, kernel, iterations=1)
        morphological_edge = dilated - eroded

        # 组合边缘检测结果
        combined_edge = edge_magnitude + morphological_edge
        edge_binary = (combined_edge > 0).astype(np.uint8)

        # 对边缘区域进行膨胀，扩大更新区域
        if edge_pixel_range > 0:
            kernel_size = 2 * edge_pixel_range + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            update_mask = cv2.dilate(edge_binary, kernel, iterations=1)
        else:
            update_mask = edge_binary

        # 也考虑图案内部区域（可能需要微调）
        pattern_inner = cv2.erode(binary_target, kernel, iterations=edge_pixel_range // 2)
        update_mask = np.logical_or(update_mask, pattern_inner).astype(np.float64)

        # 确保边缘平滑
        update_mask = cv2.GaussianBlur(update_mask.astype(np.float32), (5, 5), 1.0)

        logger.info(
            f"Edge region detection completed. Update region: {np.sum(update_mask > 0.1) / update_mask.size * 100:.1f}% of total area")

        return update_mask

    def _apply_update_mask(self, gradient, update_mask):
        """
        将梯度限制在更新区域内
        """
        # 对更新掩膜进行平滑处理，避免硬边界
        smooth_mask = cv2.GaussianBlur(update_mask, (3, 3), 0.5)

        # 应用掩膜
        masked_gradient = gradient * smooth_mask

        return masked_gradient

    def _compute_analytical_gradient(self, mask, target):
        M_fft = fftshift(fft2(mask))
        A_i_list, intensity = [], np.zeros((self.lx, self.ly))

        for s_val, H_i in zip(self.singular_values, self.eigen_functions):
            A_i = ifft2(ifftshift(M_fft * H_i))
            A_i_list.append(A_i)
            intensity += s_val * (np.abs(A_i) ** 2)

        i_min, i_max = np.min(intensity), np.max(intensity)
        denom = i_max - i_min if (i_max - i_min) > 1e-10 else 1.0
        intensity_norm = (intensity - i_min) / denom
        P = self.photoresist_model(intensity_norm)

        pe_loss = np.sum((target - P) ** 2)

        # 计算 EPE 仅用于监控
        gy, gx = np.gradient(P)
        W = np.sqrt(gx ** 2 + gy ** 2 + 1e-10)
        epe_loss = np.sum(((P - target) ** 2) * (W / np.max(W) if np.max(W) > 0 else 1))

        gradient = np.zeros_like(mask, dtype=np.complex128)
        dP_dI = (self.a * P * (1 - P)) * (1.0 / denom)
        dF_dP = -2 * (target - P)

        for s_val, H_i, A_i in zip(self.singular_values, self.eigen_functions, A_i_list):
            dF_dA_i = dF_dP * dP_dI * 2 * s_val * A_i.conj()
            gradient += ifft2(ifftshift(fftshift(fft2(dF_dA_i)) * np.conj(H_i)))

        return pe_loss, epe_loss, np.real(gradient), intensity_norm, P

    def optimize(self, initial_mask, target, learning_rate=None,
                 max_iterations=ILT_MAX_ITERATIONS,
                 log_csv=True, log_dir="logs", experiment_tag="",
                 **optimizer_params):

        mask = initial_mask.copy()
        best_mask = initial_mask.copy()
        best_pe_loss = float('inf')

        # 检测边缘区域，生成更新掩膜
        print("Detecting edge regions for constrained optimization...")
        self.update_mask = self._detect_edge_region(target, self.edge_pixel_range)

        if learning_rate is None:
            learning_rate = ILT_OPTIMIZER_CONFIGS[self.optimizer_type]['learning_rate']

        # 日志初始化
        csv_logger = None
        if log_csv:
            csv_logger = OptimizationLogger(log_dir=log_dir)
            init_pe, _, _, _, _ = self._compute_analytical_gradient(mask, target)
            csv_logger.start_logging(
                optimizer_type=f"EdgeConstrained-{self.optimizer_type}",
                loss_type="PE (Edge-Constrained)",
                learning_rate=learning_rate,
                initial_loss=init_pe,
                target_shape=f"{target.shape}",
                config_params={**optimizer_params, "edge_pixel_range": self.edge_pixel_range}
            )

        self._initialize_optimizer_state(mask.shape)
        history = {
            'pe_loss': [],
            'epe_loss': [],
            'learning_rates': [],
            'grad_norms': [],
            'update_region_size': []  # 记录实际更新的像素比例
        }
        start_time = time.time()

        print(f"Starting Edge-Constrained Base PE Optimization ({max_iterations} iters)...")
        print(f"Edge pixel range: {self.edge_pixel_range}px")
        print(f"Update region: {np.sum(self.update_mask > 0.1) / self.update_mask.size * 100:.1f}% of total area")

        for iteration in range(max_iterations):
            pe_loss, epe_loss, gradient, aerial, printed = self._compute_analytical_gradient(mask, target)

            # 应用更新掩膜
            masked_gradient = self._apply_update_mask(gradient, self.update_mask)

            # 计算实际更新的像素比例
            active_pixels = np.sum(np.abs(masked_gradient) > 1e-6)
            total_pixels = masked_gradient.size
            update_ratio = active_pixels / total_pixels * 100
            history['update_region_size'].append(update_ratio)

            # 记录梯度范数
            history['grad_norms'].append(np.linalg.norm(masked_gradient))

            # 保存 PE 最优状态
            if pe_loss < best_pe_loss:
                best_pe_loss = pe_loss
                best_mask = mask.copy()

            history['pe_loss'].append(pe_loss)
            history['epe_loss'].append(epe_loss)

            if csv_logger:
                csv_logger.log_iteration(
                    iteration,
                    pe_loss,
                    masked_gradient,
                    mask,
                    self.optimizer_state,
                    time.time() - start_time
                )

            # 使用掩膜后的梯度更新
            mask = self._update_with_optimizer(mask, masked_gradient, learning_rate, **optimizer_params)

            if iteration % 20 == 0:
                print(f"Iter {iteration}: PE Loss={pe_loss:.4f}, Best PE={best_pe_loss:.4f}, "
                      f"Update Region={update_ratio:.1f}%, Grad Norm={np.linalg.norm(masked_gradient):.4f}")

        if csv_logger:
            csv_logger.close()

        avg_update_ratio = np.mean(history['update_region_size'])
        print(f"Optimization finished. Min PE Loss: {best_pe_loss:.4f}")
        print(f"Average update region: {avg_update_ratio:.1f}% of total pixels")

        return best_mask, history, self.update_mask


def inverse_lithography_optimization_edge_constrained_base(initial_mask, target_image,
                                                           learning_rate=None,
                                                           max_iterations=ILT_MAX_ITERATIONS,
                                                           optimizer_type=OPTIMIZER_TYPE,
                                                           edge_pixel_range=10,
                                                           **optimizer_params):
    """
    边缘约束的单PE优化入口函数
    """
    optimizer = EdgeConstrainedInverseLithographyOptimizer(
        optimizer_type=optimizer_type,
        edge_pixel_range=edge_pixel_range
    )

    optimized_mask, history, update_mask = optimizer.optimize(
        initial_mask=initial_mask,
        target=target_image,
        learning_rate=learning_rate,
        max_iterations=max_iterations,
        **optimizer_params
    )

    return optimized_mask, history, update_mask