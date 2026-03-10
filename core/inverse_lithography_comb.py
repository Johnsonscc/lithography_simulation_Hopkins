import time
import logging
import numpy as np
import cv2
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.sparse.linalg import svds
from tqdm import tqdm
from scipy.sparse import lil_matrix, csr_matrix
from scipy.ndimage import laplace  # 用于稳定的 NILS 梯度
from config.parameters import *
from utils.optimization_logger import OptimizationLogger

logger = logging.getLogger(__name__)


class EdgeConstrainedInverseLithographyOptimizer:
    """
    边缘约束逆光刻优化器 (Edge-Constrained PE Base)
    包含了带“形状守卫”的拉普拉斯 NILS 联合优化方案
    """

    def __init__(self, lambda_=LAMBDA, na=NA, sigma=SIGMA, dx=DX, dy=DY,
                 lx=LX, ly=LY, k_svd=ILT_K_SVD, a=A, tr=TR,
                 optimizer_type=OPTIMIZER_TYPE, edge_pixel_range=5,
                 canny_low_threshold=1, canny_high_threshold=300,
                 lambda_nils=0.5, nils_cd=NILS_CD, nils_edge_dilation=5,
                 apply_sharpening=False, sharpening_strength=2.0, sharpening_sigma=1):
        """
        参数:
            lambda_nils: NILS 梯度的最大缩放比例 (建议 0.1~0.2，0 表示不开启)
            apply_sharpening: 是否应用空间像锐化 (建议设为 False，使用真实的 ILT NILS 优化)
        """
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
        self.canny_low_threshold = canny_low_threshold
        self.canny_high_threshold = canny_high_threshold
        self.lambda_nils = lambda_nils
        self.nils_cd = nils_cd
        self.nils_edge_dilation = nils_edge_dilation
        self.apply_sharpening = apply_sharpening
        self.sharpening_strength = sharpening_strength
        self.sharpening_sigma = sharpening_sigma
        self.optimizer_state = {}

        self.update_mask = None
        self.tcc_matrix = None
        self.singular_values = None
        self.eigen_functions = None
        self._precompute_tcc_svd()

        logger.info(
            f"EdgeConstrainedInverseLithographyOptimizer initialized: opt={optimizer_type}, "
            f"lambda_nils={lambda_nils}, apply_sharpening={apply_sharpening}")

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

        U, S, Vh = svds(TCC_csr, k=k_actual, random_state=42)
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

        self.tcc_matrix = TCC_4d
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

    def _detect_edge_region(self, target, edge_pixel_range=5):
        binary_target = (target > 0.5).astype(np.uint8)
        target_uint8 = (binary_target * 255).astype(np.uint8)

        edges = cv2.Canny(target_uint8, self.canny_low_threshold, self.canny_high_threshold)
        edge_binary = (edges > 0).astype(np.uint8)

        if edge_pixel_range > 0:
            kernel_size = 2 * edge_pixel_range + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            update_mask = cv2.dilate(edge_binary, kernel, iterations=1)
        else:
            update_mask = edge_binary

        return update_mask

    def _apply_update_mask(self, gradient, update_mask):
        smooth_mask = cv2.GaussianBlur(update_mask.astype(np.float32), (3, 3), 0.5)
        return gradient * smooth_mask

    def _compute_nils_gradient_stable(self, target, intensity_raw, A_i_list):
        """
        基于拉普拉斯算子的保形 NILS 梯度
        改进：放宽边缘带至约3像素宽，增加有效区域，不再除以面积
        """
        binary_target = (target > 0.5).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)

        # 放宽到 3 像素：用膨胀 1 次减去腐蚀 2 次的内侧
        dilated = cv2.dilate(binary_target, kernel, iterations=1)
        eroded = cv2.erode(binary_target, kernel, iterations=2)  # 腐蚀两次扩大着力带
        edge_region = (dilated - eroded) > 0

        area = np.sum(edge_region)
        if area == 0:
            return np.zeros_like(intensity_raw, dtype=np.complex128)

        # 计算负拉普拉斯 (提供边缘锐化的驱动力)，不再除以面积
        grad_nils_raw = -laplace(intensity_raw)
        dL_dI = grad_nils_raw * edge_region.astype(np.float64)

        # 反传至掩模
        gradient_nils = np.zeros_like(intensity_raw, dtype=np.complex128)
        for s_val, H_i, A_i in zip(self.singular_values, self.eigen_functions, A_i_list):
            dL_dA_i = dL_dI * 2 * s_val * A_i.conj()
            gradient_nils += ifft2(ifftshift(fftshift(fft2(dL_dA_i)) * np.conj(H_i)))

        return np.real(gradient_nils)

    def _compute_analytical_gradient(self, mask, target, current_iteration=0, max_iterations=100):
        # 1. 前向仿真
        M_fft = fftshift(fft2(mask))
        A_i_list, intensity = [], np.zeros((self.lx, self.ly))

        for s_val, H_i in zip(self.singular_values, self.eigen_functions):
            A_i = ifft2(ifftshift(M_fft * H_i))
            A_i_list.append(A_i)
            intensity += s_val * (np.abs(A_i) ** 2)

        intensity_raw = intensity.copy()
        i_min, i_max = np.min(intensity), np.max(intensity)
        denom = i_max - i_min if (i_max - i_min) > 1e-10 else 1.0
        intensity_norm = (intensity - i_min) / denom
        P = self.photoresist_model(intensity_norm)

        # 2. 基础 PE/EPE 计算
        pe_loss = np.sum((target - P) ** 2)
        gy, gx = np.gradient(P)
        W = np.sqrt(gx ** 2 + gy ** 2 + 1e-10)
        epe_loss = np.sum(((P - target) ** 2) * (W / np.max(W) if np.max(W) > 0 else 1))

        # 3. 基础 PE 梯度
        dP_dI = (self.a * P * (1 - P)) * (1.0 / denom)
        dF_dP = -2 * (target - P)

        gradient_pe = np.zeros_like(mask, dtype=np.complex128)
        for s_val, H_i, A_i in zip(self.singular_values, self.eigen_functions, A_i_list):
            dF_dA_i = dF_dP * dP_dI * 2 * s_val * A_i.conj()
            gradient_pe += ifft2(ifftshift(fftshift(fft2(dF_dA_i)) * np.conj(H_i)))
        gradient_pe = np.real(gradient_pe)

        total_gradient = gradient_pe
        total_loss = pe_loss

        # 初始化 NILS 梯度变量（用于返回）
        grad_nils_adj = None

        # ===== 4. 融合 NILS 梯度 (梯度手术 PCGrad + 严格能量匹配) =====
        warmup_iters = int(max_iterations * 0.3)

        if self.lambda_nils > 0 and current_iteration >= warmup_iters:
            # 计算原始 NILS 锐化梯度
            grad_nils = self._compute_nils_gradient_stable(target, intensity_raw, A_i_list)

            # 展平以便于计算向量点积
            g_pe_flat = gradient_pe.ravel()
            g_nils_flat = grad_nils.ravel()

            # 计算点积判断是否冲突
            dot_product = np.dot(g_nils_flat, g_pe_flat)

            # 梯度手术：如果点积小于0，剔除冲突分量
            if dot_product < 0:
                pe_norm_sq = np.dot(g_pe_flat, g_pe_flat) + 1e-12
                grad_nils_proj = grad_nils - (dot_product / pe_norm_sq) * gradient_pe
            else:
                grad_nils_proj = grad_nils

            # 采用 L2 范数（整体能量）进行匹配，而不是最大值
            norm_pe = np.linalg.norm(gradient_pe)
            norm_nils = np.linalg.norm(grad_nils_proj)

            if norm_nils > 1e-12 and norm_pe > 1e-12:
                progress = (current_iteration - warmup_iters) / (max_iterations - warmup_iters)

                # target_ratio 控制了 NILS 梯度的整体能量相对于 PE 的比例
                target_ratio = progress * self.lambda_nils

                # 强制缩放：保证 norm(grad_nils_adj) == target_ratio * norm(gradient_pe)
                scale_factor = target_ratio * (norm_pe / norm_nils)

                grad_nils_adj = grad_nils_proj * scale_factor
                total_gradient = gradient_pe + grad_nils_adj

        return total_loss, pe_loss, epe_loss, total_gradient, intensity_norm, P, intensity_raw, A_i_list, gradient_pe, grad_nils_adj

    def compute_nils(self, aerial, target, cd=NILS_CD):
        """
        修正物理量纲的严格 NILS 评估函数
        """
        gy, gx = np.gradient(aerial, self.dy, self.dx)
        grad_mag = np.sqrt(gx ** 2 + gy ** 2)

        safe_aerial = np.maximum(aerial, 1e-6)
        log_slope = grad_mag / safe_aerial
        nils_map = cd * log_slope

        binary_target = (target > 0.5).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        strict_edge_mask = (cv2.dilate(binary_target, kernel, iterations=1) -
                            cv2.erode(binary_target, kernel, iterations=1)) > 0

        if np.sum(strict_edge_mask) > 0:
            avg_nils = np.sum(nils_map * strict_edge_mask) / np.sum(strict_edge_mask)
        else:
            avg_nils = 0.0

        return avg_nils

    def optimize(self, initial_mask, target, learning_rate=None,
                 max_iterations=ILT_MAX_ITERATIONS,
                 log_csv=True, log_dir="logs", experiment_tag="",
                 **optimizer_params):

        mask = initial_mask.copy()
        best_mask = initial_mask.copy()
        best_pe_loss = float('inf')

        self.update_mask = self._detect_edge_region(target, self.edge_pixel_range)

        if learning_rate is None:
            learning_rate = ILT_OPTIMIZER_CONFIGS[self.optimizer_type]['learning_rate']

        csv_logger = None
        if log_csv:
            csv_logger = OptimizationLogger(log_dir=log_dir)
            # 现在 _compute_analytical_gradient 返回 10 个值
            total_loss, init_pe, _, _, _, _, _, _, _, _ = self._compute_analytical_gradient(mask, target, 0, max_iterations)
            csv_logger.start_logging(
                optimizer_type=f"EdgeConstrained-{self.optimizer_type}",
                loss_type="PE+NILS(Laplace Guard)",
                learning_rate=learning_rate,
                initial_loss=init_pe,
                target_shape=f"{target.shape}",
                config_params={**optimizer_params, "edge_pixel_range": self.edge_pixel_range,
                               "lambda_nils": self.lambda_nils}
            )

        self._initialize_optimizer_state(mask.shape)
        history = {'pe_loss': [], 'epe_loss': [], 'nils': [], 'grad_norms': [], 'update_region_size': []}
        start_time = time.time()

        print(f"Starting Edge-Constrained Joint Optimization ({max_iterations} iters)...")
        print(f"Warm-up phase: 0 to {int(max_iterations * 0.3)} iters.")

        for iteration in range(max_iterations):
            # 接收10个返回值
            total_loss, pe_loss, epe_loss, gradient, aerial, printed, intensity_raw, A_i_list, grad_pe, grad_nils = \
                self._compute_analytical_gradient(mask, target, iteration, max_iterations)

            nils = self.compute_nils(aerial, target, cd=self.nils_cd)
            history['nils'].append(nils)

            masked_gradient = self._apply_update_mask(gradient, self.update_mask)

            active_pixels = np.sum(np.abs(masked_gradient) > 1e-6)
            update_ratio = active_pixels / masked_gradient.size * 100
            history['update_region_size'].append(update_ratio)
            history['grad_norms'].append(np.linalg.norm(masked_gradient))

            if pe_loss < best_pe_loss:
                best_pe_loss = pe_loss
                best_mask = mask.copy()

            history['pe_loss'].append(pe_loss)
            history['epe_loss'].append(epe_loss)

            if csv_logger:
                csv_logger.log_iteration(iteration, pe_loss, masked_gradient, mask,
                                         self.optimizer_state, time.time() - start_time, nils=nils)

            mask = self._update_with_optimizer(mask, masked_gradient, learning_rate, **optimizer_params)

            # 每 10 次迭代打印详细的梯度信息
            if iteration % 10 == 0:
                grad_pe_norm = np.linalg.norm(grad_pe)
                print(f"Iter {iteration:3d}: PE Loss={pe_loss:.2f}, EPE={epe_loss:.2f}, NILS={nils:.4f}, "
                      f"GradNorm(total)={np.linalg.norm(masked_gradient):.2f}")
                if grad_nils is not None:
                    grad_nils_norm = np.linalg.norm(grad_nils)
                    grad_nils_max = np.max(np.abs(grad_nils))
                    grad_nils_mean = np.mean(np.abs(grad_nils))
                    print(f"         grad_pe norm={grad_pe_norm:.2e}, grad_nils norm={grad_nils_norm:.2e}, "
                          f"grad_nils max={grad_nils_max:.2e}, mean={grad_nils_mean:.2e}")
                else:
                    print(f"         grad_pe norm={grad_pe_norm:.2e} (NILS not yet active)")

        if csv_logger:
            csv_logger.close()

        print(f"Optimization finished. Min PE Loss: {best_pe_loss:.4f}")
        return best_mask, history, self.update_mask


def inverse_lithography_optimization_edge_constrained_base(initial_mask, target_image,
                                                           learning_rate=None,
                                                           max_iterations=ILT_MAX_ITERATIONS,
                                                           optimizer_type=OPTIMIZER_TYPE,
                                                           edge_pixel_range=10,
                                                           canny_low_threshold=50,
                                                           canny_high_threshold=150,
                                                           lambda_nils=0.5,  # 推荐设为 0.5 或更高以发挥效果
                                                           nils_cd=NILS_CD,
                                                           nils_edge_dilation=5,
                                                           apply_sharpening=False,  # 务必设为 False
                                                           sharpening_strength=1.0,
                                                           sharpening_sigma=1.0,
                                                           **optimizer_params):
    """
    边缘约束逆向光刻优化（融合 NILS 拉普拉斯守卫）
    """
    optimizer = EdgeConstrainedInverseLithographyOptimizer(
        optimizer_type=optimizer_type,
        edge_pixel_range=edge_pixel_range,
        canny_low_threshold=canny_low_threshold,
        canny_high_threshold=canny_high_threshold,
        lambda_nils=lambda_nils,
        nils_cd=nils_cd,
        nils_edge_dilation=nils_edge_dilation,
        apply_sharpening=apply_sharpening,
        sharpening_strength=sharpening_strength,
        sharpening_sigma=sharpening_sigma
    )

    optimized_mask, history, update_mask = optimizer.optimize(
        initial_mask=initial_mask,
        target=target_image,
        learning_rate=learning_rate,
        max_iterations=max_iterations,
        **optimizer_params
    )

    return optimized_mask, history, update_mask