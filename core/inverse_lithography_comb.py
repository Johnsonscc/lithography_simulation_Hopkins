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
    支持空间像锐化 (unsharp masking) 以验证锐度影响
    """

    def __init__(self, lambda_=LAMBDA, na=NA, sigma=SIGMA, dx=DX, dy=DY,
                 lx=LX, ly=LY, k_svd=ILT_K_SVD, a=A, tr=TR,
                 optimizer_type=OPTIMIZER_TYPE, edge_pixel_range=5,
                 canny_low_threshold=1, canny_high_threshold=300,
                 lambda_nils=0.0, nils_cd=NILS_CD, nils_edge_dilation=3,
                 apply_sharpening=True, sharpening_strength=2.0, sharpening_sigma=1):
        """
        参数:
            apply_sharpening: 是否应用空间像锐化
            sharpening_strength: 锐化强度 (alpha)
            sharpening_sigma: 高斯模糊的标准差
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

        # 边缘更新掩膜
        self.update_mask = None

        # TCC Matrix storage for visualization
        self.tcc_matrix = None

        self.singular_values = None
        self.eigen_functions = None
        self._precompute_tcc_svd()

        logger.info(
            f"EdgeConstrainedInverseLithographyOptimizer initialized with {optimizer_type}, edge_range={edge_pixel_range}, "
            f"Canny thresholds=({canny_low_threshold}, {canny_high_threshold}), lambda_nils={lambda_nils}, "
            f"nils_edge_dilation={nils_edge_dilation}, apply_sharpening={apply_sharpening}")

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

        # Store the TCC matrix for visualization
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

    def _detect_edge_region(self, target, edge_pixel_range=5):
        """
        检测目标图像的边缘区域，生成更新区域掩膜（用于优化更新）
        使用 Canny 检测边缘，并对边缘进行膨胀（膨胀半径由 edge_pixel_range 控制）
        """
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

        logger.info(
            f"Edge region detection completed. Update region: {np.sum(update_mask > 0.1) / update_mask.size * 100:.1f}% of total area")
        return update_mask

    def _detect_edge_region_original(self, target, edge_pixel_range=5):
        """
        原先的边缘检测方法（Sobel + 形态学梯度 + 内部区域补充）
        用于 NILS 计算，以保持指标一致性
        """
        binary_target = (target > 0.5).astype(np.uint8)

        sobelx = cv2.Sobel(binary_target, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(binary_target, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)

        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(binary_target, kernel, iterations=1)
        eroded = cv2.erode(binary_target, kernel, iterations=1)
        morphological_edge = dilated - eroded

        combined_edge = edge_magnitude + morphological_edge
        edge_binary = (combined_edge > 0).astype(np.uint8)

        if edge_pixel_range > 0:
            kernel_size = 2 * edge_pixel_range + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            update_mask = cv2.dilate(edge_binary, kernel, iterations=1)
        else:
            update_mask = edge_binary

        pattern_inner = cv2.erode(binary_target, kernel, iterations=edge_pixel_range // 2)
        update_mask = np.logical_or(update_mask, pattern_inner).astype(np.float64)
        update_mask = cv2.GaussianBlur(update_mask.astype(np.float32), (5, 5), 1.0)

        return update_mask

    def _apply_update_mask(self, gradient, update_mask):
        smooth_mask = cv2.GaussianBlur(update_mask, (3, 3), 0.5)
        masked_gradient = gradient * smooth_mask
        return masked_gradient

    def _compute_analytical_gradient(self, mask, target):
        """
        计算总损失及其梯度，支持空间像锐化
        """
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

        # ------------------ 可选的锐化步骤 ------------------
        if self.apply_sharpening:
            # 生成高斯核
            sigma = self.sharpening_sigma
            ksize = int(2 * round(3 * sigma)) + 1
            if ksize % 2 == 0:
                ksize += 1
            gauss1d = cv2.getGaussianKernel(ksize, sigma)
            gauss_kernel = gauss1d @ gauss1d.T
            # 锐化核: (1+alpha)*delta - alpha * gauss
            identity = np.zeros((ksize, ksize))
            identity[ksize//2, ksize//2] = 1
            sharpen_kernel = (1 + self.sharpening_strength) * identity - self.sharpening_strength * gauss_kernel
            # 应用锐化
            sharpened = cv2.filter2D(intensity_norm, -1, sharpen_kernel, borderType=cv2.BORDER_REPLICATE)
            sharpened = np.clip(sharpened, 0, 1)  # 保持范围
            P_input = sharpened
        else:
            P_input = intensity_norm
            sharpen_kernel = None
        # -------------------------------------------------

        P = self.photoresist_model(P_input)

        pe_loss = np.sum((target - P) ** 2)

        gy, gx = np.gradient(P)
        W = np.sqrt(gx ** 2 + gy ** 2 + 1e-10)
        epe_loss = np.sum(((P - target) ** 2) * (W / np.max(W) if np.max(W) > 0 else 1))

        # PE 梯度计算
        dP_dI_input = (self.a * P * (1 - P)) * (1.0 / denom)  # 对 P_input 的导数

        # 通过锐化核反向传播到 intensity_norm
        if self.apply_sharpening:
            dP_dI = cv2.filter2D(dP_dI_input, -1, sharpen_kernel, borderType=cv2.BORDER_REPLICATE)
        else:
            dP_dI = dP_dI_input

        dF_dP = -2 * (target - P)

        gradient_pe = np.zeros_like(mask, dtype=np.complex128)
        for s_val, H_i, A_i in zip(self.singular_values, self.eigen_functions, A_i_list):
            dF_dA_i = dF_dP * dP_dI * 2 * s_val * A_i.conj()
            gradient_pe += ifft2(ifftshift(fftshift(fft2(dF_dA_i)) * np.conj(H_i)))
        gradient_pe = np.real(gradient_pe)

        total_gradient = gradient_pe
        total_loss = pe_loss

        # NILS 正则化（可选）
        if self.lambda_nils != 0:
            nils_mean = self.compute_nils(intensity_raw, target, cd=self.nils_cd)
            nils_loss = -self.lambda_nils * nils_mean
            total_loss = pe_loss + nils_loss

            grad_nils = self._compute_nils_gradient(target, intensity_raw, A_i_list,
                                                     self.singular_values, self.eigen_functions,
                                                     cd=self.nils_cd)
            total_gradient = gradient_pe + grad_nils

        return total_loss, pe_loss, epe_loss, total_gradient, intensity_norm, P, intensity_raw, A_i_list

    def _compute_nils_gradient(self, target, intensity_raw, A_i_list, singular_values, eigen_functions, cd):
        """
        计算 -lambda_nils * NILS 对 mask 的梯度（完整变分导数，含散度项）
        """
        binary_target = (target > 0.5).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(binary_target, kernel, iterations=1)
        eroded = cv2.erode(binary_target, kernel, iterations=1)
        strict_edge = (dilated - eroded) > 0

        if self.nils_edge_dilation > 0:
            kernel_size = 2 * self.nils_edge_dilation + 1
            kernel_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            edge_mask = cv2.dilate(strict_edge.astype(np.uint8), kernel_dil, iterations=1) > 0
        else:
            edge_mask = strict_edge

        area = np.sum(edge_mask)
        if area == 0:
            return np.zeros_like(intensity_raw)

        # 对强度轻微平滑以减少噪声
        I_smooth = cv2.GaussianBlur(intensity_raw, (3, 3), 0.5)
        gy, gx = np.gradient(I_smooth, self.dy, self.dx)
        grad_mag = np.sqrt(gx**2 + gy**2 + 1e-12)
        safe_I = np.maximum(intensity_raw, 1e-12)

        # 单位方向场
        threshold = 1e-6 * np.max(grad_mag)
        ux = np.where(grad_mag > threshold, gx / grad_mag, 0.0)
        uy = np.where(grad_mag > threshold, gy / grad_mag, 0.0)

        # 散度项
        comp_x = ux / safe_I
        comp_y = uy / safe_I
        dcomp_x_dx = np.gradient(comp_x, self.dx, axis=1)
        dcomp_y_dy = np.gradient(comp_y, self.dy, axis=0)
        divergence = dcomp_x_dx + dcomp_y_dy

        # 灵敏度 dL/dI
        dL_dI = self.lambda_nils * cd * (grad_mag / (safe_I**2) + divergence) * (edge_mask.astype(np.float64) / area)

        # 反向传播
        gradient_nils = np.zeros_like(intensity_raw, dtype=np.complex128)
        for s_val, H_i, A_i in zip(singular_values, eigen_functions, A_i_list):
            dL_dA_i = dL_dI * 2 * s_val * A_i.conj()
            gradient_nils += ifft2(ifftshift(fftshift(fft2(dL_dA_i)) * np.conj(H_i)))
        gradient_nils = np.real(gradient_nils)

        return gradient_nils

    def compute_nils(self, aerial, target, cd=NILS_CD):
        """
        计算目标边缘区域的平均 NILS (使用原始空中像)
        """
        gy, gx = np.gradient(aerial, self.dy, self.dx)
        grad_mag = np.sqrt(gx ** 2 + gy ** 2)

        safe_aerial = np.maximum(aerial, 1e-6)
        log_slope = grad_mag / safe_aerial
        nils_map = cd * log_slope

        binary_target = (target > 0.5).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(binary_target, kernel, iterations=1)
        eroded = cv2.erode(binary_target, kernel, iterations=1)
        strict_edge_mask = (dilated - eroded) > 0

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

        print("Detecting edge regions for constrained optimization...")
        self.update_mask = self._detect_edge_region(target, self.edge_pixel_range)

        if learning_rate is None:
            learning_rate = ILT_OPTIMIZER_CONFIGS[self.optimizer_type]['learning_rate']

        csv_logger = None
        if log_csv:
            csv_logger = OptimizationLogger(log_dir=log_dir)
            total_loss, init_pe, _, _, _, _, _, _ = self._compute_analytical_gradient(mask, target)
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
            'nils': [],
            'learning_rates': [],
            'grad_norms': [],
            'update_region_size': []
        }
        start_time = time.time()

        print(f"Starting Edge-Constrained Base PE Optimization ({max_iterations} iters)...")
        print(f"Edge pixel range: {self.edge_pixel_range}px")
        print(f"Update region: {np.sum(self.update_mask > 0.1) / self.update_mask.size * 100:.1f}% of total area")

        for iteration in range(max_iterations):
            total_loss, pe_loss, epe_loss, gradient, aerial, printed, intensity_raw, A_i_list = \
                self._compute_analytical_gradient(mask, target)

            nils = self.compute_nils(aerial, target, cd=self.nils_cd)
            history['nils'].append(nils)

            masked_gradient = self._apply_update_mask(gradient, self.update_mask)

            active_pixels = np.sum(np.abs(masked_gradient) > 1e-6)
            total_pixels = masked_gradient.size
            update_ratio = active_pixels / total_pixels * 100
            history['update_region_size'].append(update_ratio)

            history['grad_norms'].append(np.linalg.norm(masked_gradient))

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
                    time.time() - start_time,
                    nils=nils
                )

            mask = self._update_with_optimizer(mask, masked_gradient, learning_rate, **optimizer_params)

            if iteration % 20 == 0:
                print(f"Iter {iteration}: PE Loss={pe_loss:.4f}, Best PE={best_pe_loss:.4f}, "
                      f"NILS={nils:.4f}, Update Region={update_ratio:.1f}%, Grad Norm={np.linalg.norm(masked_gradient):.4f}")

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
                                                           canny_low_threshold=50,
                                                           canny_high_threshold=150,
                                                           lambda_nils=0.0,
                                                           nils_cd=NILS_CD,
                                                           nils_edge_dilation=3,
                                                           apply_sharpening=False,
                                                           sharpening_strength=1.0,
                                                           sharpening_sigma=1.0,
                                                           **optimizer_params):
    """
    边缘约束的单PE优化入口函数，支持空间像锐化
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