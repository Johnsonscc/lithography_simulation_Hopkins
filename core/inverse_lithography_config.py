import time
import logging
import numpy as np
import torch
import cv2
from config.parameters import *
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.sparse.linalg import svds
from tqdm import tqdm
from scipy.sparse import lil_matrix, csr_matrix
from utils.optimization_logger import OptimizationLogger

logger = logging.getLogger(__name__)


class BaseEdgeConFIGOptimizer:
    """
    基础边缘约束ConFIG优化器
    使用固定权重融合PE和EPE梯度
    """

    def __init__(self, lambda_=LAMBDA, na=NA, sigma=SIGMA,
                 lx=LX, ly=LY, k_svd=ILT_K_SVD, a=A, tr=TR,
                 edge_pixel_range=5):
        # 光学参数
        self.lambda_ = lambda_
        self.na = na
        self.sigma = sigma
        self.lx = lx
        self.ly = ly
        self.k_svd = k_svd
        self.a = a
        self.tr = tr

        # 边缘约束参数
        self.edge_pixel_range = edge_pixel_range
        self.edge_mask = None

        # 固定权重参数
        self.pe_weight = 0
        self.epe_weight = 1

        # 预计算
        self.singular_values = None
        self.eigen_functions = None
        self._precompute_tcc_svd()

        logger.info(f"BaseEdgeConFIGOptimizer initialized (edge={edge_pixel_range})")

    def _detect_edge_region(self, target):
        """
        基于目标的稳定边缘检测
        """
        binary_target = (target > 0.5).astype(np.uint8)

        # 方法1: Canny边缘检测
        edges = cv2.Canny(binary_target * 255, 30, 100)
        edges = edges.astype(np.float32) / 255.0

        # 方法2: 梯度幅值
        grad_y, grad_x = np.gradient(binary_target.astype(np.float32))
        grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        grad_edges = (grad_magnitude > 0.1).astype(np.float32)

        # 组合两种检测方法
        combined_edges = np.clip(edges + 0.5 * grad_edges, 0, 1)

        # 扩展边缘区域
        if self.edge_pixel_range > 0:
            kernel_size = 2 * self.edge_pixel_range + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            edge_mask = cv2.dilate(combined_edges, kernel, iterations=1)
        else:
            edge_mask = combined_edges

        # 平滑过渡
        edge_mask = cv2.GaussianBlur(edge_mask, (7, 7), 1.5)

        return edge_mask

    def _apply_edge_constraint(self, gradient, edge_mask):
        """
        应用边缘约束
        """
        return gradient * edge_mask

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
        """TCC SVD预计算"""
        max_freq = self.na / self.lambda_
        freq = 2 * max_freq
        fx = np.linspace(-freq, freq, self.lx)
        fy = np.linspace(-freq, freq, self.ly)
        TCC_4d = self._compute_full_tcc_matrix(fx, fy)
        self.singular_values, self.eigen_functions = self._svd_of_tcc_matrix(TCC_4d, self.k_svd)

    def photoresist_model(self, intensity):
        return 1 / (1 + np.exp(-self.a * (intensity - self.tr)))

    def _compute_gradients(self, mask, target):
        """
        梯度计算
        """
        # 前向传播
        M_fft = fftshift(fft2(mask))
        A_i_list = []
        intensity = np.zeros((self.lx, self.ly), dtype=np.float64)

        for i, (s_val, H_i) in enumerate(zip(self.singular_values, self.eigen_functions)):
            A_i_fft = M_fft * H_i
            A_i = ifft2(ifftshift(A_i_fft))
            intensity += s_val * (np.abs(A_i) ** 2)
            A_i_list.append(A_i)

        # 归一化
        intensity_min = np.min(intensity)
        intensity_max = np.max(intensity)
        intensity_range = intensity_max - intensity_min

        if intensity_range < 1e-10:
            intensity_norm = np.zeros_like(intensity)
        else:
            intensity_norm = (intensity - intensity_min) / intensity_range
            intensity_norm = np.clip(intensity_norm, -10, 10)

        # 光刻胶显影
        exponent = -self.a * (intensity_norm - self.tr)
        exponent = np.clip(exponent, -100, 100)
        P = 1 / (1 + np.exp(exponent))

        # 计算PE损失
        pe_loss = np.sum((P - target) ** 2)

        ### 计算边缘权重
        target_edges = self._detect_edge_region(target)

        # EPE损失
        epe_loss = np.sum(((P - target) ** 2) * target_edges)

        # 计算PE梯度
        dP_dI = self.a * P * (1 - P)
        dJ_pe_dP = 2 * (P - target)
        dJ_pe_dI = dJ_pe_dP * dP_dI / intensity_range if intensity_range > 1e-10 else dJ_pe_dP * dP_dI

        grad_pe = np.zeros_like(mask, dtype=np.complex128)
        for i, (s_val, H_i, A_i) in enumerate(zip(
                self.singular_values, self.eigen_functions, A_i_list)):
            dI_dA_i = 2 * s_val * A_i.conj()
            dJ_dA_i = dJ_pe_dI * dI_dA_i
            dJ_dA_i_fft = fftshift(fft2(dJ_dA_i))
            grad_pe += ifft2(ifftshift(dJ_dA_i_fft * np.conj(H_i)))

        # 计算EPE梯度
        dJ_epe_dP = 2 * (P - target) * target_edges
        dJ_epe_dI = dJ_epe_dP * dP_dI / intensity_range if intensity_range > 1e-10 else dJ_epe_dP * dP_dI

        grad_epe = np.zeros_like(mask, dtype=np.complex128)
        for i, (s_val, H_i, A_i) in enumerate(zip(
                self.singular_values, self.eigen_functions, A_i_list)):
            dI_dA_i = 2 * s_val * A_i.conj()
            dJ_dA_i = dJ_epe_dI * dI_dA_i
            dJ_dA_i_fft = fftshift(fft2(dJ_dA_i))
            grad_epe += ifft2(ifftshift(dJ_dA_i_fft * np.conj(H_i)))

        return (pe_loss, epe_loss,
                np.real(grad_pe), np.real(grad_epe),
                intensity_norm, P)

    def optimize(self, initial_mask, target, learning_rate=0.1,
                 max_iterations=ILT_MAX_ITERATIONS,
                 log_csv=True, log_dir="logs", experiment_tag=""):

        mask = initial_mask.copy()

        # 初始化边缘掩膜
        self.edge_mask = self._detect_edge_region(target)
        update_ratio = np.sum(self.edge_mask > 0.1) / self.edge_mask.size * 100
        print(f"Edge region: {update_ratio:.1f}% of total area")

        # 初始化
        best_mask = mask.copy()
        best_combined_loss = float('inf')

        # 日志
        csv_logger = None
        if log_csv:
            csv_logger = OptimizationLogger(log_dir=log_dir)
            init_pe, init_epe, _, _, _, _ = self._compute_gradients(mask, target)
            csv_logger.start_logging(
                optimizer_type="Base-Edge-ConFIG",
                loss_type="PE+EPE (Fixed Weights)",
                learning_rate=learning_rate,
                initial_loss=init_pe + init_epe,
                target_shape=f"{target.shape}",
                config_params={
                    "edge_pixel_range": self.edge_pixel_range,
                    "pe_weight": self.pe_weight,
                    "epe_weight": self.epe_weight
                }
            )

        history = {
            'pe_loss': [], 'epe_loss': [], 'combined_loss': [],
            'grad_norms': []
        }

        start_time = time.time()

        print(f"Starting Base-Edge-ConFIG Optimization ({max_iterations} iterations)")
        print(f"Configuration: edge_range={self.edge_pixel_range}")

        for iteration in range(max_iterations):
            # 计算梯度
            pe_loss, epe_loss, grad_pe, grad_epe, aerial, printed = \
                self._compute_gradients(mask, target)

            # 固定权重融合梯度
            combined_gradient = self.pe_weight * grad_pe + self.epe_weight * grad_epe

            # 应用边缘约束
            edge_constrained_gradient = self._apply_edge_constraint(combined_gradient, self.edge_mask)

            # 计算组合损失
            combined_loss = pe_loss * self.pe_weight + epe_loss * self.epe_weight

            # 记录历史
            history['pe_loss'].append(pe_loss)
            history['epe_loss'].append(epe_loss)
            history['combined_loss'].append(combined_loss)
            history['grad_norms'].append(np.linalg.norm(edge_constrained_gradient))

            # 保存最佳结果
            if combined_loss < best_combined_loss:
                best_combined_loss = combined_loss
                best_mask = mask.copy()

            # 日志记录
            if csv_logger:
                csv_logger.log_iteration(
                    iteration=iteration,
                    loss=combined_loss,
                    gradient=edge_constrained_gradient,
                    mask=mask,
                    optimizer_state={
                        "pe_weight": self.pe_weight,
                        "epe_weight": self.epe_weight
                    },
                    time_elapsed=time.time() - start_time
                )

            # 更新掩膜
            mask = mask - learning_rate * edge_constrained_gradient
            mask = np.clip(mask, 0, 1)

            # 进度输出
            if iteration % 10 == 0:
                active_pixels = np.sum(np.abs(edge_constrained_gradient) > 1e-6)
                update_ratio_current = active_pixels / edge_constrained_gradient.size * 100

                print(f"Iter {iteration:3d}: "
                      f"PE={pe_loss:.0f}, EPE={epe_loss:.0f}, "
                      f"Total={combined_loss:.0f}, "
                      f"Update={update_ratio_current:.1f}%")

        if csv_logger:
            csv_logger.close()

        total_time = time.time() - start_time

        print(f"\n{'=' * 60}")
        print("BASE-EDGE-CONFIG OPTIMIZATION COMPLETED")
        print(f"{'=' * 60}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Best combined loss: {best_combined_loss:.2f}")
        print(f"PE: {history['pe_loss'][0]:.1f} -> {history['pe_loss'][-1]:.1f}")
        print(f"EPE: {history['epe_loss'][0]:.1f} -> {history['epe_loss'][-1]:.1f}")
        print(f"{'=' * 60}")

        return best_mask, history, self.edge_mask


class MomentumEdgeConFIGOptimizer(BaseEdgeConFIGOptimizer):
    """
    动量边缘约束ConFIG优化器（带梯度归一化）
    继承基础版本，添加方向动量，并在加权前对 PE 和 EPE 梯度进行量级归一化
    """

    def __init__(self, lambda_=LAMBDA, na=NA, sigma=SIGMA,
                 lx=LX, ly=LY, k_svd=ILT_K_SVD, a=A, tr=TR,
                 edge_pixel_range=5, momentum_beta=0.95):
        super().__init__(lambda_, na, sigma, lx, ly, k_svd, a, tr, edge_pixel_range)

        # 动量参数
        self.momentum_beta = momentum_beta
        self.direction_history = []

        logger.info(f"MomentumEdgeConFIGOptimizer initialized (edge={edge_pixel_range}, beta={momentum_beta})")

    def _apply_direction_momentum(self, gradient):
        """
        应用方向动量（只调整方向，不改变幅度）
        """
        if len(self.direction_history) == 0:
            self.direction_history.append(gradient)
            return gradient

        # 计算当前方向
        gradient_norm = np.linalg.norm(gradient)
        if gradient_norm < 1e-10:
            return gradient

        current_direction = gradient / gradient_norm

        # 计算历史平均方向
        avg_direction = np.mean(self.direction_history[-5:], axis=0) if len(self.direction_history) >= 5 else \
        self.direction_history[-1]
        avg_direction_norm = np.linalg.norm(avg_direction)
        if avg_direction_norm > 0:
            avg_direction = avg_direction / avg_direction_norm

        # 动量混合
        mixed_direction = (
                (1 - self.momentum_beta) * current_direction +
                self.momentum_beta * avg_direction
        )
        mixed_direction_norm = np.linalg.norm(mixed_direction)
        if mixed_direction_norm > 0:
            mixed_direction = mixed_direction / mixed_direction_norm

        # 保持原始梯度幅度
        final_gradient = mixed_direction * gradient_norm

        # 更新历史
        self.direction_history.append(current_direction)
        if len(self.direction_history) > 10:
            self.direction_history.pop(0)

        return final_gradient

    def optimize(self, initial_mask, target, learning_rate=0.01,
                 max_iterations=ILT_MAX_ITERATIONS,
                 log_csv=True, log_dir="logs", experiment_tag=""):

        mask = initial_mask.copy()

        # 初始化边缘掩膜
        self.edge_mask = self._detect_edge_region(target)
        update_ratio = np.sum(self.edge_mask > 0.1) / self.edge_mask.size * 100
        print(f"Edge region: {update_ratio:.1f}% of total area")

        # 初始化
        best_mask = mask.copy()
        best_combined_loss = float('inf')
        self.direction_history = []

        # 日志
        csv_logger = None
        if log_csv:
            csv_logger = OptimizationLogger(log_dir=log_dir)
            init_pe, init_epe, _, _, _, _ = self._compute_gradients(mask, target)
            csv_logger.start_logging(
                optimizer_type="Momentum-Edge-ConFIG",
                loss_type="PE+EPE (Fixed Weights + Momentum + Gradient Norm)",
                learning_rate=learning_rate,
                initial_loss=init_pe + init_epe,
                target_shape=f"{target.shape}",
                config_params={
                    "edge_pixel_range": self.edge_pixel_range,
                    "pe_weight": self.pe_weight,
                    "epe_weight": self.epe_weight,
                    "momentum_beta": self.momentum_beta
                }
            )

        history = {
            'pe_loss': [], 'epe_loss': [], 'combined_loss': [],
            'grad_norms': []
        }

        start_time = time.time()

        print(f"Starting Momentum-Edge-ConFIG Optimization ({max_iterations} iterations)")
        print(f"Configuration: edge_range={self.edge_pixel_range}, momentum_beta={self.momentum_beta}")

        for iteration in range(max_iterations):
            # 计算梯度
            pe_loss, epe_loss, grad_pe, grad_epe, aerial, printed = \
                self._compute_gradients(mask, target)

            # ========== 新增：梯度归一化（使 grad_pe 与 grad_epe 量级一致）==========
            grad_pe_norm = np.linalg.norm(grad_pe)
            grad_epe_norm = np.linalg.norm(grad_epe)
            eps = 1e-8
            if grad_pe_norm > eps and grad_epe_norm > eps:
                # 将 grad_epe 缩放到与 grad_pe 相同的 L2 范数
                scale = grad_pe_norm / grad_epe_norm
                grad_epe = grad_epe * scale
            # 若某一梯度为零，则不进行缩放（保留原值）
            # ==================================================================

            # 固定权重融合梯度
            combined_gradient = self.pe_weight * grad_pe + self.epe_weight * grad_epe

            # 应用方向动量
            momentum_gradient = self._apply_direction_momentum(combined_gradient)

            # 应用边缘约束
            edge_constrained_gradient = self._apply_edge_constraint(momentum_gradient, self.edge_mask)

            # 计算组合损失（注意：损失值未缩放，仍按原始权重计算）
            combined_loss = pe_loss * self.pe_weight + epe_loss * self.epe_weight

            # 记录历史
            history['pe_loss'].append(pe_loss)
            history['epe_loss'].append(epe_loss)
            history['combined_loss'].append(combined_loss)
            history['grad_norms'].append(np.linalg.norm(edge_constrained_gradient))

            # 保存最佳结果
            if combined_loss < best_combined_loss:
                best_combined_loss = combined_loss
                best_mask = mask.copy()

            # 日志记录
            if csv_logger:
                csv_logger.log_iteration(
                    iteration=iteration,
                    loss=combined_loss,
                    gradient=edge_constrained_gradient,
                    mask=mask,
                    optimizer_state={
                        "pe_weight": self.pe_weight,
                        "epe_weight": self.epe_weight,
                        "momentum_beta": self.momentum_beta
                    },
                    time_elapsed=time.time() - start_time
                )

            # 更新掩膜
            mask = mask - learning_rate * edge_constrained_gradient
            mask = np.clip(mask, 0, 1)

            # 进度输出
            if iteration % 10 == 0:
                active_pixels = np.sum(np.abs(edge_constrained_gradient) > 1e-6)
                update_ratio_current = active_pixels / edge_constrained_gradient.size * 100

                print(f"Iter {iteration:3d}: "
                      f"PE={pe_loss:.0f}, EPE={epe_loss:.0f}, "
                      f"Total={combined_loss:.0f}, "
                      f"Update={update_ratio_current:.1f}%")

        if csv_logger:
            csv_logger.close()

        total_time = time.time() - start_time

        print(f"\n{'=' * 60}")
        print("MOMENTUM-EDGE-CONFIG OPTIMIZATION COMPLETED")
        print(f"{'=' * 60}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Best combined loss: {best_combined_loss:.2f}")
        print(f"PE: {history['pe_loss'][0]:.1f} -> {history['pe_loss'][-1]:.1f}")
        print(f"EPE: {history['epe_loss'][0]:.1f} -> {history['epe_loss'][-1]:.1f}")
        print(f"{'=' * 60}")

        return best_mask, history, self.edge_mask


# 入口函数保持不变
def inverse_lithography_optimization_momentum_edge_config(initial_mask, target_image,
                                                          learning_rate=0.1,
                                                          max_iterations=ILT_MAX_ITERATIONS,
                                                          edge_pixel_range=5,
                                                          **config_params):
    """
    动量边缘约束ConFIG优化入口函数（包含梯度归一化）
    """
    optimizer = MomentumEdgeConFIGOptimizer(
        edge_pixel_range=edge_pixel_range,
        momentum_beta=0.8
    )

    optimized_mask, history, edge_mask = optimizer.optimize(
        initial_mask=initial_mask,
        target=target_image,
        learning_rate=learning_rate,
        max_iterations=max_iterations,
        **config_params
    )

    return optimized_mask, history, edge_mask