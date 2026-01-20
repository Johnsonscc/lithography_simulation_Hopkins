import time
import logging
import numpy as np
import torch
from config.parameters import *
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.sparse.linalg import svds
from tqdm import tqdm
from scipy.sparse import lil_matrix, csr_matrix
from utils.optimization_logger import OptimizationLogger

logger = logging.getLogger(__name__)


class ConFIGInverseLithographyOptimizer:
    """
    ConFIG (Conflict-Free Gradient) 优化的逆光刻优化器
    提供基础ConFIG版本和动量ConFIG版本
    """

    def __init__(self, lambda_=LAMBDA, na=NA, sigma=SIGMA,
                 lx=LX, ly=LY, k_svd=ILT_K_SVD, a=A, tr=TR,
                 use_momentum=False, beta_1=0.9, beta_2=0.999):
        # 光学参数
        self.lambda_ = lambda_
        self.na = na
        self.sigma = sigma
        self.lx = lx
        self.ly = ly
        self.k_svd = k_svd

        # 光刻胶参数
        self.a = a
        self.tr = tr

        # ConFIG 配置
        self.use_momentum = use_momentum
        self.beta_1 = beta_1
        self.beta_2 = beta_2

        # ConFIG 操作器
        self.config_operator = None
        self.momentum_operator = None

        # 动量状态
        if use_momentum:
            self._init_momentum_state()

        # 预计算TCC SVD分解
        self.singular_values = None
        self.eigen_functions = None
        self._precompute_tcc_svd()

        logger.info(f"ConFIGInverseLithographyOptimizer initialized (momentum={use_momentum})")

    def _init_momentum_state(self):
        """初始化动量状态"""
        self.momentum_state = {
            'm': None,  # 一阶动量
            's': None,  # 二阶动量
            't': 0,  # 时间步
            't_grads': [0, 0]  # 每个梯度的更新次数（EPE和PE）
        }

    # --- 光学模型函数 ---
    def pupil_response_function(self, fx, fy):
        r = np.sqrt(fx ** 2 + fy ** 2)
        r_max = self.na / self.lambda_
        P = np.where(r < r_max, self.lambda_ ** 2 / (np.pi * (self.na) ** 2), 0)
        return P

    def light_source_function(self, fx, fy):
        r = np.sqrt(fx ** 2 + fy ** 2)
        r_max = self.sigma * self.na / self.lambda_
        J = np.where(r <= r_max,
                     self.lambda_ ** 2 / (np.pi * (self.sigma * self.na) ** 2), 0.0)
        return J

    # --- TCC SVD 预计算 ---
    def _compute_full_tcc_matrix(self, fx, fy, sparsity_threshold=0.001):
        """构建 TCC 矩阵"""
        Lx, Ly = len(fx), len(fy)
        FX, FY = np.meshgrid(fx, fy, indexing='xy')
        J = self.light_source_function(FX, FY)
        P = self.pupil_response_function(FX, FY)
        tcc_kernel = J * P
        TCC_sparse = lil_matrix((Lx * Ly, Lx * Ly), dtype=np.complex128)
        neighborhood_radius = 10

        for i in tqdm(range(Lx), desc="TCC Construction"):
            for j in range(Ly):
                if np.abs(tcc_kernel[i, j]) > sparsity_threshold:
                    for m in range(max(0, i - neighborhood_radius), min(Lx, i + neighborhood_radius + 1)):
                        for n in range(max(0, j - neighborhood_radius), min(Ly, j + neighborhood_radius + 1)):
                            if np.abs(tcc_kernel[m, n]) > sparsity_threshold:
                                idx1 = i * Ly + j
                                idx2 = m * Ly + n
                                TCC_sparse[idx1, idx2] = tcc_kernel[i, j] * np.conj(tcc_kernel[m, n])

        TCC_csr = csr_matrix(TCC_sparse)
        return TCC_csr

    def _svd_of_tcc_matrix(self, TCC_csr, k, Lx, Ly):
        k_actual = min(k, min(TCC_csr.shape) - 1)
        U, S, Vh = svds(TCC_csr, k=k_actual)
        significant_mask = S > (np.max(S) * 0.01)
        S = S[significant_mask]
        U = U[:, significant_mask]
        idx = np.argsort(S)[::-1]
        S = S[idx]
        U = U[:, idx]
        H_functions = [U[:, i].reshape(Lx, Ly) for i in range(len(S))]
        return S, H_functions

    def _precompute_tcc_svd(self):
        max_freq = self.na / self.lambda_
        freq = 2 * max_freq
        fx = np.linspace(-freq, freq, self.lx)
        fy = np.linspace(-freq, freq, self.ly)
        TCC_4d = self._compute_full_tcc_matrix(fx, fy)
        self.singular_values, self.eigen_functions = self._svd_of_tcc_matrix(TCC_4d, self.k_svd, self.lx, self.ly)
        logger.info(f"TCC SVD precomputation completed with {len(self.singular_values)} singular values")

    def photoresist_model(self, intensity):
        return 1 / (1 + np.exp(-self.a * (intensity - self.tr)))

    # --- ConFIG 核心算法（移植自提供的代码）---
    def _get_cos_similarity(self, a, b):
        """计算余弦相似度"""
        return torch.dot(a, b) / (a.norm() * b.norm() + 1e-10)

    def _unit_vector(self, v):
        """单位化向量"""
        norm = v.norm()
        return v / norm if norm > 0 else torch.zeros_like(v)

    def _transfer_coef_double(self, weights, unit_1, unit_2, unit_or1, unit_or2):
        """传输系数计算（双梯度情况）"""
        w1, w2 = weights
        a1 = w1 * (torch.dot(unit_1, unit_or1))
        a2 = w2 * (torch.dot(unit_2, unit_or2))
        return a1, a2

    def _config_combine_gradients(self, grad_epe, grad_pe, losses=None):
        """
        使用完整的ConFIG算法融合EPE和PE梯度
        """
        # 转换为PyTorch tensor
        g1 = torch.tensor(grad_epe.flatten(), dtype=torch.float64)
        g2 = torch.tensor(grad_pe.flatten(), dtype=torch.float64)

        # 计算范数和单位向量
        norm_1 = g1.norm()
        norm_2 = g2.norm()

        if norm_1 < 1e-10 or norm_2 < 1e-10:
            # 如果有一个梯度很小，直接返回加权和
            return 0.5 * grad_epe + 0.5 * grad_pe

        unit_1 = g1 / norm_1
        unit_2 = g2 / norm_2

        # 计算余弦相似度
        cos_angle = self._get_cos_similarity(g1, g2)

        # 计算正交分量
        or_2 = g1 - norm_1 * cos_angle * unit_2
        or_1 = g2 - norm_2 * cos_angle * unit_1

        unit_or1 = self._unit_vector(or_1)
        unit_or2 = self._unit_vector(or_2)

        # 等权重
        weights = torch.tensor([0.5, 0.5], dtype=torch.float64)

        # 计算系数
        coef_1, coef_2 = self._transfer_coef_double(
            weights, unit_1, unit_2, unit_or1, unit_or2
        )

        # 最优方向
        best_direction = coef_1 * unit_or1 + coef_2 * unit_or2

        # 投影长度重缩放
        proj1 = torch.dot(g1, best_direction) / best_direction.norm()
        proj2 = torch.dot(g2, best_direction) / best_direction.norm()
        final_length = 0.5 * (proj1 + proj2)
        final_direction = best_direction * final_length / best_direction.norm()

        # 转换回numpy并reshape
        return final_direction.numpy().reshape(grad_epe.shape)

    def _momentum_config_combine_gradients(self, grad_epe, grad_pe, losses=None):
        """
        使用动量ConFIG算法融合梯度
        """
        # 转换为PyTorch tensor
        g1 = torch.tensor(grad_epe.flatten(), dtype=torch.float64)
        g2 = torch.tensor(grad_pe.flatten(), dtype=torch.float64)

        # 计算范数和单位向量
        norm_1 = g1.norm()
        norm_2 = g2.norm()

        if norm_1 < 1e-10 or norm_2 < 1e-10:
            return 0.5 * grad_epe + 0.5 * grad_pe

        # 更新一阶动量
        if self.momentum_state['m'] is None:
            self.momentum_state['m'] = [torch.zeros_like(g1), torch.zeros_like(g2)]

        self.momentum_state['t'] += 1
        self.momentum_state['t_grads'][0] += 1
        self.momentum_state['t_grads'][1] += 1

        # 更新动量
        self.momentum_state['m'][0] = self.beta_1 * self.momentum_state['m'][0] + (1 - self.beta_1) * g1
        self.momentum_state['m'][1] = self.beta_1 * self.momentum_state['m'][1] + (1 - self.beta_1) * g2

        # 计算偏置修正的动量
        m1_hat = self.momentum_state['m'][0] / (1 - self.beta_1 ** self.momentum_state['t_grads'][0])
        m2_hat = self.momentum_state['m'][1] / (1 - self.beta_1 ** self.momentum_state['t_grads'][1])

        # 将修正后的动量堆叠
        m_hats = torch.stack([m1_hat, m2_hat], dim=0)

        # 使用基础ConFIG融合动量向量
        unit_vectors = m_hats / m_hats.norm(dim=1).unsqueeze(1)
        unit_vectors = torch.nan_to_num(unit_vectors, 0)

        # 等权重
        weights = torch.tensor([0.5, 0.5], dtype=torch.float64)

        # 使用最小二乘法求解最优方向
        best_direction = torch.linalg.lstsq(unit_vectors, weights).solution

        # 计算伪梯度（用于二阶动量估计）
        fake_m = best_direction * (1 - self.beta_1 ** self.momentum_state['t'])
        fake_grad = (fake_m - self.beta_1 * self.momentum_state.get('fake_m', torch.zeros_like(fake_m))) / (
                    1 - self.beta_1)
        self.momentum_state['fake_m'] = fake_m

        # 更新二阶动量
        if self.momentum_state['s'] is None:
            self.momentum_state['s'] = torch.zeros_like(best_direction)

        self.momentum_state['s'] = self.beta_2 * self.momentum_state['s'] + (1 - self.beta_2) * (fake_grad ** 2)
        s_hat = self.momentum_state['s'] / (1 - self.beta_2 ** self.momentum_state['t'])

        # 最终梯度
        final_grad = best_direction / (torch.sqrt(s_hat) + 1e-8)

        # 投影长度重缩放
        proj1 = torch.dot(g1, final_grad) / final_grad.norm()
        proj2 = torch.dot(g2, final_grad) / final_grad.norm()
        final_length = 0.5 * (proj1 + proj2)
        final_direction = final_grad * final_length / final_grad.norm()

        return final_direction.numpy().reshape(grad_epe.shape)

    # --- 核心：梯度计算（分别计算EPE和PE梯度）---
    def _compute_gradients(self, mask, target, epsilon=1e-10):
        """
        分别计算EPE和PE的梯度
        """
        # 1. 前向传播
        M_fft = fftshift(fft2(mask))
        A_i_list = []
        intensity = np.zeros((self.lx, self.ly), dtype=np.float64)

        for i, (s_val, H_i) in enumerate(zip(self.singular_values, self.eigen_functions)):
            A_i_fft = M_fft * H_i
            A_i = ifft2(ifftshift(A_i_fft))
            intensity += s_val * (np.abs(A_i) ** 2)
            A_i_list.append(A_i)

        # 归一化光强
        intensity_min = np.min(intensity)
        intensity_max = np.max(intensity)
        denom = intensity_max - intensity_min if (intensity_max - intensity_min) > epsilon else 1.0
        intensity_norm = (intensity - intensity_min) / denom

        # 光刻胶显影P
        P = self.photoresist_model(intensity_norm)

        # 2. 计算损失
        pe_loss_val = np.sum((P - target) ** 2)

        # 计算边缘权重W（用于EPE）
        grad_y, grad_x = np.gradient(P)
        W = np.sqrt(grad_x ** 2 + grad_y ** 2 + epsilon)
        if np.max(W) > 0:
            W = W / np.max(W)

        epe_loss_val = np.sum(((P - target) ** 2) * W)

        # 3. 计算PE梯度（全局保真度）
        dJ_pe_dP = 2 * (P - target)  # PE: 均匀权重
        dP_dI_norm = self.a * P * (1 - P)
        dI_norm_dI = 1.0 / denom
        dP_dI = dP_dI_norm * dI_norm_dI
        dJ_pe_dI = dJ_pe_dP * dP_dI

        grad_pe = np.zeros_like(mask, dtype=np.complex128)
        for i, (s_val, H_i, A_i) in enumerate(zip(
                self.singular_values, self.eigen_functions, A_i_list)):
            dI_dA_i = 2 * s_val * A_i.conj()
            dJ_dA_i = dJ_pe_dI * dI_dA_i
            dJ_dA_i_fft = fftshift(fft2(dJ_dA_i))
            grad_pe += ifft2(ifftshift(dJ_dA_i_fft * np.conj(H_i)))

        # 4. 计算EPE梯度（边缘精度）
        dJ_epe_dP = 2 * (P - target) * W  # EPE: 边缘加权
        dJ_epe_dI = dJ_epe_dP * dP_dI

        grad_epe = np.zeros_like(mask, dtype=np.complex128)
        for i, (s_val, H_i, A_i) in enumerate(zip(
                self.singular_values, self.eigen_functions, A_i_list)):
            dI_dA_i = 2 * s_val * A_i.conj()
            dJ_dA_i = dJ_epe_dI * dI_dA_i
            dJ_dA_i_fft = fftshift(fft2(dJ_dA_i))
            grad_epe += ifft2(ifftshift(dJ_dA_i_fft * np.conj(H_i)))

        return (epe_loss_val, pe_loss_val,
                np.real(grad_epe), np.real(grad_pe),
                intensity_norm, P)

    # --- 优化主循环 ---
    def optimize(self, initial_mask, target, learning_rate=0.01,
                 max_iterations=ILT_MAX_ITERATIONS,
                 log_csv=True,
                 log_dir="logs",
                 experiment_tag=""):

        mask = initial_mask.copy()

        # 初始化CSV日志
        csv_logger = None
        if log_csv:
            target_shape = f"{target.shape[0]}x{target.shape[1]}"
            if experiment_tag:
                target_shape = f"{target_shape}_{experiment_tag}"

            # 计算初始损失
            init_epe, init_pe, _, _, _, _ = self._compute_gradients(initial_mask, target)

            csv_logger = OptimizationLogger(log_dir=log_dir)
            csv_logger.start_logging(
                optimizer_type="ConFIG" + ("+Momentum" if self.use_momentum else ""),
                loss_type="EPE+PE (ConFIG fused)",
                learning_rate=learning_rate,
                initial_loss=init_epe,
                target_shape=target_shape,
                config_params={"use_momentum": self.use_momentum, "beta_1": self.beta_1, "beta_2": self.beta_2}
            )

        history = {
            'pe_loss': [],
            'epe_loss': [],
            'loss': [],
            'grad_norms': [],
            'masks': [],
            'aerial_images': [],
            'printed_images': [],
            'learning_rates': [],
            'grad_conflicts': [],  # 记录梯度冲突程度
            'momentum_norms': [],  # 记录动量范数
            'csv_log_path': csv_logger.filepath if csv_logger else None
        }

        if self.use_momentum:
            print("Starting Momentum-ConFIG Optimization...")
            print(f"Config: Momentum ConFIG (beta1={self.beta_1}, beta2={self.beta_2})")
        else:
            print("Starting Base ConFIG Optimization...")
            print("Config: Base ConFIG (no momentum)")

        best_mask = mask.copy()
        best_epe_loss = float('inf')

        start_time = time.time()

        for iteration in range(max_iterations):
            # 计算EPE和PE梯度
            epe_loss_val, pe_loss_val, grad_epe, grad_pe, aerial_image, printed_image = \
                self._compute_gradients(mask, target)

            # 使用ConFIG融合梯度
            if self.use_momentum:
                combined_gradient = self._momentum_config_combine_gradients(grad_epe, grad_pe)
                # 记录动量范数
                if self.momentum_state['m'] is not None:
                    momentum_norm = sum(m.norm().item() for m in self.momentum_state['m']) / 2
                    history['momentum_norms'].append(momentum_norm)
            else:
                combined_gradient = self._config_combine_gradients(grad_epe, grad_pe)

            # 计算梯度冲突度（余弦相似度）
            grad_epe_flat = grad_epe.flatten()
            grad_pe_flat = grad_pe.flatten()
            norm_epe = np.linalg.norm(grad_epe_flat)
            norm_pe = np.linalg.norm(grad_pe_flat)

            if norm_epe > 0 and norm_pe > 0:
                cos_sim = np.dot(grad_epe_flat, grad_pe_flat) / (norm_epe * norm_pe)
                conflict_degree = 1 - abs(cos_sim)  # 0: 无冲突, 1: 完全冲突
            else:
                conflict_degree = 0

            # 记录历史
            history['epe_loss'].append(epe_loss_val)
            history['pe_loss'].append(pe_loss_val)
            history['loss'].append(epe_loss_val)  # 主要看EPE
            history['grad_conflicts'].append(conflict_degree)
            history['grad_norms'].append(np.linalg.norm(combined_gradient))

            # 记录最佳结果
            if epe_loss_val < best_epe_loss:
                best_epe_loss = epe_loss_val
                best_mask = mask.copy()

            if csv_logger:
                csv_logger.log_iteration(
                    iteration=iteration,
                    loss=epe_loss_val,
                    gradient=combined_gradient,
                    mask=mask,
                    optimizer_state=self.momentum_state if self.use_momentum else {},
                    time_elapsed=time.time() - start_time
                )

            # 更新mask
            mask = mask - learning_rate * combined_gradient
            mask = np.clip(mask, 0, 1)

            history['learning_rates'].append(learning_rate)

            if iteration % 20 == 0 or iteration == max_iterations - 1:
                history['masks'].append(mask.copy())
                history['aerial_images'].append(aerial_image)
                history['printed_images'].append(printed_image)

            if iteration % 10 == 0:
                momentum_info = ""
                if self.use_momentum and 'momentum_norms' in history and history['momentum_norms']:
                    momentum_info = f", Momentum={history['momentum_norms'][-1]:.4f}"

                print(f"Iter {iteration:3d}: EPE={epe_loss_val:.4f}, PE={pe_loss_val:.4f}, "
                      f"Conflict={conflict_degree:.3f}, Grad={np.linalg.norm(combined_gradient):.4f}{momentum_info}")

        if csv_logger:
            csv_logger.close()

        total_time = time.time() - start_time
        print(f"ConFIG Optimization completed in {total_time:.2f}s. Best EPE: {best_epe_loss:.4f}")

        return best_mask, history


# --- 对外暴露的优化函数 ---
def inverse_lithography_optimization_config(initial_mask, target_image,
                                            learning_rate=0.01,
                                            max_iterations=ILT_MAX_ITERATIONS,
                                            use_momentum=False,
                                            beta_1=0.9,
                                            beta_2=0.999,
                                            **config_params):
    """
    ConFIG优化的逆光刻优化入口函数
    """
    optimizer = ConFIGInverseLithographyOptimizer(
        use_momentum=use_momentum,
        beta_1=beta_1,
        beta_2=beta_2
    )

    optimized_mask, history = optimizer.optimize(
        initial_mask=initial_mask,
        target=target_image,
        learning_rate=learning_rate,
        max_iterations=max_iterations,
        **config_params
    )

    return optimized_mask, history