import time
import logging
import numpy as np
from config.parameters import *
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.sparse.linalg import svds
from tqdm import tqdm
from scipy.sparse import lil_matrix, csr_matrix
from utils.optimization_logger import OptimizationLogger

logger = logging.getLogger(__name__)


class CombInverseLithographyOptimizer:
    """
    联合优化器 (Combined Optimizer)：
    在一个阶段内同时优化 EPE (边缘精度) 和 PE (像素保真度)。
    通过 comb_weight 参数控制两者的权重平衡。
    """

    def __init__(self, lambda_=LAMBDA, na=NA, sigma=SIGMA,
                 lx=LX, ly=LY, k_svd=ILT_K_SVD, a=A, tr=TR,
                 optimizer_type=OPTIMIZER_TYPE):
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

        # 优化器配置
        self.optimizer_type = optimizer_type
        self.optimizer_state = {}

        # 预计算TCC SVD分解
        self.singular_values = None
        self.eigen_functions = None
        self._precompute_tcc_svd()

        logger.info(f"CombInverseLithographyOptimizer initialized with {optimizer_type}")

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

    # --- TCC SVD 预计算 (Comb 版本) ---
    def _compute_full_tcc_matrix_comb(self, fx, fy, sparsity_threshold=0.001):
        """构建 Comb TCC 矩阵"""
        Lx, Ly = len(fx), len(fy)
        FX, FY = np.meshgrid(fx, fy, indexing='xy')
        J = self.light_source_function(FX, FY)
        P = self.pupil_response_function(FX, FY)
        tcc_kernel = J * P
        TCC_sparse = lil_matrix((Lx * Ly, Lx * Ly), dtype=np.complex128)
        neighborhood_radius = 10

        # 修改描述为 Comb TCC
        for i in tqdm(range(Lx), desc="Comb TCC Construction"):
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
        # 调用 Comb 版本的 TCC 构建
        TCC_4d = self._compute_full_tcc_matrix_comb(fx, fy)
        self.singular_values, self.eigen_functions = self._svd_of_tcc_matrix(TCC_4d, self.k_svd, self.lx, self.ly)
        print(f"Comb TCC SVD precomputation completed with {len(self.singular_values)} singular values")

    def photoresist_model(self, intensity):
        return 1 / (1 + np.exp(-self.a * (intensity - self.tr)))

    # --- 优化器状态管理 ---
    def _initialize_optimizer_state(self, mask_shape):
        if self.optimizer_type == 'sgd':
            self.optimizer_state = {}
        elif self.optimizer_type == 'momentum':
            self.optimizer_state = {'velocity': np.zeros(mask_shape, dtype=np.float64)}
        elif self.optimizer_type == 'rmsprop':
            self.optimizer_state = {'square_avg': np.zeros(mask_shape, dtype=np.float64)}
        elif self.optimizer_type == 'cg':
            self.optimizer_state = {'prev_grad': None, 'direction': None, 't': 0}
        elif self.optimizer_type == 'adam':
            self.optimizer_state = {'m': np.zeros(mask_shape, dtype=np.float64),
                                    'v': np.zeros(mask_shape, dtype=np.float64),
                                    't': 0}

    # --- 优化器更新逻辑 ---
    def _update_with_optimizer(self, mask, gradient, learning_rate, **optimizer_params):
        if self.optimizer_type == 'sgd':
            new_mask = mask - learning_rate * gradient
        elif self.optimizer_type == 'momentum':
            momentum = optimizer_params.get('momentum', 0.9)
            velocity = self.optimizer_state['velocity']
            velocity = momentum * velocity - learning_rate * gradient
            self.optimizer_state['velocity'] = velocity
            new_mask = mask + velocity
        elif self.optimizer_type == 'rmsprop':
            decay_rate = optimizer_params.get('decay_rate', 0.99)
            epsilon = optimizer_params.get('epsilon', 1e-8)
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
            beta1 = optimizer_params.get('beta1', 0.9)
            beta2 = optimizer_params.get('beta2', 0.999)
            epsilon = optimizer_params.get('epsilon', 1e-8)
            adam_lambda = optimizer_params.get('lambda', 1 - 1e-8)
            m = self.optimizer_state['m']
            v = self.optimizer_state['v']
            t = self.optimizer_state['t'] + 1
            beta1_t = beta1 * (adam_lambda ** (t - 1))
            m = beta1_t * m + (1 - beta1) * gradient
            v = beta2 * v + (1 - beta2) * (gradient ** 2)
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            new_mask = mask - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
            self.optimizer_state['m'] = m
            self.optimizer_state['v'] = v
            self.optimizer_state['t'] = t
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_type}")

        return np.clip(new_mask, 0, 1)

    # --- 核心：联合梯度计算 ---
    def _compute_comb_gradient(self, mask, target, comb_weight=0.85, epsilon=1e-10):
        """
        计算联合梯度 (Combined Gradient)
        comb_weight: 控制 EPE (边缘) 和 PE (全局) 的权重比例。
                    comb_weight 越大，越重视边缘对齐。
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

        # 光刻胶显影 P
        P = self.photoresist_model(intensity_norm)

        # 2. 计算各个损失 (仅用于日志记录)
        pe_loss = np.sum((P - target) ** 2)

        # 计算边缘权重 W (用于 EPE)
        grad_y, grad_x = np.gradient(P)
        W = np.sqrt(grad_x ** 2 + grad_y ** 2 + epsilon)
        if np.max(W) > 0:
            W = W / np.max(W)

        epe_loss = np.sum(((P - target) ** 2) * W)

        # 3. 计算联合梯度权重 (Comb Weight Matrix)
        # 混合逻辑：Comb_Weight = comb_weight * W (边缘) + (1 - comb_weight) * 1 (全局)
        # 这样在边缘处梯度被加强，在背景处梯度被保留但较小 (防止形状崩坏)
        combined_weight_matrix = comb_weight * W + (1.0 - comb_weight)

        # dJ_comb / dP
        dJ_dP = 2 * (P - target) * combined_weight_matrix

        # 4. 链式法则反向传播
        dP_dI_norm = self.a * P * (1 - P)
        dI_norm_dI = 1.0 / denom
        dP_dI = dP_dI_norm * dI_norm_dI

        # dJ_comb / dI
        dJ_dI = dJ_dP * dP_dI

        gradient = np.zeros_like(mask, dtype=np.complex128)

        for i, (s_val, H_i, A_i) in enumerate(zip(
                self.singular_values, self.eigen_functions, A_i_list)):
            dI_dA_i = 2 * s_val * A_i.conj()
            dJ_dA_i = dJ_dI * dI_dA_i

            dJ_dA_i_fft = fftshift(fft2(dJ_dA_i))
            gradient += ifft2(ifftshift(dJ_dA_i_fft * np.conj(H_i)))

        return epe_loss, pe_loss, np.real(gradient), intensity_norm, P

    # --- 优化主循环 ---
    def optimize_comb(self, initial_mask, target, learning_rate=None,
                      max_iterations=ILT_MAX_ITERATIONS,
                      comb_weight=0.85,  # 默认联合权重
                      log_csv=True,
                      log_dir="logs",
                      experiment_tag="",
                      **optimizer_params):

        mask = initial_mask.copy()

        # 默认学习率
        if learning_rate is None:
            learning_rate = ILT_OPTIMIZER_CONFIGS[self.optimizer_type]['learning_rate']

        default_params = ILT_OPTIMIZER_CONFIGS[self.optimizer_type]['params'].copy()
        default_params.update(optimizer_params)
        optimizer_params = default_params

        # 初始化 CSV 日志
        csv_logger = None
        if log_csv:
            target_shape = f"{target.shape[0]}x{target.shape[1]}"
            if experiment_tag:
                target_shape = f"{target_shape}_{experiment_tag}"

            # 计算初始梯度信息
            init_epe, init_pe, _, _, _ = self._compute_comb_gradient(
                initial_mask, target, comb_weight=comb_weight
            )

            csv_logger = OptimizationLogger(log_dir=log_dir)
            csv_logger.start_logging(
                optimizer_type=self.optimizer_type,
                loss_type=f"Comb(w={comb_weight})",
                learning_rate=learning_rate,
                initial_loss=init_epe,  # 这里记录 EPE 作为主要参考
                target_shape=target_shape,
                config_params=optimizer_params
            )

        self._initialize_optimizer_state(mask.shape)

        history = {
            'pe_loss': [],
            'epe_loss': [],
            'loss': [],
            'grad_norms': [],
            'masks': [],
            'aerial_images': [],
            'printed_images': [],
            'learning_rates': [],
            'csv_log_path': csv_logger.filepath if csv_logger else None
        }

        print(f"Starting Comb Optimization ({max_iterations} iters)...")
        print(f"Config: Comb Weight = {comb_weight} (High=EPE focus, Low=PE focus)")

        best_mask = mask.copy()
        best_epe_loss = float('inf')  # 依然以 EPE 为最终优劣标准

        start_time = time.time()

        for iteration in range(max_iterations):
            # 调用 Comb 梯度计算
            epe_loss, pe_loss, gradient, aerial_image, printed_image = \
                self._compute_comb_gradient(mask, target, comb_weight=comb_weight)

            # 记录历史
            history['epe_loss'].append(epe_loss)
            history['pe_loss'].append(pe_loss)
            history['loss'].append(epe_loss)  # 默认 Loss 曲线画 EPE

            # 记录最佳结果 (通常以 EPE 为准，因为它是精修指标)
            if epe_loss < best_epe_loss:
                best_epe_loss = epe_loss
                best_mask = mask.copy()

            if csv_logger:
                csv_logger.log_iteration(
                    iteration=iteration,
                    loss=epe_loss,
                    gradient=gradient,
                    mask=mask,
                    optimizer_state=self.optimizer_state,
                    time_elapsed=time.time() - start_time
                )

            # 优化器更新
            mask = self._update_with_optimizer(mask, gradient, learning_rate, **optimizer_params)

            history['grad_norms'].append(np.linalg.norm(gradient))
            history['learning_rates'].append(learning_rate)

            if iteration % 20 == 0 or iteration == max_iterations - 1:
                history['masks'].append(mask.copy())
                history['aerial_images'].append(aerial_image)
                history['printed_images'].append(printed_image)

            if iteration % 10 == 0:
                print(f"Iter {iteration}: EPE={epe_loss:.4f}, PE={pe_loss:.4f}, Grad={np.linalg.norm(gradient):.4f}")

        if csv_logger:
            csv_logger.close()

        total_time = time.time() - start_time
        print(f"Comb Optimization completed in {total_time:.2f}s. Best EPE: {best_epe_loss:.4f}")

        return best_mask, history


# --- 对外暴露的 Comb 优化函数 ---
def inverse_lithography_optimization_comb(initial_mask, target_image,
                                          learning_rate=None,
                                          max_iterations=ILT_MAX_ITERATIONS,
                                          optimizer_type=OPTIMIZER_TYPE,
                                          comb_weight=0.85,  # 核心参数
                                          **optimizer_params):
    """
    单一阶段联合优化入口函数
    """
    optimizer = CombInverseLithographyOptimizer(optimizer_type=optimizer_type)

    optimized_mask, history = optimizer.optimize_comb(
        initial_mask=initial_mask,
        target=target_image,
        learning_rate=learning_rate,
        max_iterations=max_iterations,
        comb_weight=comb_weight,
        **optimizer_params
    )

    return optimized_mask, history