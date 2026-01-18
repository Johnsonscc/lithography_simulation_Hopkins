import time
import logging
import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.sparse.linalg import svds
from tqdm import tqdm
from scipy.sparse import lil_matrix, csr_matrix
from config.parameters import *
from utils.optimization_logger import OptimizationLogger

logger = logging.getLogger(__name__)


class InverseLithographyOptimizer:
    """
    逆光刻优化器 (PE Base)
    执行基于像素误差 (Pixel Error) 的梯度下降优化，保存并返回 PE 最优时的掩模。
    """

    def __init__(self, lambda_=LAMBDA, na=NA, sigma=SIGMA, dx=DX, dy=DY,
                 lx=LX, ly=LY, k_svd=ILT_K_SVD, a=A, tr=TR,
                 optimizer_type=OPTIMIZER_TYPE):
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
        self.optimizer_state = {}

        self.singular_values = None
        self.eigen_functions = None
        self._precompute_tcc_svd()

        logger.info(f"InverseLithographyOptimizer initialized with {optimizer_type}")

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

    def _update_with_optimizer(self, mask, gradient, learning_rate, **params):
        if self.optimizer_type == 'sgd':
            new_mask = mask - learning_rate * gradient
        elif self.optimizer_type == 'momentum':
            v = self.optimizer_state['velocity']
            v = params.get('momentum', 0.9) * v - learning_rate * gradient
            self.optimizer_state['velocity'] = v
            new_mask = mask + v
        elif self.optimizer_type == 'adam':
            t = self.optimizer_state['t'] + 1
            m, v = self.optimizer_state['m'], self.optimizer_state['v']
            m = params.get('beta1', 0.9) * m + (1 - 0.9) * gradient
            v = params.get('beta2', 0.999) * v + (1 - 0.999) * (gradient ** 2)
            m_hat = m / (1 - 0.9 ** t)
            v_hat = v / (1 - 0.999 ** t)
            new_mask = mask - learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)
            self.optimizer_state.update({'m': m, 'v': v, 't': t})
        # ... 其他优化器逻辑同理 ...
        return np.clip(new_mask, 0, 1)

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

        if learning_rate is None:
            learning_rate = ILT_OPTIMIZER_CONFIGS[self.optimizer_type]['learning_rate']

        # 日志初始化
        csv_logger = None
        if log_csv:
            csv_logger = OptimizationLogger(log_dir=log_dir)
            init_pe, _, _, _, _ = self._compute_analytical_gradient(mask, target)
            csv_logger.start_logging(self.optimizer_type, "PE", learning_rate, init_pe, f"{target.shape}",
                                     optimizer_params)

        self._initialize_optimizer_state(mask.shape)
        history = {'pe_loss': [], 'epe_loss': [], 'learning_rates': []}
        start_time = time.time()

        print(f"Starting Base PE Optimization ({max_iterations} iters)...")

        for iteration in range(max_iterations):
            pe_loss, epe_loss, gradient, aerial, printed = self._compute_analytical_gradient(mask, target)

            # 保存 PE 最优状态
            if pe_loss < best_pe_loss:
                best_pe_loss = pe_loss
                best_mask = mask.copy()

            history['pe_loss'].append(pe_loss)
            history['epe_loss'].append(epe_loss)

            if csv_logger:
                csv_logger.log_iteration(iteration, pe_loss, gradient, mask, self.optimizer_state,
                                         time.time() - start_time)

            mask = self._update_with_optimizer(mask, gradient, learning_rate, **optimizer_params)

            if iteration % 20 == 0:
                print(f"Iter {iteration}: PE Loss={pe_loss:.4f}, Best PE={best_pe_loss:.4f}")

        if csv_logger: csv_logger.close()
        print(f"Optimization finished. Min PE Loss: {best_pe_loss:.4f}")

        return best_mask, history


def inverse_lithography_optimization_base(initial_mask, target_image,
                                          learning_rate=None,
                                          max_iterations=ILT_MAX_ITERATIONS,
                                          optimizer_type=OPTIMIZER_TYPE,
                                          **optimizer_params):
    optimizer = InverseLithographyOptimizer(optimizer_type=optimizer_type)
    return optimizer.optimize(initial_mask, target_image, learning_rate, max_iterations, **optimizer_params)