import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.sparse.linalg import svds
import logging
from tqdm import tqdm
from scipy.sparse import lil_matrix, csr_matrix
from config.parameters import *

logger = logging.getLogger(__name__)


class InverseLithographyOptimizer:
    """
    逆光刻优化器 基于解析梯度计算，支持多种优化器
    """

    def __init__(self, lambda_=LAMBDA, na=NA, sigma=SIGMA, dx=DX, dy=DY,
                 lx=LX, ly=LY, k_svd=ILT_K_SVD, a=A, tr=TR,
                 optimizer_type=ILT_DEFAULT_OPTIMIZER):
        # 光学参数
        self.lambda_ = lambda_
        self.na = na
        self.sigma = sigma
        self.dx = dx
        self.dy = dy
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

        logger.info(f"InverseLithographyOptimizer initialized with {optimizer_type} optimizer")

    def pupil_response_function(self, fx, fy):
        """修正：添加self参数"""
        r = np.sqrt(fx ** 2 + fy ** 2)
        r_max = self.na / self.lambda_
        P = np.where(r < r_max, self.lambda_ ** 2 / (np.pi * (self.na) ** 2), 0)
        return P

    def light_source_function(self, fx, fy):
        """添加光源函数"""
        r = np.sqrt(fx ** 2 + fy ** 2)
        r_max = self.sigma * self.na / self.lambda_
        J = np.where(r <= r_max,
                     self.lambda_ ** 2 / (np.pi * (self.sigma * self.na) ** 2), 0.0)
        return J

    def _compute_full_tcc_matrix(self, fx, fy, sparsity_threshold=0.001):
        """修正：使用成员函数"""
        # 创建频域网格
        Lx, Ly = len(fx), len(fy)
        FX, FY = np.meshgrid(fx, fy, indexing='xy')

        # 使用成员函数计算
        J = self.light_source_function(FX, FY)
        P = self.pupil_response_function(FX, FY)

        tcc_kernel = J * P

        print(f"Building 4D TCC matrix ({Lx}x{Ly}x{Lx}x{Ly})...")

        # 在邻域搜索范围计算频率相互作用
        TCC_sparse = lil_matrix((Lx * Ly, Lx * Ly), dtype=np.complex128)
        neighborhood_radius = 10

        for i in tqdm(range(Lx), desc="Improved TCC Construction"):
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

    def _svd_of_tcc_matrix(self, TCC_csr, k, Lx=LX, Ly=LY):
        print("Performing SVD decomposition...")
        k_actual = min(k, min(TCC_csr.shape) - 1)

        U, S, Vh = svds(TCC_csr, k=k_actual)

        # 过滤掉太小的奇异值
        significant_mask = S > (np.max(S) * 0.01)
        S = S[significant_mask]
        U = U[:, significant_mask]

        idx = np.argsort(S)[::-1]
        S = S[idx]
        U = U[:, idx]

        H_functions = []
        for i in range(len(S)):
            H_i = U[:, i].reshape(Lx, Ly)
            H_functions.append(H_i)

        return S, H_functions

    def _precompute_tcc_svd(self):
        # 频域坐标
        max_freq = self.na / self.lambda_
        freq = 2 * max_freq

        fx = np.linspace(-freq, freq, self.lx)
        fy = np.linspace(-freq, freq, self.ly)

        # 完整计算TCC矩阵
        TCC_4d = self._compute_full_tcc_matrix(fx, fy)

        # 对TCC矩阵进行SVD分解
        self.singular_values, self.eigen_functions = self._svd_of_tcc_matrix(TCC_4d, self.k_svd)

        print(f"TCC SVD precomputation completed with {len(self.singular_values)} singular values")

    def photoresist_model(self, intensity):
        # 光刻胶模型 - sigmoid函数
        return 1 / (1 + np.exp(-self.a * (intensity - self.tr)))

    def _initialize_optimizer_state(self, mask_shape):
        """初始化优化器状态"""
        if self.optimizer_type == 'sgd':
            self.optimizer_state = {}

        elif self.optimizer_type == 'momentum':
            self.optimizer_state = {
                'velocity': np.zeros(mask_shape, dtype=np.float64)
            }

        elif self.optimizer_type == 'rmsprop':
            self.optimizer_state = {
                'square_avg': np.zeros(mask_shape, dtype=np.float64)
            }

        elif self.optimizer_type == 'cg':
            self.optimizer_state = {
                'prev_grad': None,
                'direction': None,
                't': 0
            }

        elif self.optimizer_type == 'adam':
            self.optimizer_state = {
                'm': np.zeros(mask_shape, dtype=np.float64),
                'v': np.zeros(mask_shape, dtype=np.float64),
                't': 0
            }

    def _update_with_optimizer(self, mask, gradient, learning_rate, **optimizer_params):
        """使用选择的优化器更新掩模"""
        # 1. SGD
        if self.optimizer_type == 'sgd':
            new_mask = mask - learning_rate * gradient

        # 2. Momentum
        elif self.optimizer_type == 'momentum':
            momentum = optimizer_params.get('momentum', 0.9)
            velocity = self.optimizer_state['velocity']
            velocity = momentum * velocity - learning_rate * gradient
            self.optimizer_state['velocity'] = velocity
            new_mask = mask + velocity

        # 3. RMSProp
        elif self.optimizer_type == 'rmsprop':
            decay_rate = optimizer_params.get('decay_rate', 0.99)
            epsilon = optimizer_params.get('epsilon', 1e-8)
            square_avg = self.optimizer_state['square_avg']

            square_avg = decay_rate * square_avg + (1 - decay_rate) * (gradient ** 2)
            self.optimizer_state['square_avg'] = square_avg

            new_mask = mask - learning_rate * gradient / (np.sqrt(square_avg) + epsilon)

        # 4. Conjugate Gradient
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

        # 5. Adam
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
            raise ValueError(f"不支持的优化器类型: {self.optimizer_type}")

        # 投影到可行集 [0, 1]
        new_mask = np.clip(new_mask, 0, 1)
        return new_mask

    def _compute_analytical_gradient(self, mask, target):
        # 前向传播计算中间变量
        M_fft = fftshift(fft2(mask))

        # 存储中间结果用于梯度计算
        A_i_list = []  # A_i = h_i ⊗ M
        I_i_list = []  # I_i = |A_i|²

        # 计算光强和各中间项
        intensity = np.zeros((self.lx, self.ly), dtype=np.float64)
        for i, (s_val, H_i) in enumerate(zip(self.singular_values, self.eigen_functions)):
            A_i_fft = M_fft * H_i
            A_i = ifft2(ifftshift(A_i_fft))
            I_i = np.abs(A_i) ** 2

            A_i_list.append(A_i)
            I_i_list.append(I_i)
            intensity += s_val * I_i

        # 归一化光强
        intensity_min = np.min(intensity)
        intensity_max = np.max(intensity)
        if intensity_max - intensity_min > 1e-10:
            intensity_norm = (intensity - intensity_min) / (intensity_max - intensity_min)
        else:
            intensity_norm = intensity / (intensity_max + 1e-10)

        # 光刻胶输出
        P = self.photoresist_model(intensity_norm)

        # 计算损失
        loss = np.sum((target - P) ** 2)

        # 计算梯度 ∂F/∂M
        gradient = np.zeros_like(mask, dtype=np.complex128)

        # 1. ∂F/∂P = -2(z - P)
        dF_dP = -2 * (target - P)

        # 2. ∂P/∂I = a * P * (1 - P) * (∂I_norm/∂I)
        dP_dI_norm = self.a * P * (1 - P)

        # 归一化的导数
        if intensity_max - intensity_min > 1e-10:
            dI_norm_dI = 1.0 / (intensity_max - intensity_min)
        else:
            dI_norm_dI = 1.0 / (intensity_max + 1e-10)

        dP_dI = dP_dI_norm * dI_norm_dI

        # 3. ∂I/∂M 的计算
        for i, (s_val, H_i, A_i, I_i) in enumerate(zip(
                self.singular_values, self.eigen_functions, A_i_list, I_i_list
        )):
            dF_dA_i = dF_dP * dP_dI * 2 * s_val * A_i.conj()
            dF_dA_i_fft = fftshift(fft2(dF_dA_i))
            gradient_contribution = ifft2(ifftshift(dF_dA_i_fft * np.conj(H_i)))
            gradient += gradient_contribution

        # 取实部，因为掩模是实数
        gradient_real = np.real(gradient)
        return loss, gradient_real, intensity_norm, P

    def optimize(self, initial_mask, target, learning_rate=None, max_iterations=ILT_MAX_ITERATIONS, **optimizer_params):
        """
        使用选择的优化器进行优化过程
        """
        mask = initial_mask.copy()

        # 使用配置的默认学习率
        if learning_rate is None:
            learning_rate = ILT_OPTIMIZER_CONFIGS[self.optimizer_type]['learning_rate']

        # 合并默认参数和用户参数
        default_params = ILT_OPTIMIZER_CONFIGS[self.optimizer_type]['params'].copy()
        default_params.update(optimizer_params)
        optimizer_params = default_params

        # 初始化优化器状态
        self._initialize_optimizer_state(mask.shape)

        history = {
            'loss': [],
            'masks': [],
            'aerial_images': [],
            'printed_images': [],
            'learning_rates': []
        }

        print(f"Starting ILT optimization with {max_iterations} iterations...")
        print(f"Using {self.optimizer_type} optimizer (LR: {learning_rate})...")
        if optimizer_params:
            print(f"Optimizer parameters: {optimizer_params}")

        best_mask = mask.copy()
        best_loss = float('inf')

        for iteration in range(max_iterations):
            # 使用解析梯度计算损失和梯度
            loss, gradient, aerial_image, printed_image = self._compute_analytical_gradient(mask, target)

            # 记录最佳掩模
            if loss < best_loss:
                best_loss = loss
                best_mask = mask.copy()

            # 使用选择的优化器更新掩模
            mask = self._update_with_optimizer(mask, gradient, learning_rate, **optimizer_params)

            # 记录历史
            history['loss'].append(loss)
            history['learning_rates'].append(learning_rate)

            if iteration % 20 == 0 or iteration == max_iterations - 1:
                history['masks'].append(mask.copy())
                history['aerial_images'].append(aerial_image)
                history['printed_images'].append(printed_image)

            # 打印进度
            if iteration % 10 == 0:
                grad_norm = np.linalg.norm(gradient)
                if self.optimizer_type == 'momentum':
                    velocity_norm = np.linalg.norm(self.optimizer_state['velocity'])
                    print(
                        f"Iteration {iteration}: Loss = {loss:.6f}, Grad Norm = {grad_norm:.6f}, Velocity Norm = {velocity_norm:.6f}")
                elif self.optimizer_type == 'adam':
                    print(
                        f"Iteration {iteration}: Loss = {loss:.6f}, Grad Norm = {grad_norm:.6f}, t = {self.optimizer_state['t']}")
                else:
                    print(f"Iteration {iteration}: Loss = {loss:.6f}, Grad Norm = {grad_norm:.6f}")

            # 早期停止
            if loss < 1e-6 and iteration > 20:
                print(f"Early stopping at iteration {iteration}")
                break

        print(f"Optimization completed. Best loss: {best_loss:.6f}")
        return best_mask, history


def inverse_lithography_optimization(initial_mask, target_image,
                                     learning_rate=None,
                                     max_iterations=ILT_MAX_ITERATIONS,
                                     optimizer_type='sgd',
                                     **optimizer_params):
    """
    逆光刻优化主函数

    参数:
    - optimizer_type: 优化器类型 ('sgd', 'momentum', 'rmsprop', 'cg', 'adam')
    - learning_rate: 学习率 (None时使用预设值)
    - optimizer_params: 优化器特定参数
    """
    optimizer = InverseLithographyOptimizer(optimizer_type=optimizer_type)

    # 执行优化
    optimized_mask, history = optimizer.optimize(
        initial_mask=initial_mask,
        target=target_image,
        learning_rate=learning_rate,
        max_iterations=max_iterations,
        **optimizer_params
    )

    return optimized_mask, history


