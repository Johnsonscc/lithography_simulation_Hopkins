import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from config.parameters import *
from scipy.sparse.linalg import svds
from tqdm import tqdm
from scipy.sparse import lil_matrix, csr_matrix
import logging


# 设置日志
logger = logging.getLogger(__name__)


def light_source_function(fx, fy, sigma=SIGMA, na=NA, lambda_=LAMBDA):
    r = np.sqrt(fx ** 2 + fy ** 2)
    r_max = sigma * na / lambda_
    J = np.where(r <= r_max, lambda_ ** 2 / (np.pi * (sigma * na) ** 2), 0)
    return J


def pupil_response_function(fx, fy, na=NA, lambda_=LAMBDA):
    r = np.sqrt(fx ** 2 + fy ** 2)
    r_max = na / lambda_
    P = np.where(r < r_max, lambda_ ** 2 / (np.pi * (na) ** 2), 0)
    return P


def compute_tcc_svd(J, P, fx, fy, k, sparsity_threshold=0.001):

    FX, FY = np.meshgrid(fx, fy, indexing='xy')

    J_vals = J(FX, FY)
    P_vals = P(FX, FY)

    # 确保瞳函数包含适当的高频截止
    tcc_kernel = J_vals * P_vals
    Lx, Ly = len(fx), len(fy)

    print("Building TCC matrix...")

    # 在邻域搜索范围计算频率相互作用
    TCC_sparse = lil_matrix((Lx * Ly, Lx * Ly), dtype=np.complex128)
    neighborhood_radius = 10

    for i in tqdm(range(Lx), desc="TCC Construction"):
        for j in range(Ly):
            #计算核函数大于阈值的频率
            if np.abs(tcc_kernel[i, j]) > sparsity_threshold:
                for m in range(max(0, i - neighborhood_radius),min(Lx, i + neighborhood_radius + 1)):
                    for n in range(max(0, j - neighborhood_radius),min(Ly, j + neighborhood_radius + 1)):
                        if np.abs(tcc_kernel[m, n]) > sparsity_threshold:
                            idx1 = i * Ly + j
                            idx2 = m * Ly + n
                            TCC_sparse[idx1, idx2] = tcc_kernel[i, j] * np.conj(tcc_kernel[m, n])

    TCC_csr = csr_matrix(TCC_sparse)


    print(f"Performing SVD with {k} components...")
    U, S, Vh = svds(TCC_csr, k=k,random_state=42)

    # 过滤掉太小的奇异值
    significant_mask = S > (np.max(S) * 0.01)  # 只保留大于1%最大值的奇异值
    S = S[significant_mask]
    U = U[:, significant_mask]

    idx = np.argsort(S)[::-1]
    S = S[idx]
    U = U[:, idx]

    H_functions = []
    for i in tqdm(range(len(S)), desc="Extracting eigenfunctions"):
        H_i = U[:, i].reshape(Lx, Ly)
        H_functions.append(H_i)

    return S, H_functions


def hopkins_digital_lithography_simulation(mask, lambda_=LAMBDA, lx=LX, ly=LY,
                                            sigma=SIGMA, na=NA, k_svd=K_SVD):
    # 频域坐标
    max_freq = na / lambda_
    freq = 2 * max_freq

    fx = np.linspace(-freq, freq, lx)
    fy = np.linspace(-freq, freq, ly)

    # 定义光源和瞳函数
    J = lambda fx, fy: light_source_function(fx, fy, sigma, na, lambda_)
    P = lambda fx, fy: pupil_response_function(fx, fy, na, lambda_)

    # 计算TCC并进行SVD分解
    singular, H_functions = compute_tcc_svd(J, P, fx, fy, k_svd)

    # 掩模的傅里叶变换
    M_fft = fftshift(fft2(mask))

    # 初始化光强
    intensity = np.zeros((lx, ly), dtype=np.float64)

    # 根据SVD分解计算光强
    print(f"Computing intensity using {len(singular)} singular values...")
    for i, (s_val, H_i) in enumerate(zip(singular, H_functions)):
        # 滤波后的频谱
        filtered_fft = M_fft * H_i

        # 逆傅里叶变换
        filtered_space = ifft2(ifftshift(filtered_fft))

        # 累加光强贡献
        intensity += s_val * np.abs(filtered_space) ** 2

    # 最终结果
    result = intensity

    # 修复归一化：避免除0
    intensity_min = np.min(intensity)
    intensity_max = np.max(intensity)

    if intensity_max - intensity_min > 1e-10:
        result = (intensity - intensity_min) / (intensity_max - intensity_min)
    else:
        print("警告: 光强分布范围过小，使用备选归一化")
        result = intensity / (intensity_max + 1e-10)  # 避免除0

    return result


def photoresist_model(intensity, a=A, Tr=TR):
    # 应用sigmoid函数
    resist_pattern = 1 / (1 + np.exp(-a * (intensity - Tr)))
    return resist_pattern

