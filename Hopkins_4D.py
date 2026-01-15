import time
import numpy as np
import imageio.v2 as iio
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.sparse.linalg import svds
from tqdm import tqdm
from scipy.sparse import lil_matrix, csr_matrix
from matplotlib.colors import LinearSegmentedColormap
from skimage import filters
from skimage.feature import canny
from scipy.ndimage import gaussian_filter


# 光刻仿真参数
LAMBDA = 405  # 波长（单位：纳米）
Z = 803000000  # 距离（单位：纳米）
LX = LY = 1000 # 图像尺寸（单位：像素）
DX = DY = 7560  # 像素尺寸（单位：纳米）
N = 1.5  # 折射率（无量纲）
SIGMA = 0.5  # 部分相干因子（无量纲）
NA = 0.5  # 数值孔径（无量纲）
K_SVD = 20  # 奇异值数目

# DMD调制参数
WX = 7560  # 微镜宽度（单位：纳米）
WY = 7560  # 微镜高度（单位：纳米）
TX = 8560  # 微镜周期（x方向）（单位：纳米）
TY = 8560  # 微镜周期（y方向）（单位：纳米）

# 光刻胶参数
A = 20.0  # sigmoid函数梯度
TR = 0.3  # 阈值参数

# 文件路径
INITIAL_MASK_PATH = "../lithography_simulation_Hopkins/data/input/cell1000_inverse.png"
TARGET_IMAGE_PATH = "../lithography_simulation_Hopkins/data/input/cell1000_inverse.png"
RESULTS_IMAGE_PATH = "../lithography_simulation_Hopkins/data/output/results_comparison_cell1000_inverse.png"
OUTPUT_EDGE_PATH = "../lithography_simulation_Hopkins/data/output/edge_cell1000_inverse.png"
OUTPUT_PEDGE_PATH = "../lithography_simulation_Hopkins/data/output/edgep_cell1000_inverse.png"

# 可视化参数
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为SimHei
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 18
plt.rcParams['font.family'] = 'Times New Roman'


def load_image(path, grayscale=True):
    image = iio.imread(path)
    if grayscale and len(image.shape) > 2:  # 将彩色通道图像转化为灰度图
        image = rgb2gray(image)
    return image


def save_image(image, path):
    plt.imsave(path, image, cmap='gray', vmin=image.min(), vmax=image.max())  # 保存灰度图像


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
    FX, FY = np.meshgrid(fx, fy, indexing='ij')

    J_vals = J(FX, FY)
    P_vals = P(FX, FY)

    # 确保瞳函数包含适当的高频截止
    tcc_kernel = J_vals * P_vals
    Lx, Ly = len(fx), len(fy)

    print("Building improved TCC matrix...")

    # 在邻域搜索范围计算频率相互作用
    TCC_sparse = lil_matrix((Lx * Ly, Lx * Ly), dtype=np.complex128)
    neighborhood_radius = 10

    for i in tqdm(range(Lx), desc="Improved TCC Construction"):
        for j in range(Ly):
            # 计算核函数大于阈值的频率
            if np.abs(tcc_kernel[i, j]) > sparsity_threshold:
                for m in range(max(0, i - neighborhood_radius), min(Lx, i + neighborhood_radius + 1)):
                    for n in range(max(0, j - neighborhood_radius), min(Ly, j + neighborhood_radius + 1)):
                        if np.abs(tcc_kernel[m, n]) > sparsity_threshold:
                            idx1 = i * Ly + j
                            idx2 = m * Ly + n
                            TCC_sparse[idx1, idx2] = tcc_kernel[i, j] * np.conj(tcc_kernel[m, n])

    TCC_csr = csr_matrix(TCC_sparse)

    print(f"Performing SVD with {k} components...")
    U, S, Vh = svds(TCC_csr, k=k)

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
    resist_pattern = 1 / (1 + np.exp(-a * (intensity - Tr)))

    return resist_pattern


def compute_mepe(target, printed_image):

    # 1. 精确边缘检测
    target_edges = filters.roberts(target) > 0.05
    printed_edges  = canny(printed_image, sigma=2.0)

    save_image(target_edges, OUTPUT_EDGE_PATH)
    save_image(printed_edges, OUTPUT_PEDGE_PATH)

    # 2. 获取边缘点坐标
    target_positions = np.argwhere(target_edges)
    printed_positions = np.argwhere(printed_edges)

    # 3. 检查是否有边缘点
    if len(target_positions) == 0:
        print("警告: 未检测到目标图像边缘")
        return 1.0
    if len(printed_positions) == 0:
        print("警告: 未检测到打印图像边缘")
        return 1.0

    # 4. 计算每个目标边缘点到最近打印边缘点的距离
    distances = []
    for target_pos in target_positions:
        # 计算到所有打印边缘点的距离
        dists = np.linalg.norm(printed_positions - target_pos, axis=1)
        min_dist = np.min(dists)
        distances.append(min_dist)

    # 5. 计算平均边缘放置误差
    mepe = np.mean(distances)

    return mepe


def epe_loss(target_image, printed_image, sigma=1.0, epsilon=1e-10, gamma_scale = 10.0):

    # 目标图像平滑（用于鲁棒的梯度计算）
    smoothed_target = gaussian_filter(target_image, sigma=sigma)

    # 计算梯度返回 (dZ_T/dy, dZ_T/dx)
    grad_y, grad_x = np.gradient(smoothed_target)
    weights = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # 权重误差的总和 Sum[ (P - Z_T)^2 * w ] 权重总和 Sum[ w ]
    error_squared = (printed_image - target_image) ** 2
    loss = np.sum(error_squared * weights)

    return loss

def create_black_red_yellow_cmap():
    """创建黑-红-黄的颜色映射"""
    colors = ['black', 'darkred', 'red', 'orange', 'yellow']
    return LinearSegmentedColormap.from_list('black_red_yellow', colors, N=256)


def plot_comparison(target_image, aerial_image_initial, print_image_initial,
                    pe_initial, epe_initial, save_path=None):
    """使用黑红黄梯度表示光强分布的比较图"""

    # 创建自定义颜色映射
    intensity_cmap = create_black_red_yellow_cmap()

    plt.figure(figsize=(18, 12))

    # 目标图像 - 保持灰度
    plt.subplot(231)
    plt.imshow(target_image, cmap='gray')
    plt.title('Target Image')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')

    # 原始掩膜曝光后的图像 - 使用黑红黄梯度
    plt.subplot(232)
    plt.imshow(aerial_image_initial, cmap=intensity_cmap)
    plt.title('Aerial Image (Original)')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')

    # 原始掩膜曝光后的值图像 - 保持灰度
    plt.subplot(233)
    plt.imshow(print_image_initial, cmap='gray')
    plt.title('Printed Image (Original)')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.text(0.75, -0.1, f'PE = {pe_initial:.2f}', transform=plt.gca().transAxes,
             bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    plt.text(0, -0.1, f'EPE = {epe_initial:.2f}', transform=plt.gca().transAxes,
              bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def main():
    # 开始计时
    start_time = time.time()

    # 加载图像
    print("Loading images...")
    initial_mask = load_image(INITIAL_MASK_PATH)
    target_image = load_image(TARGET_IMAGE_PATH)

    # 初始掩膜的光刻仿真
    print("Running initial lithography simulation...")
    aerial_image_initial = hopkins_digital_lithography_simulation(initial_mask)
    print_image_initial = photoresist_model(aerial_image_initial)

    # 计算传统PE
    PE_initial = np.sum((target_image - print_image_initial) ** 2)

    # 计算EPE
    print("Computing MEPE...")
    EPE_initial = epe_loss(target_image , print_image_initial)

    end_time = time.time()
    print(f'Running time: {end_time - start_time:.3f} seconds')
    print(f'Initial PE: {PE_initial}')
    print(f'Initial EPE: {EPE_initial}')

    # 可视化结果
    plot_comparison(target_image, aerial_image_initial, print_image_initial,
                    PE_initial, EPE_initial, RESULTS_IMAGE_PATH)


if __name__ == "__main__":
    main()