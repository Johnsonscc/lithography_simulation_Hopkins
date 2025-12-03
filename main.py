import time
import numpy as np
from config.parameters import *
from utils.image_processing import load_image, save_image
from core.lithography_simulation import hopkins_digital_lithography_simulation, photoresist_model
from core.inverse_lithography import inverse_lithography_optimization
from utils.visualization import plot_comparison,plot_optimization_history
from scipy.ndimage import gaussian_filter
from skimage import filters
from skimage.feature import canny

def compute_mepe(target, printed_image):

    # 1. 精确边缘检测
    target_edges = filters.roberts(target) > 0.05
    printed_edges  = canny(printed_image, sigma=2.0)

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


def mepe_loss(target_image, printed_image, sigma=3.0, epsilon=1e-10, gamma_scale = 10.0):

    # 目标图像平滑（用于鲁棒的梯度计算）
    smoothed_target = gaussian_filter(target_image, sigma=sigma)

    # 计算梯度返回 (dZ_T/dy, dZ_T/dx)
    grad_y, grad_x = np.gradient(smoothed_target)
    weights = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # 权重误差的总和 Sum[ (P - Z_T)^2 * w ] 权重总和 Sum[ w ]
    error_squared = (printed_image - target_image) ** 2
    numerator = np.sum(error_squared * weights)
    denominator = np.sum(weights)

    if denominator < epsilon:
        # 如果没有边缘，损失为 0
        return 0.0

    #计算平均边缘放置误差
    loss = numerator / (denominator + epsilon)
    # 引入缩放因子以匹配物理量纲
    loss_scaled = loss * (gamma_scale)  # 乘以 10

    return loss_scaled


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
    PE_initial = np.sum((target_image - print_image_initial) ** 2)
    MEPE_initial = mepe_loss(target_image, print_image_initial)

    # 使用逆光刻优化
    print("Starting inverse lithography optimization...")

    best_mask, history = inverse_lithography_optimization(
        initial_mask=initial_mask,
        target_image=target_image
    )

    # 最佳掩膜的光刻仿真
    print("Running lithography simulation for optimized mask...")
    best_aerial_image = hopkins_digital_lithography_simulation(best_mask)
    best_print_image = photoresist_model(best_aerial_image)
    PE_best = np.sum((target_image - best_print_image) ** 2)
    MEPE_best = mepe_loss(target_image, best_print_image)

    # 结束计时
    end_time = time.time()
    print(f'Running time: {end_time - start_time:.3f} seconds')
    print(f' Initial PE: {PE_initial}, Best PE: {PE_best}')
    print(f' Initial MEPE:{MEPE_initial}, Best MEPE: {MEPE_best}')

    # 保存优化后的掩膜
    save_image(best_mask, OUTPUT_MASK_PATH)

    # 可视化结果
    plot_comparison(target_image, aerial_image_initial, print_image_initial,
                    best_mask, best_aerial_image, best_print_image,
                    PE_initial, PE_best, MEPE_initial, MEPE_best, RESULTS_IMAGE_PATH)

    plot_optimization_history(history,FITNESS_PLOT_PATH)


if __name__ == "__main__":
    main()