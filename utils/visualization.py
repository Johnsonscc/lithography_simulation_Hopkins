import matplotlib.pyplot as plt
import numpy as np
from config.parameters import *
from matplotlib.colors import LinearSegmentedColormap


def create_black_red_yellow_cmap():
    """创建黑-红-黄的颜色映射"""
    colors = ['black', 'darkred', 'red', 'orange', 'yellow']
    return LinearSegmentedColormap.from_list('black_red_yellow', colors, N=256)


def plot_comparison(target_image, aerial_image_initial, print_image_initial,
                    best_mask, best_simulated_image, optimized_binary_simulated_image,
                    pe_initial, pe_best, mepe_initial, mepe_best,save_path=None):
     # 创建自定义颜色映射
    intensity_cmap = create_black_red_yellow_cmap()

    plt.figure(figsize=(18, 12))

    # 目标图像
    plt.subplot(231)
    plt.imshow(target_image, cmap='gray')
    plt.title('Target Image')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')

    # 原始掩膜曝光后的图像
    plt.subplot(232)
    plt.imshow(aerial_image_initial, cmap=intensity_cmap)
    plt.title('Aerial Image (Original)')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')

    # 原始掩膜曝光后的二值图像
    plt.subplot(233)
    plt.imshow(print_image_initial, cmap='gray')
    plt.title('Printed Image (Original)')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.text(0.75, -0.12, f'PE = {pe_initial:.2f}', transform=plt.gca().transAxes,
             bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    plt.text(0, -0.12, f'MEPE = {mepe_initial:.2f}', transform=plt.gca().transAxes,
              bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))


# 优化后的掩膜
    plt.subplot(234)
    plt.imshow(best_mask, cmap='gray')
    plt.title('Optimized Mask')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')

    # 优化掩膜曝光后的图像
    plt.subplot(235)
    plt.imshow(best_simulated_image, cmap=intensity_cmap)
    plt.title('Aerial Image (Optimized)')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')

    # 优化掩膜曝光后的二值图像
    plt.subplot(236)
    plt.imshow(optimized_binary_simulated_image, cmap='gray')
    plt.title('Printed Image (Optimized)')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')

    plt.text(0.75, -0.12, f'PE = {pe_best:.2f}',transform=plt.gca().transAxes,
             bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    plt.text(0, -0.12, f'MEPE = {mepe_best:.2f}', transform=plt.gca().transAxes,
              bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_optimization_history(history, save_path=None):
    """简化优化历史 - 只显示损失和梯度"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 确保有损失数据
    if not history or 'loss' not in history or len(history['loss']) == 0:
        print("No optimization history data available")
        return

    # 1. 损失曲线
    iterations = range(len(history['loss']))
    ax1.plot(iterations, history['loss'], 'b-', linewidth=2, label='Loss')
    ax1.set_title('Loss Evolution')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)

    # 2. 梯度信息
    if 'gradient_norms' in history and len(history['gradient_norms']) > 0:
        # 绘制梯度范数
        ax2.plot(iterations[:len(history['gradient_norms'])],
                 history['gradient_norms'], 'g-', linewidth=2, label='Gradient Norm')
        ax2.set_title('Gradient Norm Evolution')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Gradient Norm')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)

    else:
        # 如果没有梯度范数数据，显示收敛分析
        loss_diff = np.diff(history['loss'])
        ax2.plot(iterations[1:], loss_diff, 'r-', linewidth=2, label='Loss Change')
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax2.set_title('Loss Change per Iteration')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Δ Loss')
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
