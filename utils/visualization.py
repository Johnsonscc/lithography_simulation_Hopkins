import numpy as np
from config.parameters import *
from matplotlib.colors import LinearSegmentedColormap

def create_black_red_yellow_cmap():
    """创建黑-红-黄的颜色映射"""
    colors = ['black', 'darkred', 'red', 'orange', 'yellow']
    return LinearSegmentedColormap.from_list('black_red_yellow', colors, N=256)


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
    plt.text(0, -0.12, f'EPE = {mepe_initial:.2f}', transform=plt.gca().transAxes,
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
    plt.text(0, -0.12, f'EPE = {mepe_best:.2f}', transform=plt.gca().transAxes,
              bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_dual_axis_loss_history(history, save_path=None):
    """
    自适应双轴损失曲线绘制
    """
    # 提取数据，如果你的 history 里记录了 pe_loss 和 epe_loss
    # 如果只有单一 loss，可以将 history['loss'] 分配给主轴
    pe_data = history.get('pe_loss', history.get('loss', []))
    epe_data = history.get('epe_loss', [])

    iterations = np.arange(len(pe_data))

    # 开启美化风格
    plt.style.use('seaborn-v0_8-muted')  # 或者使用 plt.rcParams 自定义
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=100)

    # 主轴：PE Loss
    color_pe = '#1f77b4'  # 深蓝色
    ax1.set_xlabel('Iterations', fontsize=12, fontweight='bold')
    ax1.set_ylabel('PE Loss (Pixel Error)', color=color_pe, fontsize=12, fontweight='bold')
    line1, = ax1.plot(iterations, pe_data, color=color_pe, linewidth=2, label='PE Loss', alpha=0.8)
    ax1.tick_params(axis='y', labelcolor=color_pe)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # 次轴：EPE Loss (如果有数据的话)
    if len(epe_data) > 0:
        ax2 = ax1.twinx()
        color_epe = '#d62728'  # 红色
        ax2.set_ylabel('EPE Loss (Edge Placement)', color=color_epe, fontsize=12, fontweight='bold')
        line2, = ax2.plot(iterations, epe_data, color=color_epe, linewidth=2, label='EPE Loss', alpha=0.8)
        ax2.tick_params(axis='y', labelcolor=color_epe)

        # 合并图例
        lines = [line1, line2]
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper right', frameon=True, shadow=True)
    else:
        ax1.legend(loc='upper right')

    plt.title('Optimization History (200 Iterations)', fontsize=14, pad=15)

    # 填充曲线下方，增加美感
    ax1.fill_between(iterations, pe_data, color=color_pe, alpha=0.1)

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Loss curve saved to: {save_path}")
    plt.show()