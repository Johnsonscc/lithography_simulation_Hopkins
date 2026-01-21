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


import numpy as np
import matplotlib.pyplot as plt
import cv2


def plot_edge_constraint_visualization(target_image, initial_mask, final_mask,
                                       update_mask, edge_pixel_range, save_path=None):
    """
    绘制边缘约束优化的可视化图
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. 目标图像
    axes[0, 0].imshow(target_image, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('Target Image')
    axes[0, 0].axis('off')

    # 2. 更新掩膜
    axes[0, 1].imshow(update_mask, cmap='hot', vmin=0, vmax=1)
    axes[0, 1].set_title(f'Update Region Mask\n(Edge Range: {edge_pixel_range}px)')
    axes[0, 1].axis('off')

    # 3. 初始掩膜
    axes[0, 2].imshow(initial_mask, cmap='gray', vmin=0, vmax=1)
    axes[0, 2].set_title('Initial Mask')
    axes[0, 2].axis('off')

    # 4. 最终掩膜
    axes[1, 0].imshow(final_mask, cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title('Optimized Mask')
    axes[1, 0].axis('off')

    # 5. 掩膜差异（只显示更新区域）
    mask_diff = final_mask - initial_mask
    masked_diff = mask_diff * update_mask
    diff_abs = np.abs(masked_diff)

    im = axes[1, 1].imshow(masked_diff, cmap='coolwarm', vmin=-0.5, vmax=0.5)
    axes[1, 1].set_title('Mask Change (Update Region Only)')
    axes[1, 1].axis('off')
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

    # 6. 背景区域变化（应该很小）
    background_mask = (target_image < 0.1).astype(np.float64)
    background_change = np.abs(final_mask - initial_mask) * background_mask

    im2 = axes[1, 2].imshow(background_change, cmap='Reds', vmin=0, vmax=0.1)
    axes[1, 2].set_title('Background Region Changes')
    axes[1, 2].axis('off')
    plt.colorbar(im2, ax=axes[1, 2], fraction=0.046, pad=0.04)

    plt.suptitle(f'Edge-Constrained Inverse Lithography Optimization\n(Edge Range: {edge_pixel_range} pixels)',
                 fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_mask_evolution(mask_history, target_image, save_path=None):
    """
    绘制掩膜演化过程
    """
    n_masks = len(mask_history)
    if n_masks == 0:
        return

    # 选择要显示的关键帧
    if n_masks > 9:
        indices = np.linspace(0, n_masks - 1, 9, dtype=int)
    else:
        indices = range(n_masks)

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        if i >= len(axes):
            break

        mask = mask_history[idx]
        axes[i].imshow(mask, cmap='gray', vmin=0, vmax=1)

        # 叠加目标边缘作为参考
        edges = cv2.Canny((target_image * 255).astype(np.uint8), 50, 150)
        axes[i].contour(edges, colors='red', linewidths=0.5, alpha=0.5)

        axes[i].set_title(f'Iteration {idx}')
        axes[i].axis('off')

    plt.suptitle('Mask Evolution During Optimization', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()