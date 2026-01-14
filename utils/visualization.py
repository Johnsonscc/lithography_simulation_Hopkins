import matplotlib.pyplot as plt
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



def plot_dual_axis_loss_history(pe_history, epe_history, save_path=None):
    """
    在同一个折线图中用两个Y轴同时显示PE-loss和EPE-loss

    参数:
        pe_history: PE优化阶段的历史记录，包含'pe_loss'和'epe_loss'
        epe_history: EPE优化阶段的历史记录，包含'pe_loss'和'epe_loss'
        save_path: 保存路径
    """
    # 提取两个阶段的损失数据
    pe_loss_stage1 = pe_history.get('pe_loss', [])
    epe_loss_stage1 = pe_history.get('epe_loss', [])

    pe_loss_stage2 = epe_history.get('pe_loss', [])
    epe_loss_stage2 = epe_history.get('epe_loss', [])

    # 组合全阶段数据
    total_pe_loss = pe_loss_stage1 + pe_loss_stage2
    total_epe_loss = epe_loss_stage1 + epe_loss_stage2

    # 创建迭代次数索引
    iterations = np.arange(1, len(total_pe_loss) + 1)

    # 创建图形和第一个Y轴
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # 绘制PE损失曲线（左轴）
    color1 = 'blue'
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('PE Loss', color=color1, fontsize=12)
    ax1.plot(iterations, total_pe_loss, color=color1, linewidth=2.5, label='PE Loss')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3, linestyle='--')

    # 创建第二个Y轴
    ax2 = ax1.twinx()

    # 绘制EPE损失曲线（右轴）
    color2 = 'red'
    ax2.set_ylabel('EPE Loss', color=color2, fontsize=12)
    ax2.plot(iterations, total_epe_loss, color=color2, linewidth=2.5, label='EPE Loss')
    ax2.tick_params(axis='y', labelcolor=color2)

    # 标记阶段转换位置
    if len(pe_loss_stage1) > 0:
        # 添加垂直线标记阶段转换
        switch_point = len(pe_loss_stage1)
        ax1.axvline(x=switch_point, color='gray', linestyle='--', alpha=0.7, linewidth=1)

        # 添加文本标注
        y_pos = ax1.get_ylim()[1] * 0.95  # 在PE轴顶部附近
        ax1.text(switch_point, y_pos, '  Stage Switch (PE→EPE)',
                 color='gray', alpha=0.8, fontsize=10, verticalalignment='top')

    # 添加图例（合并两个轴的图例）
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    # 设置标题
    plt.title('Two-Stage ILT Optimization: PE and EPE Loss History',
              fontsize=14, fontweight='bold', pad=20)

    # 调整布局
    plt.tight_layout()

    # 保存图像
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
    plt.close()


def plot_two_stage_history(pe_history, epe_history, save_path=None):
    """双Y轴绘制优化历史"""
    # 这里保持原有功能，调用新的双轴函数
    plot_dual_axis_loss_history(pe_history, epe_history, save_path)