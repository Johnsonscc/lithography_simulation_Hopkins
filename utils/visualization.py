import numpy as np
import matplotlib.pyplot as plt
import cv2
from config.parameters import *
from matplotlib.colors import LinearSegmentedColormap


def create_black_red_yellow_cmap():
    """创建黑-红-黄的颜色映射"""
    colors = ['black', 'darkred', 'red', 'orange', 'yellow']
    return LinearSegmentedColormap.from_list('black_red_yellow', colors, N=256)


def plot_comparison(target_image, aerial_image_initial, print_image_initial,
                    best_mask, best_simulated_image, optimized_binary_simulated_image,
                    pe_initial, pe_best, mepe_initial, mepe_best, save_path=None):
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

    plt.text(0.75, -0.12, f'PE = {pe_best:.2f}', transform=plt.gca().transAxes,
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


def plot_tcc_visualization(tcc_data, save_path=None):
    """
    绘制TCC矩阵的可视化图
    包括奇异值分布、特征函数和TCC矩阵切片
    """
    singular_values = tcc_data['singular_values']
    eigen_functions = tcc_data['eigen_functions']
    tcc_slice = tcc_data.get('tcc_slice', None)

    fig = plt.figure(figsize=(15, 10))

    # 1. 奇异值分布图
    ax1 = plt.subplot(2, 3, 1)
    n_singular = min(len(singular_values), 20)  # 只显示前20个奇异值
    indices = np.arange(1, n_singular + 1)
    ax1.bar(indices, singular_values[:n_singular], color='steelblue', alpha=0.7)
    ax1.set_xlabel('Singular Value Index')
    ax1.set_ylabel('Singular Value Magnitude')
    ax1.set_title('Singular Values Distribution')
    ax1.grid(True, alpha=0.3)

    # 添加奇异值累积能量
    ax1_2 = ax1.twinx()
    cumulative_energy = np.cumsum(singular_values[:n_singular]) / np.sum(singular_values[:n_singular])
    ax1_2.plot(indices, cumulative_energy, 'r-', linewidth=2, label='Cumulative Energy')
    ax1_2.set_ylabel('Cumulative Energy Ratio', color='red')
    ax1_2.tick_params(axis='y', labelcolor='red')
    ax1_2.set_ylim(0, 1.1)

    # 2. 奇异值对数图
    ax2 = plt.subplot(2, 3, 2)
    ax2.semilogy(indices, singular_values[:n_singular], 'bo-', linewidth=2, markersize=4)
    ax2.set_xlabel('Singular Value Index')
    ax2.set_ylabel('Singular Value (Log Scale)')
    ax2.set_title('Singular Values (Log Scale)')
    ax2.grid(True, alpha=0.3)

    # 3. TCC矩阵切片（如果存在）
    if tcc_slice is not None:
        ax3 = plt.subplot(2, 3, 3)
        im = ax3.imshow(np.log10(tcc_slice + 1e-10), cmap='hot',
                        extent=[-1, 1, -1, 1], aspect='auto')
        ax3.set_xlabel('Frequency (fx)')
        ax3.set_ylabel('Frequency (fy)')
        ax3.set_title('TCC Matrix Slice (log scale)')
        plt.colorbar(im, ax=ax3, label='Log10(Magnitude)')

    # 4-6. 前三个特征函数
    n_eigen = min(3, len(eigen_functions))
    for i in range(n_eigen):
        ax = plt.subplot(2, 3, 4 + i)

        # 获取特征函数并归一化显示
        eigen_func = eigen_functions[i]
        eigen_func_norm = np.abs(eigen_func) / np.max(np.abs(eigen_func))

        im = ax.imshow(eigen_func_norm, cmap='viridis',
                       extent=[-1, 1, -1, 1], aspect='auto')
        ax.set_xlabel('Frequency (fx)')
        ax.set_ylabel('Frequency (fy)')
        ax.set_title(f'Eigen Function {i + 1}\nSV={singular_values[i]:.2e}')
        plt.colorbar(im, ax=ax, label='Normalized Magnitude')

    plt.suptitle('TCC Matrix Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为标题留出空间

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"TCC visualization saved to: {save_path}")
        plt.close()
    else:
        plt.show()


def plot_tcc_eigenfunctions(eigen_functions, singular_values, save_path=None):
    """
    专门绘制TCC特征函数的可视化图
    """
    n_eigen = min(9, len(eigen_functions))

    # 计算布局
    n_cols = 3
    n_rows = (n_eigen + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))

    # 如果只有一行，确保axes是二维数组
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_eigen):
        row = i // n_cols
        col = i % n_cols

        ax = axes[row, col]

        # 获取特征函数并计算幅度
        eigen_func = eigen_functions[i]
        eigen_func_mag = np.abs(eigen_func)

        # 归一化到[0, 1]范围
        if np.max(eigen_func_mag) > 0:
            eigen_func_mag = eigen_func_mag / np.max(eigen_func_mag)

        im = ax.imshow(eigen_func_mag, cmap='plasma',
                       extent=[-1, 1, -1, 1], aspect='auto')
        ax.set_title(f'Eigen Function {i + 1}\nSV={singular_values[i]:.2e}')
        ax.set_xlabel('Frequency (fx)')
        ax.set_ylabel('Frequency (fy)')

        plt.colorbar(im, ax=ax, label='Normalized Magnitude')

    # 隐藏多余的子图
    for i in range(n_eigen, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')

    plt.suptitle('TCC Eigen Functions (Coherent Modes)', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"TCC eigenfunctions saved to: {save_path}")
        plt.close()
    else:
        plt.show()


def plot_tcc_analysis_comprehensive(tcc_data, save_path_prefix=None):
    """
    绘制全面的TCC分析图，包含多个子图
    """
    singular_values = tcc_data['singular_values']
    eigen_functions = tcc_data['eigen_functions']

    # 创建一个大图
    fig = plt.figure(figsize=(18, 12))

    # 1. 奇异值分布（线性和对数）
    ax1 = plt.subplot(2, 3, 1)
    n_singular = min(len(singular_values), 30)
    indices = np.arange(1, n_singular + 1)

    # 线性尺度
    bars = ax1.bar(indices, singular_values[:n_singular], color='steelblue', alpha=0.7)
    ax1.set_xlabel('Singular Value Index')
    ax1.set_ylabel('Singular Value', color='steelblue')
    ax1.set_title('Singular Values Distribution')
    ax1.grid(True, alpha=0.3)

    # 对数尺度（右侧Y轴）
    ax1_log = ax1.twinx()
    ax1_log.semilogy(indices, singular_values[:n_singular], 'ro-', linewidth=2, markersize=4)
    ax1_log.set_ylabel('Singular Value (Log Scale)', color='red')
    ax1_log.tick_params(axis='y', labelcolor='red')

    # 2. 累积能量
    ax2 = plt.subplot(2, 3, 2)
    cumulative = np.cumsum(singular_values)
    cumulative_normalized = cumulative / np.sum(singular_values)

    ax2.plot(np.arange(1, len(singular_values) + 1), cumulative_normalized,
             'g-', linewidth=3, label='Cumulative Energy')
    ax2.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='90% Threshold')
    ax2.axhline(y=0.95, color='orange', linestyle='--', alpha=0.5, label='95% Threshold')

    # 找到达到90%和95%所需的最小奇异值数量
    idx_90 = np.argmax(cumulative_normalized >= 0.9) + 1
    idx_95 = np.argmax(cumulative_normalized >= 0.95) + 1

    ax2.axvline(x=idx_90, color='r', linestyle=':', alpha=0.7)
    ax2.axvline(x=idx_95, color='orange', linestyle=':', alpha=0.7)

    ax2.text(idx_90, 0.5, f'{idx_90} modes\nfor 90%',
             fontsize=10, ha='center', va='center',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax2.text(idx_95, 0.7, f'{idx_95} modes\nfor 95%',
             fontsize=10, ha='center', va='center',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax2.set_xlabel('Number of Modes')
    ax2.set_ylabel('Cumulative Energy Ratio')
    ax2.set_title('Cumulative Energy of Singular Values')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)

    # 3. 特征值谱
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(indices, singular_values[:n_singular], 'b-', linewidth=2)
    ax3.fill_between(indices, 0, singular_values[:n_singular], alpha=0.3)
    ax3.set_xlabel('Index')
    ax3.set_ylabel('Singular Value')
    ax3.set_title('Singular Value Spectrum')
    ax3.grid(True, alpha=0.3)

    # 4-6. 前三个最重要的特征函数
    n_show = min(3, len(eigen_functions))

    for i in range(n_show):
        ax = plt.subplot(2, 3, 4 + i)
        eigen_func = eigen_functions[i]

        # 计算幅度和相位
        magnitude = np.abs(eigen_func)
        phase = np.angle(eigen_func)

        # 创建RGB图像：R=幅度，G=正相位部分，B=负相位部分
        magnitude_norm = magnitude / np.max(magnitude)
        phase_norm = (phase + np.pi) / (2 * np.pi)  # 归一化到[0, 1]

        # 使用HSV色彩空间：色相表示相位，饱和度表示幅度
        hsv_image = np.zeros((eigen_func.shape[0], eigen_func.shape[1], 3))
        hsv_image[..., 0] = phase_norm  # 色相
        hsv_image[..., 1] = magnitude_norm  # 饱和度
        hsv_image[..., 2] = magnitude_norm  # 明度

        # 转换为RGB
        from matplotlib.colors import hsv_to_rgb
        rgb_image = hsv_to_rgb(hsv_image)

        im = ax.imshow(rgb_image, extent=[-1, 1, -1, 1], aspect='auto')
        ax.set_title(f'Eigen Function {i + 1}\nSV={singular_values[i]:.2e}')
        ax.set_xlabel('Frequency (fx)')
        ax.set_ylabel('Frequency (fy)')

        # 添加颜色条说明
        from matplotlib.patches import Rectangle
        from matplotlib.colors import Normalize

        # 创建自定义颜色条
        sm = plt.cm.ScalarMappable(cmap='hsv', norm=Normalize(0, 2 * np.pi))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical')
        cbar.set_label('Phase (radians)')

        # 添加幅度说明
        ax.text(0.02, 0.98, f'Max Mag: {np.max(magnitude):.2e}',
                transform=ax.transAxes, fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.suptitle('Comprehensive TCC Matrix Analysis', fontsize=18, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path_prefix:
        save_path = f"{save_path_prefix}_tcc_comprehensive.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comprehensive TCC analysis saved to: {save_path}")
        plt.close()
    else:
        plt.show()


import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec


def plot_soco_visualization(mask, eigen_functions, singular_values, save_path=None):
    """
    绘制SOCO(Sum of Coherent Systems)可视化图
    左侧：掩模图形 M(x, y)
    中间：3-4个并行成像通道，每个通道显示卷积运算
    右侧：合成最终光强图像
    """

    # 选择前3-4个特征通道进行可视化
    num_channels = min(4, len(eigen_functions))
    selected_indices = list(range(num_channels))

    # 计算每个通道的卷积核（空域核）
    spatial_kernels = []
    for i in selected_indices:
        H_i = eigen_functions[i]  # 频域核
        # 逆傅里叶变换到空域
        h_i = np.fft.ifft2(np.fft.ifftshift(H_i))
        # 取实部，并归一化以便可视化
        h_i_real = np.real(h_i)
        spatial_kernels.append(h_i_real)

    # 计算每个通道的卷积结果
    mask_fft = np.fft.fft2(mask)
    mask_fft_shifted = np.fft.fftshift(mask_fft)

    channel_results = []
    for i in selected_indices:
        H_i = eigen_functions[i]
        # 频域相乘 = 空域卷积
        A_i = np.fft.ifft2(np.fft.ifftshift(mask_fft_shifted * H_i))
        # 取模的平方得到部分光强
        I_i = np.abs(A_i) ** 2
        channel_results.append(I_i)

    # 计算总光强（加权和）
    total_intensity = np.zeros_like(channel_results[0])
    for i, idx in enumerate(selected_indices):
        total_intensity += singular_values[idx] * channel_results[i]

    # 创建图形
    fig = plt.figure(figsize=(18, 10))

    # 使用GridSpec创建复杂的布局
    gs = gridspec.GridSpec(3, num_channels + 2, width_ratios=[1] + [1] * num_channels + [1.2],
                           height_ratios=[1, 0.2, 1])

    # 1. 左侧：掩模图形
    ax_mask = plt.subplot(gs[0, 0])
    ax_mask.imshow(mask, cmap='gray', vmin=0, vmax=1)
    ax_mask.set_title('Mask Pattern\n$M(x, y)$', fontsize=14, fontweight='bold')
    ax_mask.set_xlabel('x', fontsize=12)
    ax_mask.set_ylabel('y', fontsize=12)
    ax_mask.text(0.5, -0.15, 'Input', transform=ax_mask.transAxes,
                 ha='center', fontsize=12, fontweight='bold')

    # 添加箭头指向第一个通道
    arrow_x = 1.1
    arrow_y = 0.5
    arrow = FancyArrowPatch(
        (arrow_x, arrow_y), (1.2, arrow_y),
        arrowstyle='->', mutation_scale=20,
        linewidth=2, color='red'
    )
    ax_mask.add_patch(arrow)
    ax_mask.text(arrow_x + 0.05, arrow_y + 0.1, 'Convolution',
                 transform=ax_mask.transAxes, fontsize=10, color='red')

    # 2. 中间：并行通道
    channel_axes = []
    for i, idx in enumerate(selected_indices):
        ax_channel = plt.subplot(gs[0, i + 1])

        # 显示卷积核
        kernel_display = spatial_kernels[i]
        # 归一化到0-1以便显示
        kernel_norm = (kernel_display - np.min(kernel_display)) / (
                    np.max(kernel_display) - np.min(kernel_display) + 1e-10)

        im_kernel = ax_channel.imshow(kernel_norm, cmap='hot')
        ax_channel.set_title(f'Channel {i + 1}\n$h_{{{i + 1}}}(x, y)$', fontsize=12)
        ax_channel.set_xlabel('x', fontsize=10)
        if i == 0:
            ax_channel.set_ylabel('y', fontsize=10)

        # 添加权重标注
        sigma_i = singular_values[idx]
        ax_channel.text(0.5, -0.25, f'$\\times\\, \\sigma_{{{i + 1}}} = {sigma_i:.3f}$',
                        transform=ax_channel.transAxes, ha='center', fontsize=11,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

        channel_axes.append(ax_channel)

        # 添加向下箭头到卷积结果
        ax_arrow = plt.subplot(gs[1, i + 1])
        ax_arrow.axis('off')
        arrow_down = FancyArrowPatch(
            (0.5, 0.8), (0.5, 0.2),
            arrowstyle='->', mutation_scale=20,
            linewidth=2, color='blue',
            transform=ax_arrow.transAxes
        )
        ax_arrow.add_patch(arrow_down)
        ax_arrow.text(0.5, 0.5, '$|\\cdot|^2$', transform=ax_arrow.transAxes,
                      ha='center', va='center', fontsize=12, fontweight='bold',
                      bbox=dict(boxstyle="round,pad=0.2", facecolor="lightblue", alpha=0.8))

    # 3. 卷积结果显示
    result_axes = []
    for i, idx in enumerate(selected_indices):
        ax_result = plt.subplot(gs[2, i + 1])

        # 显示部分光强
        I_i = channel_results[i]
        im_result = ax_result.imshow(I_i, cmap='viridis')
        ax_result.set_title(f'$I_{{{i + 1}}}(x, y)$', fontsize=12)
        ax_result.set_xlabel('x', fontsize=10)
        if i == 0:
            ax_result.set_ylabel('y', fontsize=10)

        # 添加权重标注
        sigma_i = singular_values[idx]
        intensity_max = np.max(I_i)
        ax_result.text(0.02, 0.98, f'$\\sigma_{{{i + 1}}}={sigma_i:.3f}$',
                       transform=ax_result.transAxes, fontsize=10, color='white',
                       bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7),
                       verticalalignment='top')

        result_axes.append(ax_result)

        # 添加向右的箭头（最后一个通道除外）
        if i < num_channels - 1:
            ax_arrow_right = plt.subplot(gs[2, i + 1])
            ax_arrow_right_right = plt.subplot(gs[2, i + 2])

            # 在坐标轴之间添加箭头
            fig.patches.extend([
                FancyArrowPatch(
                    (ax_result.get_position().x1, ax_result.get_position().y0 + 0.5 * ax_result.get_position().height),
                    (ax_arrow_right_right.get_position().x0,
                     ax_arrow_right_right.get_position().y0 + 0.5 * ax_arrow_right_right.get_position().height),
                    arrowstyle='->', mutation_scale=20,
                    linewidth=2, color='green',
                    transform=fig.transFigure
                )
            ])

    # 4. 右侧：总光强图像
    ax_total = plt.subplot(gs[0:3, -1])

    # 显示总光强
    im_total = ax_total.imshow(total_intensity, cmap='plasma')
    ax_total.set_title('Total Aerial Image\n$I(x, y)$', fontsize=14, fontweight='bold')
    ax_total.set_xlabel('x', fontsize=12)
    ax_total.set_ylabel('y', fontsize=12)

    # 添加求和公式
    sum_text = '$I(x, y) = \\sum_{i=1}^{' + str(num_channels) + '} \\sigma_i \\, |h_i(x, y) \\otimes M(x, y)|^2$'
    ax_total.text(0.5, -0.1, sum_text, transform=ax_total.transAxes,
                  ha='center', fontsize=13, fontweight='bold',
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))

    # 添加从最后一个通道到总光强的箭头
    last_ax = result_axes[-1]
    fig.patches.extend([
        FancyArrowPatch(
            (last_ax.get_position().x1, last_ax.get_position().y0 + 0.5 * last_ax.get_position().height),
            (ax_total.get_position().x0, ax_total.get_position().y0 + 0.5 * ax_total.get_position().height),
            arrowstyle='->', mutation_scale=25,
            linewidth=3, color='purple',
            transform=fig.transFigure
        )
    ])

    # 在所有通道上方添加求和符号
    ax_sum = plt.subplot(gs[1, num_channels])
    ax_sum.axis('off')
    ax_sum.text(0.5, 0.5, '$\\sum$', fontsize=30, ha='center', va='center',
                color='darkgreen', fontweight='bold')

    plt.suptitle('SOCO (Sum of Coherent Systems) Imaging Model',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout()

    # 调整整体布局
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.05, right=0.95,
                        hspace=0.3, wspace=0.4)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"SOCO visualization saved to: {save_path}")
        plt.close()
    else:
        plt.show()


def plot_soco_simplified(mask, eigen_functions, singular_values, save_path=None):
    """
    简化版SOCO可视化，适合快速查看
    """
    num_channels = min(4, len(eigen_functions))

    fig, axes = plt.subplots(2, num_channels + 2, figsize=(16, 8))

    # 左侧：掩模
    axes[0, 0].imshow(mask, cmap='gray')
    axes[0, 0].set_title('Mask $M(x,y)$')
    axes[0, 0].axis('off')

    axes[1, 0].axis('off')
    axes[1, 0].text(0.5, 0.5, 'Input', ha='center', va='center', fontsize=12)

    # 中间：通道
    for i in range(num_channels):
        # 卷积核
        H_i = eigen_functions[i]
        h_i = np.real(np.fft.ifft2(np.fft.ifftshift(H_i)))
        axes[0, i + 1].imshow(h_i, cmap='hot')
        axes[0, i + 1].set_title(f'Kernel $h_{{{i + 1}}}$')
        axes[0, i + 1].axis('off')

        # 权重
        sigma_i = singular_values[i]
        axes[0, i + 1].text(0.5, -0.15, f'$\\times\\, \\sigma_{{{i + 1}}}={sigma_i:.3f}$',
                            transform=axes[0, i + 1].transAxes, ha='center', fontsize=10)

        # 卷积结果
        mask_fft = np.fft.fft2(mask)
        mask_fft_shifted = np.fft.fftshift(mask_fft)
        A_i = np.fft.ifft2(np.fft.ifftshift(mask_fft_shifted * H_i))
        I_i = np.abs(A_i) ** 2
        axes[1, i + 1].imshow(I_i, cmap='viridis')
        axes[1, i + 1].set_title(f'$I_{{{i + 1}}}$')
        axes[1, i + 1].axis('off')

    # 右侧：总光强
    total_intensity = np.zeros_like(I_i)
    for i in range(num_channels):
        H_i = eigen_functions[i]
        mask_fft = np.fft.fft2(mask)
        mask_fft_shifted = np.fft.fftshift(mask_fft)
        A_i = np.fft.ifft2(np.fft.ifftshift(mask_fft_shifted * H_i))
        I_i = np.abs(A_i) ** 2
        total_intensity += singular_values[i] * I_i

    axes[0, -1].imshow(total_intensity, cmap='plasma')
    axes[0, -1].set_title('Total Intensity $I(x,y)$')
    axes[0, -1].axis('off')

    axes[1, -1].axis('off')
    axes[1, -1].text(0.5, 0.5, '$I = \\sum \\sigma_i |h_i \\otimes M|^2$',
                     ha='center', va='center', fontsize=12,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

    plt.suptitle('SOCO Imaging Model Visualization', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

