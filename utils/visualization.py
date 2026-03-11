import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # Explicit import for 3D plotting
from config.parameters import *
from matplotlib.colors import LinearSegmentedColormap


def create_black_red_yellow_cmap():
    """Create black-red-yellow colormap"""
    colors = ['black', 'darkred', 'red', 'orange', 'yellow']
    return LinearSegmentedColormap.from_list('black_red_yellow', colors, N=256)


def plot_tcc_structure(tcc_matrix, save_path=None):
    """
    Visualizes the TCC Matrix (Transmission Cross Coefficients).
    Includes:
    1. 2D Heatmap
    2. 3D Surface Plot

    Args:
        tcc_matrix: Sparse or dense TCC matrix (Lx*Ly, Lx*Ly).
                   Note: Will be cropped if too large for performance.
        save_path: Path to save the image.
    """
    # 1. Prepare Data
    # Convert to dense if sparse
    try:
        data = tcc_matrix.toarray()
    except:
        data = tcc_matrix

    # If matrix is huge, crop the center for visualization clarity and performance
    # A full TCC matrix for 64x64 mask is 4096*4096, which is too big to plot effectively.
    # We take a representative central block (e.g., 200x200) representing low-frequency interactions.
    max_dim = 200
    if data.shape[0] > max_dim:
        center = data.shape[0] // 2
        start = center - max_dim // 2
        end = start + max_dim
        data_crop = data[start:end, start:end]
        title_suffix = f"(Center {max_dim}x{max_dim} Crop)"
    else:
        data_crop = data
        title_suffix = "(Full Matrix)"

    # We visualize the magnitude of the complex TCC
    Z = np.abs(data_crop)

    # Create coordinate grid
    x = np.arange(Z.shape[1])
    y = np.arange(Z.shape[0])
    X, Y = np.meshgrid(x, y)

    # 2. Setup Plot
    fig = plt.figure(figsize=(16, 7))

    # --- Subplot 1: 2D Heatmap ---
    ax1 = fig.add_subplot(1, 2, 1)
    im = ax1.imshow(Z, cmap='viridis', interpolation='nearest')
    ax1.set_title(f'TCC Matrix Magnitude - Heatmap\n{title_suffix}')
    ax1.set_xlabel('Matrix Column Index (Frequency Mode j)')
    ax1.set_ylabel('Matrix Row Index (Frequency Mode i)')
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    # --- Subplot 2: 3D Surface Plot ---
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    surf = ax2.plot_surface(X, Y, Z, cmap='viridis',
                            linewidth=0, antialiased=False, alpha=0.9)

    ax2.set_title(f'TCC Matrix Magnitude - 3D Surface\n{title_suffix}')
    ax2.set_xlabel('Index j')
    ax2.set_ylabel('Index i')
    ax2.set_zlabel('Magnitude |TCC|')

    # Adjust view angle for better 3D perception
    ax2.view_init(elev=45, azim=-45)

    # Add colorbar for 3D plot
    fig.colorbar(surf, ax=ax2, shrink=0.5, aspect=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"TCC Visualization saved to {save_path}")

    plt.show()


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
    # [Keep original implementation]
    pe_data = history.get('pe_loss', history.get('loss', []))
    epe_data = history.get('epe_loss', [])
    iterations = np.arange(len(pe_data))
    plt.style.use('seaborn-v0_8-muted')
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=100)
    color_pe = '#1f77b4'
    ax1.set_xlabel('Iterations');
    ax1.set_ylabel('PE Loss', color=color_pe)
    ax1.plot(iterations, pe_data, color=color_pe, label='PE Loss')
    if len(epe_data) > 0:
        ax2 = ax1.twinx();
        color_epe = '#d62728'
        ax2.set_ylabel('EPE Loss', color=color_epe)
        ax2.plot(iterations, epe_data, color=color_epe, label='EPE Loss')
    plt.tight_layout()
    if save_path: plt.savefig(save_path)
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


def plot_loss_and_nils(history, save_path=None):
    """
    绘制优化过程中的 PE、EPE 损失和 NILS 曲线
    history : 字典，必须包含 'pe_loss'，可选 'epe_loss' 和 'nils'
    """
    plt.style.use('seaborn-v0_8-muted')
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=100)

    iterations = np.arange(len(history['pe_loss']))

    # 左轴：PE 和 EPE 损失
    color_pe = '#1f77b4'
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss', color='black')
    ax1.plot(iterations, history['pe_loss'], color=color_pe, label='PE Loss')
    if 'epe_loss' in history and len(history['epe_loss']) > 0:
        color_epe = '#d62728'
        ax1.plot(iterations, history['epe_loss'], color=color_epe, label='EPE Loss')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.legend(loc='upper left')

    # 右轴：NILS
    if 'nils' in history and len(history['nils']) > 0:
        ax2 = ax1.twinx()
        color_nils = '#2ca02c'
        ax2.set_ylabel('NILS', color=color_nils)
        ax2.plot(iterations, history['nils'], color=color_nils,
                 linestyle='--', label='NILS')
        ax2.tick_params(axis='y', labelcolor=color_nils)
        ax2.legend(loc='upper right')

    plt.title('Optimization Progress: Loss and NILS')
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()