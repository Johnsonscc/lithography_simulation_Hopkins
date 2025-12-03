import imageio.v2 as iio
import numpy as np
from skimage.color import rgb2gray
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

INPUT_MASK_PATH = "../lithography_simulation_Hopkins/data/input/cell1000_inverse.png"
OUTPUT_IMAGE_PATH = "../lithography_simulation_Hopkins/data/output/cell1000_inverse.png"

def load_image(path, grayscale=True):
    image = iio.imread(path)
    if grayscale and len(image.shape) > 2:  # 将彩色通道图像转化为灰度图
        image = rgb2gray(image)
    return image

def save_image(image, path):
    plt.imsave(path, image, cmap='gray', vmin=image.min(), vmax=image.max())  # 保存灰度图像

def visualize_gradient_weights(weights, grad_x, grad_y):
    """可视化梯度权重分布"""

    # 创建图形
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('gradient_anylyes', fontsize=16, fontweight='bold')

    # 1. 权重矩阵热图
    im1 = axes[0, 0].imshow(weights, cmap='hot', aspect='auto')
    axes[0, 0].set_title('image')
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('Y')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)

    # 2. 梯度向量场（采样显示，避免太密集）
    if weights.shape[0] > 50 and weights.shape[1] > 50:
        step_x = weights.shape[1] // 30
        step_y = weights.shape[0] // 30
    else:
        step_x = 1
        step_y = 1

    y_indices, x_indices = np.meshgrid(
        np.arange(0, weights.shape[0], step_y),
        np.arange(0, weights.shape[1], step_x),
        indexing='ij'
    )

    axes[0, 1].quiver(x_indices, y_indices,
                      grad_x[::step_y, ::step_x],
                      grad_y[::step_y, ::step_x],
                      weights[::step_y, ::step_x],
                      cmap='hot', scale=50, width=0.002)
    axes[0, 1].set_title('chang')
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('Y')
    axes[0, 1].invert_yaxis()  # 图像坐标Y轴向下

    # 3. 权重直方图
    axes[0, 2].hist(weights.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 2].set_title('image')
    axes[0, 2].set_xlabel('weight')
    axes[0, 2].set_ylabel('frequency')
    axes[0, 2].grid(True, alpha=0.3)

    # 4. X方向梯度分布
    axes[1, 0].hist(grad_x.flatten(), bins=50, alpha=0.7, color='red', edgecolor='black')
    axes[1, 0].set_title('X')
    axes[1, 0].set_xlabel('gradint')
    axes[1, 0].set_ylabel('frequency')
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Y方向梯度分布
    axes[1, 1].hist(grad_y.flatten(), bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[1, 1].set_title('Y')
    axes[1, 1].set_xlabel('gradint')
    axes[1, 1].set_ylabel('frequency')
    axes[1, 1].grid(True, alpha=0.3)

    # 6. 权重等高线图（如果图像不太大）
    if weights.shape[0] <= 100 and weights.shape[1] <= 100:
        contour = axes[1, 2].contourf(weights, levels=20, cmap='viridis')
        axes[1, 2].set_title('image')
        axes[1, 2].set_xlabel('X')
        axes[1, 2].set_ylabel('Y')
        plt.colorbar(contour, ax=axes[1, 2], fraction=0.046, pad=0.04)
    else:
        # 对于大图像，显示权重分布箱线图
        axes[1, 2].boxplot([weights.flatten(), grad_x.flatten(), grad_y.flatten()],
                           labels=['wright', 'Grad X', 'Grad Y'])
        axes[1, 2].set_title('image')
        axes[1, 2].set_ylabel('vaule')
        axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 打印额外的统计信息
    print(f"\n可视化统计信息:")
    print(f"  权重值 > 0.1 的像素比例: {np.sum(weights > 0.1) / weights.size:.2%}")
    print(f"  权重值 > 0.5 的像素比例: {np.sum(weights > 0.5) / weights.size:.2%}")
    print(f"  权重值 > 1.0 的像素比例: {np.sum(weights > 1.0) / weights.size:.2%}")

    # 打印梯度方向统计
    gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    gradient_angle = np.arctan2(grad_y, grad_x) * 180 / np.pi
    print(f"  平均梯度幅值: {np.mean(gradient_magnitude):.6f}")
    print(f"  梯度方向范围: [{np.min(gradient_angle):.1f}°, {np.max(gradient_angle):.1f}°]")

def main():

    int_image = load_image(INPUT_MASK_PATH)

    # 目标图像平滑（用于鲁棒的梯度计算）
    smoothed_target = gaussian_filter(int_image, sigma=3)

    save_image(smoothed_target, OUTPUT_IMAGE_PATH)

    grad_y, grad_x = np.gradient(smoothed_target)
    weights = np.sqrt(grad_x ** 2 + grad_y ** 2)

    visualize_gradient_weights(weights,grad_x,grad_y)


if __name__ == "__main__":
    main()