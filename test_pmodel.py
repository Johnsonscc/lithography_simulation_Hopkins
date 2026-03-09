import numpy as np
import matplotlib.pyplot as plt
from core.lithography_simulation import photoresist_model
from core.evaluation_function import pe_loss, epe_loss
from utils.image_processing import load_image
from config.parameters import *

def test_photoresist():
    """
    测试用例：掩膜不经过光刻仿真，直接输入光刻胶模型，计算PE和EPE。
    """
    # 1. 加载图像
    print("Loading images...")
    mask = load_image(INITIAL_MASK_PATH)          # 原始掩膜
    target = load_image(TARGET_IMAGE_PATH)        # 目标图形

    # 2. 直接应用光刻胶模型（假设掩膜作为空间像强度）
    print("Applying photoresist model directly to mask...")
    resist = photoresist_model(mask, a=A, Tr=TR)  # 使用配置中的光刻胶参数

    # 3. 计算损失
    pe = pe_loss(target, resist)
    epe = epe_loss(target, resist)

    print("\n=== Results ===")
    print(f"PE loss : {pe:.4f}")
    print(f"EPE loss: {epe:.4f}")

    # 4. 可视化
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(target, cmap='gray')
    plt.title('Target')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap='gray')
    plt.title('Mask (as intensity)')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(resist, cmap='gray')
    plt.title('Resist (direct)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_photoresist()