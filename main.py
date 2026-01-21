import time
import os
import numpy as np
from config.parameters import *
from utils.image_processing import load_image, save_image
from core.lithography_simulation import hopkins_digital_lithography_simulation, photoresist_model
from core.evaluation_function import pe_loss, epe_loss
from utils.visualization import plot_comparison, plot_dual_axis_loss_history, plot_edge_constraint_visualization

# 导入边缘约束ConFIG优化器
from core.inverse_lithography_config import inverse_lithography_optimization_edge_constrained


def main():
    start_time = time.time()

    # 1. 加载图像
    print("Loading images...")
    initial_mask = load_image(INITIAL_MASK_PATH)
    target_image = load_image(TARGET_IMAGE_PATH)

    target_name = os.path.basename(TARGET_IMAGE_PATH)
    experiment_tag = os.path.splitext(target_name)[0]

    # 2. 初始状态评估
    print("Running initial lithography simulation...")
    aerial_init = hopkins_digital_lithography_simulation(initial_mask)
    resist_init = photoresist_model(aerial_init)
    pe_init = pe_loss(target_image, resist_init)
    epe_init = epe_loss(target_image, resist_init)

    # 3. 边缘约束ConFIG优化
    print("\n" + "=" * 70)
    print(">>> Starting Edge-Constrained ConFIG Inverse Lithography Optimization")
    print("=" * 70)

    # 配置选项
    USE_MOMENTUM = True  # 是否使用动量
    EDGE_PIXEL_RANGE = 5  # 边缘像素范围

    print(f"Configuration:")
    print(f"  - Edge Pixel Range: {EDGE_PIXEL_RANGE} pixels")
    print(f"  - Use Momentum: {USE_MOMENTUM}")
    print(f"  - Beta1: {0.9 if USE_MOMENTUM else 'N/A'}")
    print(f"  - Beta2: {0.999 if USE_MOMENTUM else 'N/A'}")
    print("-" * 70)

    # 运行边缘约束优化
    final_mask, history, update_mask = inverse_lithography_optimization_edge_constrained(
        initial_mask=initial_mask,
        target_image=target_image,
        learning_rate=0.01,
        max_iterations=500,
        use_momentum=USE_MOMENTUM,
        beta_1=0.9,
        beta_2=0.999,
        edge_pixel_range=EDGE_PIXEL_RANGE,
        log_csv=True,
        experiment_tag=f"{experiment_tag}_edge_constrained",
        log_dir="logs"
    )

    # 4. 最终结果评估
    print("\nRunning final evaluation...")
    aerial_best = hopkins_digital_lithography_simulation(final_mask)
    resist_best = photoresist_model(aerial_best)
    pe_final = pe_loss(target_image, resist_best)
    epe_final = epe_loss(target_image, resist_best)

    end_time = time.time()

    # 5. 输出统计
    print("\n" + "=" * 70)
    print("EDGE-CONSTRAINED CONFIG OPTIMIZATION RESULTS")
    print("=" * 70)
    print(
        f"Optimization Mode: {'Momentum-ConFIG' if USE_MOMENTUM else 'Base-ConFIG'} (Edge Range: {EDGE_PIXEL_RANGE}px)")
    print(f"Total Process Time: {end_time - start_time:.2f}s")
    print(f"PE  Loss: {pe_init:.2f} -> {pe_final:.2f} (Improvement: {pe_init - pe_final:.2f})")
    print(f"EPE Loss: {epe_init:.4f} -> {epe_final:.4f} (Improvement: {epe_init - epe_final:.4f})")

    # 输出更新区域统计
    if 'update_region_size' in history:
        avg_update = np.mean(history['update_region_size'])
        max_update = np.max(history['update_region_size'])
        min_update = np.min(history['update_region_size'])
        print(f"\nUpdate Region Analysis:")
        print(f"  Average: {avg_update:.1f}% of pixels updated")
        print(f"  Maximum: {max_update:.1f}% of pixels updated")
        print(f"  Minimum: {min_update:.1f}% of pixels updated")

    # 输出梯度冲突分析
    if 'grad_conflicts' in history:
        avg_conflict = sum(history['grad_conflicts']) / len(history['grad_conflicts'])
        max_conflict = max(history['grad_conflicts'])
        min_conflict = min(history['grad_conflicts'])
        print(f"\nGradient Conflict Analysis:")
        print(f"  Average: {avg_conflict:.3f}")
        print(f"  Maximum: {max_conflict:.3f}")
        print(f"  Minimum: {min_conflict:.3f}")
        print(f"  (0 = no conflict, 1 = maximum conflict)")

    print("=" * 70)

    # 6. 保存结果
    print(f"\nSaving optimized mask to {OUTPUT_MASK_PATH}...")
    save_image(final_mask, OUTPUT_MASK_PATH)

    # 保存更新掩膜
    update_mask_path = OUTPUT_MASK_PATH.replace('.png', '_update_mask.png')
    print(f"Saving update region mask to {update_mask_path}...")
    save_image(update_mask, update_mask_path)

    # 7. 可视化
    print(f"\nGenerating visualizations...")

    # 标准对比图
    print(f"Saving comparison plot to {RESULTS_IMAGE_PATH}...")
    plot_comparison(
        target_image, aerial_init, resist_init,
        final_mask, aerial_best, resist_best,
        pe_init, pe_final, epe_init, epe_final,
        save_path=RESULTS_IMAGE_PATH
    )

    # 损失历史图
    print(f"Saving loss history plot to {FITNESS_PLOT_PATH}...")

    # 准备可视化数据
    history_enhanced = history.copy()
    additional_metrics = {
        'Gradient Conflict': history['grad_conflicts']
    }

    # 添加更新区域信息
    if 'update_region_size' in history:
        additional_metrics['Update Region %'] = history['update_region_size']

    history_enhanced['additional_metrics'] = additional_metrics

    plot_dual_axis_loss_history(history_enhanced, save_path=FITNESS_PLOT_PATH)

    # 边缘约束可视化
    edge_vis_path = FITNESS_PLOT_PATH.replace('.png', '_edge_constraint.png')
    print(f"Saving edge constraint visualization to {edge_vis_path}...")
    plot_edge_constraint_visualization(
        target_image=target_image,
        initial_mask=initial_mask,
        final_mask=final_mask,
        update_mask=update_mask,
        edge_pixel_range=EDGE_PIXEL_RANGE,
        save_path=edge_vis_path
    )

    # 背景区域稳定性分析
    print("\nBackground Stability Analysis:")

    # 检测背景区域（目标图像中接近0的区域）
    background_mask = (target_image < 0.1).astype(np.float64)
    background_pixels = np.sum(background_mask > 0.5)

    if background_pixels > 0:
        # 计算初始掩膜和最终掩膜在背景区域的差异
        background_change = np.abs(final_mask - initial_mask) * background_mask
        avg_background_change = np.sum(background_change) / background_pixels
        max_background_change = np.max(background_change)

        print(f"  Background pixels: {background_pixels} ({background_pixels / target_image.size * 100:.1f}% of total)")
        print(f"  Average change in background: {avg_background_change:.6f}")
        print(f"  Maximum change in background: {max_background_change:.6f}")

        if avg_background_change < 0.01:
            print("  ✓ Background is well preserved (low change)")
        else:
            print("  ⚠ Background has noticeable changes")
    else:
        print("  No significant background region detected")

    print("\nOptimization completed successfully!")


if __name__ == "__main__":
    main()