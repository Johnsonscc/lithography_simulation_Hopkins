import time
import os
import numpy as np
from config.parameters import *
from utils.image_processing import load_image, save_image
from utils.visualization import plot_comparison, plot_dual_axis_loss_history, plot_edge_constraint_visualization, \
    plot_tcc_structure, plot_loss_and_nils

# 导入边缘约束优化器
from core.inverse_lithography_comb import EdgeConstrainedInverseLithographyOptimizer


def main():
    start_time = time.time()

    # 1. 加载图像
    print("Loading images...")
    initial_mask = load_image(INITIAL_MASK_PATH)
    target_image = load_image(TARGET_IMAGE_PATH)

    target_name = os.path.basename(TARGET_IMAGE_PATH)
    experiment_tag = os.path.splitext(target_name)[0]

    # 2. 配置参数
    EDGE_PIXEL_RANGE = 5  # 边缘像素范围
    OPTIMIZER_TYPE = 'sgd'  # 优化器类型

    print(f"Edge-Constrained PE Optimization Configuration:")
    print(f"  - Edge Pixel Range: {EDGE_PIXEL_RANGE}px")
    print(f"  - Optimizer Type: {OPTIMIZER_TYPE}")
    print("-" * 50)

    # 3. 实例化优化器
    optimizer = EdgeConstrainedInverseLithographyOptimizer(
        optimizer_type=OPTIMIZER_TYPE,
        edge_pixel_range=EDGE_PIXEL_RANGE
    )

    # 4. 进行初始状态评估
    print("Running initial evaluation...")
    _, pe_init, epe_init, _, aerial_init, resist_init, _, _ = optimizer._compute_analytical_gradient(
        initial_mask, target_image
    )

    # 5. 执行优化
    print("\nStarting edge-constrained optimization process...")
    final_mask, history, update_mask = optimizer.optimize(
        initial_mask=initial_mask,
        target=target_image,
        learning_rate=0.1,
        max_iterations=100,
        log_csv=True,
        experiment_tag=f"{experiment_tag}_edge_constrained_base",
        log_dir="logs"
    )

    # 6. 最终结果评估
    print("\nRunning final evaluation...")
    _, pe_final, epe_final, _, aerial_best, resist_best, _, _ = optimizer._compute_analytical_gradient(
        final_mask, target_image
    )

    end_time = time.time()

    # 7. 输出统计
    print("\n" + "=" * 60)
    print("EDGE-CONSTRAINED PE OPTIMIZATION RESULTS")
    print("=" * 60)
    print(f"Edge Pixel Range: {EDGE_PIXEL_RANGE}px")
    print(f"Total Process Time: {end_time - start_time:.2f}s")
    print(f"PE Improvement:     {pe_init:.2f} -> {pe_final:.2f} (Δ={pe_init - pe_final:.2f})")
    print(f"EPE Improvement:   {epe_init:.4f} -> {epe_final:.4f} (Δ={epe_init - epe_final:.4f})")

    # 输出更新区域统计
    if 'update_region_size' in history:
        avg_update = np.mean(history['update_region_size'])
        max_update = np.max(history['update_region_size'])
        min_update = np.min(history['update_region_size'])
        print(f"\nUpdate Region Analysis:")
        print(f"  Average: {avg_update:.1f}% of pixels updated")
        print(f"  Maximum: {max_update:.1f}% of pixels updated")
        print(f"  Minimum: {min_update:.1f}% of pixels updated")

    # 背景区域稳定性分析
    background_mask = (target_image < 0.1).astype(np.float64)
    background_pixels = np.sum(background_mask > 0.5)

    if background_pixels > 0:
        background_change = np.abs(final_mask - initial_mask) * background_mask
        avg_background_change = np.sum(background_change) / background_pixels
        max_background_change = np.max(background_change)

        print(f"\nBackground Stability:")
        print(f"  Background pixels: {background_pixels} ({background_pixels / target_image.size * 100:.1f}% of total)")
        print(f"  Average change: {avg_background_change:.6f}")
        print(f"  Maximum change: {max_background_change:.6f}")

        if avg_background_change < 0.01:
            print("  ✓ Background well preserved")
        else:
            print("  ⚠ Background has noticeable changes")

    print("=" * 60)

    # 8. 保存结果
    save_image(final_mask, OUTPUT_MASK_PATH)

    # 保存更新掩膜
    update_mask_path = OUTPUT_MASK_PATH.replace('.png', '_update_mask.png')
    save_image(update_mask, update_mask_path)
    print(f"Saved optimized mask to {OUTPUT_MASK_PATH}")
    print(f"Saved update region mask to {update_mask_path}")

    # 9. 可视化
    print(f"\nGenerating visualizations...")

    # 标准对比图
    print(f"Saving comparison plot to {RESULTS_IMAGE_PATH}...")
    plot_comparison(
        target_image, aerial_init, resist_init,
        final_mask, aerial_best, resist_best,
        pe_init, pe_final, epe_init, epe_final,
        save_path=RESULTS_IMAGE_PATH
    )

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

    # 新增：PE、EPE和NILS曲线图
    nils_plot_path = FITNESS_PLOT_PATH.replace('.png', '_loss_nils.png')
    print(f"Saving loss and NILS plot to {nils_plot_path}...")
    plot_loss_and_nils(history, save_path=nils_plot_path)

    print("\nEdge-constrained PE optimization completed successfully!")


if __name__ == "__main__":
    main()