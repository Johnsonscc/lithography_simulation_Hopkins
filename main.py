import time
import os
import numpy as np

from config.parameters import *
from utils.image_processing import load_image, save_image
from core.lithography_simulation import hopkins_digital_lithography_simulation, photoresist_model
from core.evaluation_function import pe_loss, epe_loss
from utils.visualization import plot_comparison, plot_dual_axis_loss_history, plot_edge_constraint_visualization

# 导入两种优化器
from core.inverse_lithography_config import (
    inverse_lithography_optimization_base_edge_config,
    inverse_lithography_optimization_momentum_edge_config
)


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

    # 3. 选择优化类型
    OPTIMIZATION_TYPE = "config_momentum_edge"  # 可选: "config_edge" 或 "config_momentum_edge"
    EDGE_PIXEL_RANGE = 5

    print("\n" + "=" * 70)
    if OPTIMIZATION_TYPE == "config_edge":
        print(">>> Starting Edge-Constrained ConFIG Inverse Lithography Optimization")
        print(f"  - Edge Pixel Range: {EDGE_PIXEL_RANGE} pixels")
        print("-" * 70)

        # 运行基础边缘约束优化
        final_mask, history, edge_mask = inverse_lithography_optimization_base_edge_config(
            initial_mask=initial_mask,
            target_image=target_image,
            learning_rate=0.1,
            max_iterations=100,
            edge_pixel_range=EDGE_PIXEL_RANGE,
            log_csv=True,
            experiment_tag=f"{experiment_tag}_edge_config"
        )

    elif OPTIMIZATION_TYPE == "config_momentum_edge":
        print(">>> Starting Momentum Edge-Constrained ConFIG Inverse Lithography Optimization")
        print(f"  - Edge Pixel Range: {EDGE_PIXEL_RANGE} pixels")
        print("-" * 70)

        # 运行动量边缘约束优化
        final_mask, history, edge_mask = inverse_lithography_optimization_momentum_edge_config(
            initial_mask=initial_mask,
            target_image=target_image,
            learning_rate=0.1,
            max_iterations=100,
            edge_pixel_range=EDGE_PIXEL_RANGE,
            log_csv=True,
            experiment_tag=f"{experiment_tag}_momentum_edge_config"
        )

    else:
        raise ValueError(f"Unknown optimization type: {OPTIMIZATION_TYPE}")

    # 4. 最终结果评估
    print("\nRunning final evaluation...")
    aerial_best = hopkins_digital_lithography_simulation(final_mask)
    resist_best = photoresist_model(aerial_best)
    pe_final = pe_loss(target_image, resist_best)
    epe_final = epe_loss(target_image, resist_best)

    end_time = time.time()

    # 5. 输出统计
    print("\n" + "=" * 70)
    print(f"{OPTIMIZATION_TYPE.upper()} OPTIMIZATION RESULTS")
    print("=" * 70)
    print(f"Optimization Type: {OPTIMIZATION_TYPE}")
    print(f"Total Process Time: {end_time - start_time:.2f}s")
    print(f"PE  Loss: {pe_init:.2f} -> {pe_final:.2f} (Improvement: {pe_init - pe_final:.2f})")
    print(f"EPE Loss: {epe_init:.4f} -> {epe_final:.4f} (Improvement: {epe_init - epe_final:.4f})")
    print("=" * 70)

    # 6. 保存结果
    print(f"\nSaving optimized mask to {OUTPUT_MASK_PATH}...")
    save_image(final_mask, OUTPUT_MASK_PATH)

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
    plot_dual_axis_loss_history(history, save_path=FITNESS_PLOT_PATH)

    # 边缘约束可视化
    edge_vis_path = FITNESS_PLOT_PATH.replace('.png', '_edge_constraint.png')
    print(f"Saving edge constraint visualization to {edge_vis_path}...")
    plot_edge_constraint_visualization(
        target_image=target_image,
        initial_mask=initial_mask,
        final_mask=final_mask,
        update_mask=edge_mask,
        edge_pixel_range=EDGE_PIXEL_RANGE,
        save_path=edge_vis_path
    )

    print("\nOptimization completed successfully!")


if __name__ == "__main__":
    main()