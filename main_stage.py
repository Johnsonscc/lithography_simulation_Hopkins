import time
import os
import numpy as np
from config.parameters import *
from utils.image_processing import load_image, save_image
from core.lithography_simulation import hopkins_digital_lithography_simulation, photoresist_model
from core.evaluation_function import pe_loss, mepe_loss
from utils.visualization_two import plot_comparison, plot_dual_axis_loss_history
from core.inverse_lithography_base import inverse_lithography_optimization_base as optimize_pe_stage1
from core.inverse_lithography_epe import inverse_lithography_optimization_epe as optimize_epe_stage2


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
    mepe_init = mepe_loss(target_image, resist_init)

    # Stage 1: PE Optimization (全局拓扑搜索)
    # 策略：使用自适应热启动，当损失不再显著下降时自动切换到下一阶段
    print("\n" + "=" * 50)
    print(">>> Stage 1: PE Optimization (Adaptive Warm Start)")
    print("=" * 50)

    pe_mask_result, pe_history = optimize_pe_stage1(
        initial_mask=initial_mask,
        target_image=target_image,
        optimizer_type='sgd',
        learning_rate=0.01,  # 较大的学习率快速成型
        max_iterations=300,  # 设置较高上限，依靠 patience 提前停止

        enable_adaptive_switch=True,
        patience=20,  # 连续 20 次迭代无明显进展则切换

        log_csv=True,
        experiment_tag=f"{experiment_tag}_stage1",
        log_dir="logs"
    )

    print(f"Stage 1 Best PE: {min(pe_history['pe_loss']):.4f}")

    # Stage 2: EPE Optimization (边缘精修)
    # 策略：接力 Stage 1 的结果，使用小学习率微调边缘

    print("\n" + "=" * 50)
    print(">>> Stage 2: EPE Optimization (Fine-tuning)")
    print("=" * 50)

    final_mask, epe_history = optimize_epe_stage2(
        initial_mask=pe_mask_result,  # 热启动：传入 Stage 1 的结果
        target_image=target_image,
        optimizer_type='cg',
        learning_rate=0.005,  # 降低学习率，防止在极值点附近震荡
        max_iterations=1,
        log_csv=True,
        experiment_tag=f"{experiment_tag}_stage2",
        log_dir="logs"
    )

    # 3. 最终结果评估
    print("\nRunning final evaluation...")
    aerial_best = hopkins_digital_lithography_simulation(final_mask)
    resist_best = photoresist_model(aerial_best)
    pe_final = pe_loss(target_image, resist_best)
    mepe_final = mepe_loss(target_image, resist_best)

    end_time = time.time()

    # 4. 输出统计
    print("-" * 50)
    print(f"Total Process Time: {end_time - start_time:.2f}s")
    print(f"PE Improvement:     {pe_init:.2f} -> {pe_final:.2f}")
    print(f"MEPE Improvement:   {mepe_init:.4f} -> {mepe_final:.4f}")
    print("-" * 50)

    # 5. 保存结果
    save_image(final_mask, OUTPUT_MASK_PATH)

    # 6. 可视化
    plot_comparison(
        target_image, aerial_init, resist_init,
        final_mask, aerial_best, resist_best,
        pe_init, pe_final, mepe_init, mepe_final,
        save_path=RESULTS_IMAGE_PATH
    )
    plot_dual_axis_loss_history(
        pe_history,
        epe_history,
        save_path=FITNESS_PLOT_PATH.replace('.png', '_dual_axis_loss.png')
    )


if __name__ == "__main__":
    main()