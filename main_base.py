import time
import os
from config.parameters import *
from utils.image_processing import load_image, save_image
from core.lithography_simulation import hopkins_digital_lithography_simulation, photoresist_model
from core.evaluation_function import pe_loss, mepe_loss
from utils.visualization import plot_comparison,plot_dual_axis_loss_history
from core.inverse_lithography_base import inverse_lithography_optimization_base


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

    final_mask, history = inverse_lithography_optimization_base(
        initial_mask=initial_mask,
        target_image=target_image,

        optimizer_type='momentum',  # 推荐：Momentum 比较稳健
        learning_rate=0.01,  # 单阶段通常可以用适中的学习率
        max_iterations=300,  # 一次性跑完

        log_csv=True,
        experiment_tag=f"{experiment_tag}_base",
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
    # 绘制 EPE 和 PE 的变化曲线

    print(f"Saving comparison plot to {RESULTS_IMAGE_PATH}...")
    plot_comparison(
        target_image, aerial_init, resist_init,
        final_mask, aerial_best, resist_best,
        pe_init, pe_final, mepe_init, mepe_final,
        save_path=RESULTS_IMAGE_PATH
    )

    print(f"Saving loss history plot to {FITNESS_PLOT_PATH}...")
    plot_dual_axis_loss_history(history, save_path=FITNESS_PLOT_PATH)


if __name__ == "__main__":
    main()