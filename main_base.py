import time
import os
from config.parameters import *
from utils.image_processing import load_image, save_image
from core.evaluation_function import pe_loss, epe_loss
from utils.visualization import plot_comparison, plot_dual_axis_loss_history
from core.inverse_lithography_base import InverseLithographyOptimizer

def main():
    start_time = time.time()

    # 1. 加载图像
    print("Loading images...")
    initial_mask = load_image(INITIAL_MASK_PATH)
    target_image = load_image(TARGET_IMAGE_PATH)

    target_name = os.path.basename(TARGET_IMAGE_PATH)
    experiment_tag = os.path.splitext(target_name)[0]

    # 2. 实例化优化器 (这会预计算 TCC SVD，确保全局唯一性)
    optimizer = InverseLithographyOptimizer(
        optimizer_type='momentum'
    )

    # 3. 进行初始状态评估
    print("Running initial evaluation (Internal)...")
    pe_init, epe_init, _, aerial_init, resist_init = optimizer._compute_analytical_gradient(
        initial_mask, target_image
    )
    # 计算 EPE
    epe_init = epe_loss(target_image, resist_init)

    # 4. 执行优化 (直接调用实例的 optimize 方法)
    print("\nStarting optimization process...")
    final_mask, history = optimizer.optimize(
        initial_mask=initial_mask,
        target=target_image,
        learning_rate=0.01,
        max_iterations=300,
        log_csv=True,
        experiment_tag=f"{experiment_tag}_base",
        log_dir="logs"
    )

    # 5. 最终结果评估 (使用相同的 optimizer 实例和相同的 SVD 核)
    print("\nRunning final evaluation (Internal)...")
    pe_final, epe_final, _, aerial_best, resist_best = optimizer._compute_analytical_gradient(
        final_mask, target_image
    )
    epe_final = epe_loss(target_image, resist_best)

    end_time = time.time()

    # 6. 输出统计
    print("-" * 50)
    print(f"Total Process Time: {end_time - start_time:.2f}s")
    print(f"PE Improvement:     {pe_init:.2f} -> {pe_final:.2f}")
    print(f"EPE Improvement:   {epe_init:.4f} -> {epe_final:.4f}")
    print("-" * 50)

    # 7. 保存结果
    save_image(final_mask, OUTPUT_MASK_PATH)

    # 8. 可视化
    print(f"Saving comparison plot to {RESULTS_IMAGE_PATH}...")
    plot_comparison(
        target_image, aerial_init, resist_init,
        final_mask, aerial_best, resist_best,
        pe_init, pe_final, epe_init, epe_final,
        save_path=RESULTS_IMAGE_PATH
    )

    print(f"Saving loss history plot to {FITNESS_PLOT_PATH}...")
    plot_dual_axis_loss_history(history, save_path=FITNESS_PLOT_PATH)

if __name__ == "__main__":
    main()