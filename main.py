import time
import os
from config.parameters import *
from utils.image_processing import load_image, save_image
from core.lithography_simulation import hopkins_digital_lithography_simulation, photoresist_model
from core.evaluation_function import pe_loss, epe_loss
from utils.visualization import plot_comparison, plot_dual_axis_loss_history

# 导入ConFIG优化器
from core.inverse_lithography_config import inverse_lithography_optimization_config


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

    # 3. ConFIG优化（可选择使用动量）
    print("\n" + "=" * 70)
    print(">>> Starting ConFIG Inverse Lithography Optimization")
    print("=" * 70)

    # 配置选项
    USE_MOMENTUM = True  # 设置为True使用动量ConFIG，False使用基础ConFIG
    BETA_1 = 0.9  # 一阶动量衰减率
    BETA_2 = 0.999  # 二阶动量衰减率

    if USE_MOMENTUM:
        print("Mode: Momentum ConFIG")
        print(f"Parameters: beta1={BETA_1}, beta2={BETA_2}")
    else:
        print("Mode: Base ConFIG (no momentum)")

    print("-" * 70)

    final_mask, history = inverse_lithography_optimization_config(
        initial_mask=initial_mask,
        target_image=target_image,
        learning_rate=0.03,
        max_iterations=1000,
        use_momentum=USE_MOMENTUM,
        beta_1=BETA_1,
        beta_2=BETA_2,
        log_csv=True,
        experiment_tag=f"{experiment_tag}_config{'m' if USE_MOMENTUM else ''}",
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
    print("CONFIG OPTIMIZATION RESULTS")
    print("=" * 70)
    print(f"Optimization Mode: {'Momentum-ConFIG' if USE_MOMENTUM else 'Base-ConFIG'}")
    print(f"Total Process Time: {end_time - start_time:.2f}s")
    print(f"PE  Loss: {pe_init:.2f} -> {pe_final:.2f} (Improvement: {pe_init - pe_final:.2f})")
    print(f"EPE Loss: {epe_init:.4f} -> {epe_final:.4f} (Improvement: {epe_init - epe_final:.4f})")

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

    # 输出动量信息（如果使用动量）
    if USE_MOMENTUM and 'momentum_norms' in history:
        avg_momentum = sum(history['momentum_norms']) / len(history['momentum_norms'])
        max_momentum = max(history['momentum_norms'])
        min_momentum = min(history['momentum_norms'])
        print(f"\nMomentum Analysis:")
        print(f"  Average Norm: {avg_momentum:.4f}")
        print(f"  Maximum Norm: {max_momentum:.4f}")
        print(f"  Minimum Norm: {min_momentum:.4f}")

    print("=" * 70)

    # 6. 保存结果
    print(f"\nSaving optimized mask to {OUTPUT_MASK_PATH}...")
    save_image(final_mask, OUTPUT_MASK_PATH)

    # 7. 可视化
    print(f"Saving comparison plot to {RESULTS_IMAGE_PATH}...")
    plot_comparison(
        target_image, aerial_init, resist_init,
        final_mask, aerial_best, resist_best,
        pe_init, pe_final, epe_init, epe_final,
        save_path=RESULTS_IMAGE_PATH
    )

    print(f"Saving loss history plot to {FITNESS_PLOT_PATH}...")

    # 准备可视化数据
    history_enhanced = history.copy()
    additional_metrics = {
        'Gradient Conflict': history['grad_conflicts']
    }

    # 如果使用动量，添加动量信息
    if USE_MOMENTUM and 'momentum_norms' in history:
        additional_metrics['Momentum Norm'] = history['momentum_norms']

    history_enhanced['additional_metrics'] = additional_metrics

    plot_dual_axis_loss_history(history_enhanced, save_path=FITNESS_PLOT_PATH)



if __name__ == "__main__":
    main()