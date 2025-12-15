import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class DMDOptimizer:
    def __init__(self, target_mask_shape=(20, 20), pixel_size=1.2,
                 grid_range=15, grid_step=0.1, threshold=1.5):
        """
        初始化优化器
        :param target_mask_shape: 掩模尺寸，默认为 20x20 [cite: 1015]
        :param pixel_size: 单个微镜成像后的尺寸，默认为 1.2 um [cite: 402]
        :param grid_range: 仿真区域范围 (-15 to 15 um) [cite: 1322]
        :param grid_step: 仿真网格步长 (0.1 um) [cite: 1322]
        :param threshold: 光刻胶曝光阈值，默认为 1.5 [cite: 1313]
        """
        self.mask_h, self.mask_w = target_mask_shape
        self.pixel_size = pixel_size
        self.threshold = threshold

        # 建立坐标系
        x = np.arange(-grid_range, grid_range + grid_step, grid_step)
        y = np.arange(-grid_range, grid_range + grid_step, grid_step)
        self.XX, self.YY = np.meshgrid(x, y)

        # 预计算每个像素的光学点扩散函数 (PSF) 以加速计算

        print("正在预计算光学基础矩阵 (Pre-computing optical basis)...")
        self.optical_basis = self._precompute_basis()
        print("初始化完成。")

    def _precompute_basis(self):
        """
        预先计算20x20个像素中每一个像素单独点亮时在基底上的光强分布。
        
        """
        num_pixels = self.mask_h * self.mask_w
        basis = np.zeros((num_pixels, self.XX.shape[0], self.XX.shape[1]), dtype=np.float32)

        idx = 0
        # 注意：这里使用行优先顺序，与Python习惯一致
        for r in range(self.mask_h):
            for c in range(self.mask_w):


                center_x = (c - self.mask_w / 2 + 0.5) * self.pixel_size

                center_y = (self.mask_h / 2 - r - 0.5) * self.pixel_size


                factor = 0.9967 / np.pi

                # Sinc 函数项
                term_x = np.sinc(factor * (self.XX - center_x))  # np.sinc 包含 pi
                term_y = np.sinc(factor * (self.YY - center_y))

                basis[idx, :, :] = (term_x ** 2) * (term_y ** 2)
                idx += 1
        return basis

    def generate_ideal_image(self, mask):
        """
        生成理想的曝光图形（目标图形）
        对应附录代码2 [cite: 1313]
        """
        ideal_img = np.zeros_like(self.XX)
        # 简单的矩形叠加，模拟理想成像
        for r in range(self.mask_h):
            for c in range(self.mask_w):
                if mask[r, c] > 0:
                    center_x = (c - self.mask_w / 2 + 0.5) * self.pixel_size
                    center_y = (self.mask_h / 2 - r - 0.5) * self.pixel_size
                    half_w = 0.6  # 1.2 / 2

                    # 创建矩形区域
                    rect = (np.abs(self.XX - center_x) <= half_w) & \
                           (np.abs(self.YY - center_y) <= half_w)
                    ideal_img += rect.astype(float) * self.threshold
        return ideal_img

    def calculate_fitness(self, population, ideal_img):
        """
        计算种群适应度

        """
        pop_size = population.shape[0]
        fitness = np.zeros(pop_size)

        # 利用矩阵乘法一次性计算所有个体的光强分布
        # (N, 400) @ (400, H*W) -> (N, H*W)
        basis_flat = self.optical_basis.reshape(self.optical_basis.shape[0], -1)
        intensity_flat = np.dot(population, basis_flat)

        ideal_binary = (ideal_img >= self.threshold).astype(int)

        for i in range(pop_size):
            # 还原为2D图像
            I_img = intensity_flat[i].reshape(self.XX.shape)

            # 阈值处理
            I_binary = (I_img >= self.threshold).astype(int)

            # 计算适应度: 异或区域面积 (Union - Intersection)

            or_pixel = np.sum(np.logical_or(I_binary, ideal_binary))
            and_pixel = np.sum(np.logical_and(I_binary, ideal_binary))

            fitness[i] = or_pixel - and_pixel

        return fitness

    def run_genetic_algorithm(self, initial_mask, generations=1000, pop_size=50,
                              crossover_prob=0.9, mutation_prob_start=0.1):
        """
        运行遗传算法主循环

        """
        gene_num = self.mask_h * self.mask_w

        # 1. 初始化种群
        initial_gene = initial_mask.flatten()
        population = np.tile(initial_gene, (pop_size, 1)).astype(np.float32)

        # --- 关键修正：确定允许变异的基因索引 ---
        # 只有初始掩模中非零的区域才允许变异，背景保持纯黑

        valid_indices = np.where(initial_gene > 0)[0]

        # 计算初始理想图像
        ideal_img = self.generate_ideal_image(initial_mask)

        fitness_history = []
        best_mask = None

        # 初始适应度
        fitness = self.calculate_fitness(population, ideal_img)
        sorted_idx = np.argsort(fitness)
        population = population[sorted_idx]
        fitness = fitness[sorted_idx]

        best_mask = population[0].reshape(self.mask_h, self.mask_w)

        # 迭代循环
        pbar = tqdm(range(generations), desc="OPC Optimization")
        for g in range(generations):
            pop_old = population.copy()
            fit_old = fitness.copy()

            # --- 变异操作 (Mutation) ---
            # 变异概率随迭代次数线性递减
            current_mut_prob = mutation_prob_start * (1 - g / generations)

            # 1. 生成全局随机变异矩阵
            random_matrix = np.random.rand(pop_size, gene_num)
            mutation_mask = random_matrix < current_mut_prob

            # 2. 生成随机增减量 (随机值 imvar)
            delta = np.random.rand(pop_size, gene_num)
            signs = np.where(np.random.rand(pop_size, gene_num) > 0.5, 1, -1)

            # 3. --- 应用变异限制 ---
            # 创建一个全 False 的 mask
            restricted_mutation_mask = np.zeros_like(mutation_mask, dtype=bool)
            # 仅将有效区域(图形区域)内的变异标志复制过来
            restricted_mutation_mask[:, valid_indices] = mutation_mask[:, valid_indices]

            # 4. 应用变异
            population[restricted_mutation_mask] += signs[restricted_mutation_mask] * delta[restricted_mutation_mask]

            # 5. 截断到 [0, 1] 区间
            population = np.clip(population, 0.0, 1.0)

            # --- 交叉操作 (Crossover) ---

            for m in range(0, pop_size - 1, 2):
                if np.random.rand() < crossover_prob:
                    # 基因交叉
                    exchange_mask = np.random.rand(gene_num) < 0.5
                    temp = population[m, exchange_mask].copy()
                    population[m, exchange_mask] = population[m + 1, exchange_mask]
                    population[m + 1, exchange_mask] = temp

            # --- 选择下一代 (Selection) ---
            fit_new = self.calculate_fitness(population, ideal_img)

            # 合并新旧种群并保留前N个最优个体
            pop_combined = np.vstack((pop_old, population))
            fit_combined = np.concatenate((fit_old, fit_new))

            sorted_idx = np.argsort(fit_combined)
            population = pop_combined[sorted_idx[:pop_size]]
            fitness = fit_combined[sorted_idx[:pop_size]]

            fitness_history.append(fitness[0])
            best_mask = population[0].reshape(self.mask_h, self.mask_w)

            pbar.set_postfix({'Best Fitness': f"{fitness[0]:.2f}"})

        return best_mask, fitness_history


# ================= 使用示例 =================

if __name__ == "__main__":
    # 1. 定义初始掩模 (例如 "L" 形)
    initial_mask = np.zeros((20, 20))
    # 绘制一个L形 (宽度2像素) [cite: 1039]
    initial_mask[4:16, 4:6] = 1  # 竖条
    initial_mask[14:16, 4:16] = 1  # 横条

    # 2. 实例化优化器
    optimizer = DMDOptimizer(target_mask_shape=(20, 20))

    # 3. 运行遗传算法

    optimized_mask, history = optimizer.run_genetic_algorithm(
        initial_mask,
        generations=1000,
        pop_size=50,
        crossover_prob=0.9,
        mutation_prob_start=0.1
    )

    # 4. 结果可视化
    # 将浮点数掩模量化为8位灰度
    final_mask_uint8 = (optimized_mask * 255).astype(np.uint8)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 3, 1)
    plt.title("Initial Mask (Binary)")
    plt.imshow(initial_mask, cmap='gray', vmin=0, vmax=1)
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.title("Optimized Mask (Grayscale)")
    plt.imshow(final_mask_uint8, cmap='gray', vmin=0, vmax=255)
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.title("Convergence Curve")
    plt.plot(history)
    plt.xlabel("Generation")
    plt.ylabel("Fitness (Error)")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    print("优化完成。")