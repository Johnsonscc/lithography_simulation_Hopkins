# 姓名：唐远卓
# 开发时间：2023-06-1714:00
import logbook as logbook
import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import imageio.v2 as iio
from skimage.io import imsave
from skimage.color import rgb2gray
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
from matplotlib import rcParams
import time
# 开始时间
start_time = time.time()
rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为SimHei
rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# Step 1: 导入初始掩膜图像和目标图像
initial_mask = iio.imread("../lithography_simulation_Hopkins/data/input/test00.png")
if len(initial_mask.shape)>2:
    initial_mask = rgb2gray(initial_mask)

target_image = iio.imread("../lithography_simulation_Hopkins/data/input/test00.png")
if len(target_image.shape)>2:
    target_image = rgb2gray(target_image)

# Step 2: 定义光刻仿真函数
def transfer_function(fx, fy, lambda_, z, n):
    # 公式中包含指数和复数计算，用于模拟光波的相位变化
    H = np.exp(-1j * np.pi * lambda_ * z * (fx ** 2 + fy ** 2) / n ** 2)
    return H
'''这个函数模拟光波通过光刻系统时的相位变化，使用了指数函数和复数。公式 H = e^(-i * π * λ * z * (fx^2 + fy^2) / n^2) 是一个常见的光学传递函数公式，
其中 λ 是波长，z 是距离，n 是折射率，fx 和 fy 是空间频率坐标。'''
# 光源函数 J(f, g)，描述光源在各个传播方向的光波的光强分布
def light_source_function(fx, fy, sigma, NA, lambda_):
    # 根据频域坐标和系统参数计算光源函数
    condition = (fx**2 + fy**2) <= (sigma * NA / lambda_)**2
    J = np.where(condition, (lambda_**2) / (np.pi * (sigma * NA)**2), 0)
    return J
'''这个函数描述光源在不同方向的光强分布。条件 (fx^2 + fy^2) <= (σ * NA / λ)^2 决定了光源分布的有效范围，其中 σ 是部分相干因子，NA 是数值孔径。'''
# 脉冲响应函数 P(f, g)，反映透镜作为空间频率低通滤波器的特性
def impulse_response_function(fx, fy, NA, lambda_):
    # 根据频域坐标和系统参数计算脉冲响应函数
    condition = (fx**2 + fy**2) <= (NA / lambda_)**2
    P = np.where(condition, (lambda_**2) / (np.pi * NA**2), 0)
    return P
'''这个函数反映了透镜作为低通滤波器的特性。它根据空间频率和光学参数决定脉冲响应的范围和强度。'''
# 计算交叉传输系数 TCC，描述了从照明光源到像平面的整个光学成像系统的作用
def compute_tcc(J, P, fx, fy):
    # 使用卷积积分来计算TCC
    tcc = np.convolve(J(fx, fy) * P(fx, fy), J(fx, fy) * P(fx, fy), mode='same')
    return tcc
'''TCC是描述从照明光源到像平面的整个光学成像系统作用的系数。这个函数通过卷积积分来计算TCC，涉及光源函数和脉冲响应函数。'''
# 霍普金斯数字光刻仿真的主函数
def hopkins_digital_lithography_simulation(mask, lambda_, Lx, Ly, z, dx, dy, n, sigma, NA):
    # 计算空间频率坐标
    fx = np.linspace(-0.5/dx, 0.5/dx, Lx)
    fy = np.linspace(-0.5/dy, 0.5/dy, Ly)
    # 生成光源函数和脉冲响应函数
    J = lambda fx, fy: light_source_function(fx, fy, sigma, NA, lambda_)
    P = lambda fx, fy: impulse_response_function(fx, fy, NA, lambda_)
    # 计算TCC
    tcc = compute_tcc(J, P, fx, fy)
    # 将掩模图像转换到频域
    M_fft = fftshift(fft2(mask))
    # 应用TCC和传递函数
    filtered_fft = M_fft * tcc
    H = transfer_function(fx, fy, lambda_, z, n)
    result_fft = filtered_fft * H
    # 计算最终的图像并转换回空间域
    result = ifft2(ifftshift(result_fft))
    return np.abs(result)
'''这个函数是仿真的核心，它将掩膜图像转换到频域，应用计算得到的TCC和传递函数，然后将结果转换回空间域，得到最终的仿真图像。'''
# 参数设置
lambda_ = 405  # 波长（单位：纳米）
z = 803000000  # 距离（单位：纳米），这里是假设值
dx = dy = 7560  # 每个微镜的尺寸（单位：纳米）
Lx = Ly = 2048  # 图像尺寸（单位：像素）
n = 1.5  # 折射率（无量纲）
sigma = 0.5  # 部分相干因子（无量纲）
NA = 0.5  # 数值孔径（无量纲）
def binarize_image(image, threshold):
    return (image > threshold).astype(np.uint8)

# Step 3: 对初始掩膜进行光刻仿真，并计算图形偏差PE
simulated_image_initial = hopkins_digital_lithography_simulation(initial_mask, lambda_, Lx, Ly, z, dx, dy, n, sigma, NA)
threshold = 0.5 * np.max(simulated_image_initial)  # 定义阈值
binary_image_initial = binarize_image(simulated_image_initial, threshold)
PE_initial = np.sum(np.abs(binary_image_initial.astype(np.float32) - target_image.astype(np.float32)))
print(f'Initial PE: {PE_initial}')
# 结束时间
end_time = time.time()
print('Running time: %.3f seconds' % (end_time - start_time))
# 使用相同的阈值进行二值化
simulated_image = simulated_image_initial
initial_binary_simulated_image = binarize_image(simulated_image, threshold)

'''
#引入噪声到初始种群
def create_individual_with_noise(initial_mask_flat, noise_scale=0.02):
    noise = np.random.normal(0, noise_scale, size=initial_mask_flat.shape)
    individual = initial_mask_flat + noise
    return np.clip(individual, 0, 1)  # 确保个体中的值在0和1之间
# Step 4: 使用遗传算法进行掩膜优化
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.rand)
# 使用原始掩膜和噪声创建个体
toolbox.register("individual", tools.initIterate, creator.Individual,
                 lambda: create_individual_with_noise(initial_mask.flatten()))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evalPE(individual):
    mask = np.array(individual, dtype=np.float32).reshape((Lx, Ly))
    simulated_image = hopkins_digital_lithography_simulation(mask, lambda_, Lx, Ly, z, dx, dy, n, sigma, NA)
    binary_image = binarize_image(simulated_image, threshold)
    PE = np.mean((binary_image.astype(np.float32) - target_image.astype(np.float32)) ** 2)
    return PE,

toolbox.register("evaluate", evalPE)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.02)
toolbox.register("select", tools.selTournament, tournsize=3)

pop = toolbox.population(n=200)
hof = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.4, mutpb=0.4 , ngen=10, stats=stats, halloffame=hof, verbose=True)



# Step 5: 输出优化后的掩膜，并进行光刻仿真，比较优化前后的图形偏差PE
best_mask = np.array(hof[0], dtype=np.float32).reshape((Lx, Ly))
best_simulated_image = hopkins_digital_lithography_simulation(best_mask, lambda_, Lx, Ly, z, dx, dy, n, sigma, NA)
best_binary_image = binarize_image(best_simulated_image, threshold)
PE_best = np.sum(np.abs(best_binary_image.astype(np.float32) - target_image.astype(np.float32)))

print(f'Initial PE: {PE_initial}, Best PE: {PE_best}')

# Step 6: 保存优化后的掩膜图像
best_mask_normalized = (255 * (best_mask - best_mask.min()) / (best_mask.max() - best_mask.min())).astype(np.uint8)
# imsave("optimized_mask00.bmp", best_mask_normalized)


# 初始掩膜进行仿真后的二值图像
initial_binary_simulated_image = binarize_image(simulated_image, threshold)

# 优化后的二值图像
optimized_binary_mask = binarize_image(best_mask, threshold)

# 优化后的图像进行光刻仿真后的二值图像
optimized_binary_simulated_image = binarize_image(best_simulated_image, threshold)
'''

# 设置图像、文字的清晰度以及字体
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 20
plt.rcParams['font.family'] = 'Times New Roman'
# Display the images
plt.figure(figsize=(24, 18))
plt.subplot(231)
plt.imshow(target_image, cmap='gray')
plt.title('Original Image')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')

# Original Image after Exposure
plt.subplot(232)
plt.imshow(simulated_image, cmap='gray')
plt.title('Image after Exposure from Original Mask')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')

# Binary Image after Exposure from Original Mask
plt.subplot(233)
plt.imshow(initial_binary_simulated_image, cmap='gray')
plt.title('Binary Image after Exposure from Original Mask')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.text(0.05, 0.95, f'PE = {PE_initial:.2f}', transform=plt.gca().transAxes,
         bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

'''
# Optimized Mask
plt.subplot(234)
plt.imshow(best_mask, cmap='gray')
plt.title('Optimized Mask')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')

# Image after Exposure from Optimized Mask
plt.subplot(235)
plt.imshow(best_simulated_image, cmap='gray')
plt.title('Image after Exposure from Optimized Mask')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')

# Binary Image after Exposure from Optimized Mask
plt.subplot(236)
plt.imshow(optimized_binary_simulated_image, cmap='gray')
plt.title('Binary Image after Exposure from Optimized Mask')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
'''

plt.tight_layout()
#plt.text(0, optimized_binary_simulated_image.shape[0], f'PE : {PE_best}', verticalalignment='bottom', horizontalalignment='left', color='red')  # Add PE value to the image
plt.savefig("1.png")
plt.tight_layout()  # Ensure spacing between subplots

plt.show()

# Plotting the fitness evolution
plt.figure()
minFitnessValues, meanFitnessValues = log.select("min", "avg")
plt.plot(minFitnessValues, color='red', label='Min Fitness')
plt.plot(meanFitnessValues, color='green', label='Average Fitness')
plt.xlabel('Generation')
plt.ylabel('Min / Average Fitness')
plt.title('Min and Average Fitness over Generations')
plt.legend(loc='upper right')
plt.tight_layout()
# plt.savefig("2.png")
plt.show()
'''
gen: 这代表当前的"世代"或"代数"。在遗传算法中，我们通过迭代的方式进行优化，每一次迭代我们会生成一个新的"世代"的候选解，并通过选择、交叉、突变等操作对这些候选解进行优化。因此，gen代表了当前是第几次迭代或者说是第几个"世代"。

nevals: 这代表在当前世代中，进行了多少次适应度函数的评估。在遗传算法中，我们需要评估每一个候选解的适应度，以决定其在后续的选择、交叉、突变等操作中的"生存"机会。因此，nevals代表了我们在当前世代中评估了多少次适应度函数。

avg: 这代表当前世代中所有候选解的适应度的平均值。适应度的定义取决于具体的问题，一般来说，适应度越高代表该候选解越优秀。因此，avg可以用来衡量当前世代的平均优秀程度。

std: 这代表当前世代中所有候选解的适应度的标准差。标准差用来衡量数据的离散程度，标准差越大，代表当前世代中候选解的优秀程度差距越大；标准差越小，代表当前世代中候选解的优秀程度差距越小。

min: 这代表当前世代中所有候选解的适应度的最小值。在最小化问题中，适应度的最小值对应着最优解。

max: 这代表当前世代中所有候选解的适应度的最大值。在最大化问题中，适应度的最大值对应着最优解。
'''