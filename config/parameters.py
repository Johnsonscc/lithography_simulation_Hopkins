# 参数配置
import matplotlib.pyplot as plt
# 光刻仿真参数
LAMBDA = 405  # 波长（单位：纳米）
Z = 803000000  # 距离（单位：纳米）774
DX = DY = 7560  # 像素尺寸（单位：纳米）
LX = LY = 100  # 图像尺寸（单位：像素）
N = 1.5  # 折射率（无量纲）
SIGMA = 0.5  # 部分相干因子（无量纲）
NA = 0.4  # 数值孔径（无量纲）
K_SVD = 20 #奇异值数目（无量纲）

# 光刻胶参数
A = 20.0            # sigmoid函数梯度
TR = 0.3            # 阈值参数


# DMD调制参数
WX = 7560  # 微镜宽度（单位：纳米）
WY = 7560  # 微镜高度（单位：纳米）
TX = 8560  # 微镜周期（x方向）（单位：纳米）
TY = 8560  # 微镜周期（y方向）（单位：纳米）

# 逆光刻优化参数
ILT_LEARNING_RATE = 0.01
ILT_K_SVD = 20
ILT_MAX_ITERATIONS=100


# 文件路径
INITIAL_MASK_PATH = "../lithography_simulation_Hopkins/data/input/t100_inverse.png"
TARGET_IMAGE_PATH = "../lithography_simulation_Hopkins/data/input/t100_inverse.png"
OUTPUT_MASK_PATH = "../lithography_simulation_Hopkins/data/output/optimized_mask_t100_inverse.png"
RESULTS_IMAGE_PATH = "../lithography_simulation_Hopkins/data/output/results_comparison_t100_inverse.png"
FITNESS_PLOT_PATH = "../lithography_simulation_Hopkins/data/output/fitness_evolution_t100_inverse.png"

# 可视化参数
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为SimHei
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 20
plt.rcParams['font.family'] = 'Times New Roman'
