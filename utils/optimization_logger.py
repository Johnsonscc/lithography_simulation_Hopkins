# 文件：utils/optimization_logger.py
import csv
import os
import numpy as np
from datetime import datetime


class OptimizationLogger:
    """优化过程CSV日志记录器"""

    def __init__(self, log_dir="../lithography_simulation_Hopkins/data/logs", filename=None):
        """
        初始化日志记录器

        参数:
            log_dir: 日志保存目录
            filename: CSV文件名（默认为自动生成的时间戳）
        """
        # 创建日志目录
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # 生成文件名
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"optimization_log_{timestamp}.csv"

        self.filepath = os.path.join(log_dir, filename)
        self.file = None
        self.writer = None

    def start_logging(self, optimizer_type, loss_type, learning_rate,
                      initial_loss, target_shape, config_params=None):
        """开始新的优化日志"""
        self.file = open(self.filepath, 'w', newline='', encoding='utf-8')
        self.writer = csv.writer(self.file)

        # 写入实验配置信息
        self.writer.writerow(["# ==========================================="])
        self.writer.writerow(["# Inverse Lithography Optimization Log"])
        self.writer.writerow(["# ==========================================="])
        self.writer.writerow([f"# Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"])
        self.writer.writerow([f"# Optimizer Type: {optimizer_type}"])
        self.writer.writerow([f"# Loss Type: {loss_type}"])
        self.writer.writerow([f"# Learning Rate: {learning_rate}"])
        self.writer.writerow([f"# Initial Loss: {initial_loss:.6f}"])
        self.writer.writerow([f"# Target Shape: {target_shape}"])

        if config_params:
            self.writer.writerow(["# Optimizer Parameters:"])
            for key, value in config_params.items():
                self.writer.writerow([f"#   {key}: {value}"])

        self.writer.writerow(["# ==========================================="])
        self.writer.writerow([])  # 空行分隔

        # CSV表头
        headers = ["iteration", "loss", "grad_norm",
                   "mask_min", "mask_max", "mask_mean", "mask_std"]

        # 根据优化器类型添加特定字段
        if optimizer_type == 'momentum':
            headers.append("velocity_norm")
        elif optimizer_type == 'adam':
            headers.extend(["adam_t", "m_norm", "v_norm"])
        elif optimizer_type == 'rmsprop':
            headers.append("square_avg_norm")
        elif optimizer_type == 'cg':
            headers.extend(["direction_norm", "beta"])

        # 添加时间戳（用于分析收敛速度）
        headers.append("time_elapsed")

        self.writer.writerow(headers)
        self.file.flush()

        print(f"CSV日志已创建: {self.filepath}")
        return self.filepath

    def log_iteration(self, iteration, loss, gradient, mask,
                      optimizer_state=None, time_elapsed=0):
        """记录单次迭代数据"""
        # 计算基础指标
        grad_norm = float(np.linalg.norm(gradient))
        mask_min = float(np.min(mask))
        mask_max = float(np.max(mask))
        mask_mean = float(np.mean(mask))
        mask_std = float(np.std(mask))

        # 构建数据行
        row_data = [iteration, float(loss), grad_norm,
                    mask_min, mask_max, mask_mean, mask_std]

        # 添加优化器特定指标
        if optimizer_state is not None:
            if 'velocity' in optimizer_state:
                velocity_norm = float(np.linalg.norm(optimizer_state['velocity']))
                row_data.append(velocity_norm)
            elif 't' in optimizer_state:
                t_val = optimizer_state['t']
                row_data.append(t_val)
                if 'm' in optimizer_state and 'v' in optimizer_state:
                    m_norm = float(np.linalg.norm(optimizer_state['m']))
                    v_norm = float(np.linalg.norm(optimizer_state['v']))
                    row_data.extend([m_norm, v_norm])
            elif 'square_avg' in optimizer_state:
                square_avg_norm = float(np.linalg.norm(optimizer_state['square_avg']))
                row_data.append(square_avg_norm)
            elif 'direction' in optimizer_state:
                direction_norm = float(np.linalg.norm(optimizer_state['direction']))
                # 简化处理beta值
                beta = 0.0
                row_data.extend([direction_norm, beta])

        # 添加时间
        row_data.append(time_elapsed)

        # 写入CSV
        self.writer.writerow(row_data)

        # 定期刷新，确保数据及时写入磁盘
        if iteration % 50 == 0:
            self.file.flush()

    def close(self):
        """关闭日志文件"""
        if self.file:
            self.file.close()
            print(f"CSV日志已保存: {self.filepath}")

    def __enter__(self):
        """支持with语句"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出with语句时自动关闭"""
        self.close()