import numpy as np

def pe_loss(target_image, printed_image):
    loss = np.sum((target_image - printed_image)**2)
    return loss

def mepe_loss(target_image, printed_image, epsilon=1e-10, gamma_scale = 10.0):

    # 计算梯度返回 (dZ_T/dy, dZ_T/dx)
    grad_y, grad_x = np.gradient(printed_image)
    weights = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # 权重误差的总和 Sum[ (P - Z_T)^2 * w ] 权重总和 Sum[ w ]
    error_squared = (printed_image - target_image) ** 2
    numerator = np.sum(error_squared * weights)
    denominator = np.sum(weights)

    if denominator < epsilon:
        # 如果没有边缘，损失为 0
        return 0.0

    #计算平均边缘放置误差
    loss = numerator / (denominator + epsilon)
    # 引入缩放因子以匹配物理量纲
    loss_scaled = loss * (gamma_scale)  # 乘以 10

    return loss_scaled