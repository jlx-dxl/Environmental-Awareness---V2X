# @Author  : 贾林轩 1853688
# @IDE: PyCharm
# @python：3.9.7
# function：存放滤波函数以供调用（包括3x3sobel滤波、3x3导数滤波、7x7双边滤波及7x7引导滤波）

import cv2
import numpy as np

# 用于sobel和derivative的卷积核库
def Operator(roi, operator_type):
    if operator_type == "sobel_horizontal":
        operator = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / 8   # 用于提取横向边界的sobel滤波器卷积核
    elif operator_type == "sobel_vertical":
        operator = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8   # 用于提取纵向边界的sobel滤波器卷积核
    elif operator_type == "derivative_horizontal":
        operator = np.array([[0, 1, 0], [0, 0, 0], [0, -1, 0]])   # 用于提取横向边界的derivative滤波器卷积核
    elif operator_type == "derivative_vertical":
        operator = np.array([[0, 0, 0], [1, 0, -1], [0, 0, 0]])   # 用于提取纵向边界的derivative滤波器卷积核
    else:
        raise "type Error"
    result = np.abs(np.sum(roi * operator))   # 卷积并取绝对值（sobel和derivative核卷积会出现负值像素，表示梯度的方向）
    return result

# sobel滤波函数接口（参数：输入图片，滤波类型（横or纵））
def Sobel(image, operator_type):
    new_image = np.zeros((image.shape[0] - 2, image.shape[1] - 2))   # 建立zero矩阵用于储存卷积结果（由于核大小为3*3，故上下左右各缩小一个像素）
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):   # 使卷积核遍历整个图片（考虑到核的大小，卷积核中心不会去往最外一圈像素）
            new_image[i - 1, j - 1] = Operator(image[i - 1:i + 2, j - 1:j + 2], operator_type)   # 从库中取卷积核并进行卷积操作
    new_image = cv2.copyMakeBorder(new_image, 1, 1, 1, 1, cv2.BORDER_REPLICATE)   # 以复制边缘像素点的方式扩展卷积结果到原图像大小
    return new_image.astype(np.uint8)   # 以uint8类型返回图像

# 导数滤波函数接口（参数：输入图片，滤波类型（横or纵））
def Derivative(image, operator_type):
    new_image = np.zeros((image.shape[0] - 2, image.shape[1] - 2))   # 建立zero矩阵用于储存卷积结果（由于核大小为3*3，故上下左右各缩小一个像素）
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):   # 使卷积核遍历整个图片（考虑到核的大小，卷积核中心不会去往最外一圈像素）
            new_image[i - 1, j - 1] = Operator(image[i - 1:i + 2, j - 1:j + 2], operator_type)   # 从库中取卷积核并进行卷积操作
    new_image = cv2.copyMakeBorder(new_image, 1, 1, 1, 1, cv2.BORDER_REPLICATE)   # 以复制边缘像素点的方式扩展卷积结果到原图像大小
    return new_image.astype(np.uint8)   # 以uint8类型返回图像

# 双边滤波函数接口（参数：输入图像，radius:滤波器窗口半径，sigma_color:颜色域方差，sigma_space:空间域方差）
def Bilateral(image, radius, sigma_color, sigma_space):
    H, W = image.shape[0], image.shape[1]   # H，W储存输入图像的高和宽
    new_image = np.zeros((H - 2 * radius, W - 2 * radius))   # 建立zeros矩阵用于储存卷积结果（由于上下左右各缩小radius个像素）
    # 首先计算空间域的权重系数（空间域权重系数并不会根据像素点的遍历过程而改变，因此在遍历像素进行卷积前事先计算可以缩减程序运行时间）
    for x in range(-radius, radius + 1):
        for y in range(-radius, radius + 1):   # 遍历窗口内像素
            spatial_weight = -(x ** 2 + y ** 2) / (2 * (sigma_space ** 2))   # 计算空间域权重系数
    # 进行卷积操作
    for i in range(radius, H - radius):
        for j in range(radius, W - radius):   # 使卷积窗口遍历整个图像（考虑到核的大小，卷积核中心不会去往最外一圈（距离radius）的像素）
            weight_sum = 0.0
            pixel_sum = 0.0
            for x in range(-radius, radius + 1):
                for y in range(-radius, radius + 1):   # 在每一个窗口位置下遍历窗口内像素计算颜色域权重系数
                    # 计算颜色域权重
                    color_weight = -(int(image[i][j]) - int(image[i + x][j + y])) ** 2 / (2 * (sigma_color ** 2))
                    # 计算像素整体权重
                    weight = np.exp(spatial_weight + color_weight)
                    # 求权重和，用于归一化
                    weight_sum += weight
                    pixel_sum += (weight * image[i + x][j + y])
            # 计算归一化后的像素值
            value = pixel_sum / weight_sum
            # 将像素值结果储存到对应位置
            new_image[i - radius][j - radius] = value
    new_image = cv2.copyMakeBorder(new_image, radius, radius, radius, radius, cv2.BORDER_REPLICATE)   # 以复制边缘像素点的方式扩展卷积结果到原图像大小
    return new_image.astype(np.uint8)   # 以uint8类型返回图像

# 导向滤波函数接口（参数：输入图像，引导图，卷积窗口大小，正则化参数）
def Guided(srcImg, guidedImg, size, eps):
    # P：输入图像；I：引导图像；a，b：线性系数；
    # 对输入图像进行均值滤波
    P_mean = cv2.boxFilter(srcImg, -1, (size, size), normalize=True, borderType=cv2.BORDER_REPLICATE)
    # 对引导图像进行均值滤波
    I_mean = cv2.boxFilter(guidedImg, -1, (size, size), normalize=True, borderType=cv2.BORDER_REPLICATE)

    # 对输入图像的平方进行均值滤波
    I_square_mean = cv2.boxFilter(np.multiply(guidedImg, guidedImg), -1, (size, size), normalize=True, borderType=cv2.BORDER_REPLICATE)
    # 对输入图像和引导图像的乘积进行均值滤波
    I_mul_P_mean = cv2.boxFilter(np.multiply(srcImg, guidedImg), -1, (size, size), normalize=True, borderType=cv2.BORDER_REPLICATE)

    # 引导图在窗口中的方差（每个像素位置均有值）
    var_I = I_square_mean - np.multiply(I_mean, I_mean)
    # 引导图和输入图像在窗口中的协方差（每个像素位置均有图像）
    cov_I_P = I_mul_P_mean - np.multiply(I_mean, P_mean)

    # a = cov_I_P / (var_I + eps)   # 按论文所述此处应通过协方差与方差（加正则化系数）相除得到最小二乘回归的线性系数，但<br/>
    # 实际操作中发现部分窗口方差为0，无法作为分母计算，考虑到采用原图作为引导图，故在这些情况下将对应的a值设为1（需要遍历进行判断）
    a = cov_I_P.copy()
    for i in range(cov_I_P.shape[0]):
        for j in range(cov_I_P.shape[1]):   # 遍历整个图像
            if var_I[i, j] == 0:
                a[i, j] = 1   # 如果方差为0则将a设为1
            else:
                a[i, j] = cov_I_P[i, j] / (var_I[i, j] + eps)   # 如果方差不为0则正常计算
    # 根据最小二乘原理得到系数b
    b = P_mean - np.multiply(a, I_mean)

    # 由于一个像素会被多个窗口包含，即每个像素由多个线性系数a,b描述，为保证输出的统一性，在窗口内将所有参数值平权（均值滤波）
    a_mean = cv2.boxFilter(a, -1, (size, size), normalize=True, borderType=cv2.BORDER_REPLICATE)
    b_mean = cv2.boxFilter(b, -1, (size, size), normalize=True, borderType=cv2.BORDER_REPLICATE)

    # 根据引导滤波线性原理，计算滤波结果
    dstImg = np.multiply(a_mean, guidedImg) + b_mean
    return dstImg   #返回滤波结果
