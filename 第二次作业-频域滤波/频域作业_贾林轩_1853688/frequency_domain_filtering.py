# @Author  : 贾林轩 1853688
# @IDE: PyCharm
# @python：3.9.7
# @catalogue：
#   0 定义函数接口
#      0.1 生成幅值谱函数接口（generate_magnitude_spectrum）
#      0.2 幅值谱可视化函数接口（magnitude_spectrum_visualization）
#      0.3 滤波过程函数接口（频谱相乘）（frequency_spectrum_multiply）
#      0.4 还原滤波结果函数接口（restore_filtering_results）
#   1 读取图像并预处理
#      1.1 依次读入图像并转化为灰度图
#      1.2 将灰度图转化为float_32格式
#   2 对三张原图做傅里叶变换
#      2.1 调用上述函数接口（0.1）计算幅值谱
#      2.2 调用上述函数接口（0.2）进行幅值谱可视化
#      2.3 储存可视化的幅值谱
#   3 构建滤波器
#      3.1 构建空域滤波器（float32格式）
#      3.2 调用上述函数接口（0.1）计算滤波器的幅值谱
#      3.3 调用上述函数接口（0.2）进行滤波器幅值谱可视化
#      3.4 储存可视化的滤波器幅值谱
#   4 滤波操作
#      4.1 调用上述函数接口（0.3）进行滤波操作（返回结果为频谱）
#      4.2 调用上述函数接口（0.2）将滤波结果频谱可视化
#      4.3 储存可视化的滤波结果频谱
#    5 利用傅里叶反变换（IDFT）将滤波结果还原到空域
#      5.1 调用上述函数接口（0.4）还原滤波结果
#      5.2 储存已还原的滤波结果（空域图像）


import os
import cv2
import numpy as np


# 0 定义函数接口
# 0.1 生成幅值谱函数接口（参数：img：输入图像）
def generate_magnitude_spectrum(img):
    img = cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT)  # 对图像进行傅里叶正向变换
    img = np.fft.fftshift(img)  # 把频谱原点转换到图像中心
    return img  # 返回结果


# 0.2 幅值谱可视化函数接口（参数：img：输入图像）
def magnitude_spectrum_visualization(img):
    result_raw = np.log(cv2.magnitude(img[:, :, 0], img[:, :, 1]) + 1)  # 用对数表示形式表示幅值频谱
    result = (result_raw / result_raw.max()) * 255  # 按最大值归一化
    return result  # 返回结果


# 0.3 滤波过程（频谱相乘）（参数：img：输入图像频谱；filter：滤波器频谱）
def frequency_spectrum_multiply(img, filter):
    result = np.zeros(img.shape)   # 用于存放结果
    # 复数相乘规则(a+bi)(c+di)=(ac-bd)+(bc+ad)i
    result[:, :, 0] = (img[:, :, 0] * filter[:, :, 0]) - (img[:, :, 1] * filter[:, :, 1])   # 计算实部
    result[:, :, 1] = (img[:, :, 1] * filter[:, :, 0]) + (img[:, :, 0] * filter[:, :, 1])   # 计算虚部
    return result  # 返回结果


# 0.4 还原滤波结果函数接口（参数：img：输入图像）
def restore_filtering_results(img):
    img = np.fft.ifftshift(img)   # 将频谱原点移回左上角
    img_back = cv2.idft(img)   # 傅里叶反变换
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])   # 反变换后每个像素点的值为虚数，取幅值
    img_back = (img_back / img_back.max()) * 255   # 归一化
    return img_back  # 返回结果


# 1 读取图像并预处理
# 1.1 依次读入图像并转化为灰度图
img_gaussian_noise_gray = cv2.imread('./source_images/gaussian_noise.png', 0)
img_origin_gray = cv2.imread('./source_images/origin_image.png', 0)
img_pepper_noise_gray = cv2.imread('./source_images/pepper_noise.png', 0)

# 1.2 将灰度图转化为float_32格式
img_gaussian_noise_gray_float32 = np.float32(img_gaussian_noise_gray)
img_origin_gray_float32 = np.float32(img_origin_gray)
img_pepper_noise_gray_float32 = np.float32(img_pepper_noise_gray)


# 2 对三张原图做傅里叶变换
# 2.1 调用上述函数接口（0.1）计算幅值谱
img_gaussian_noise_gray_magnitude_spectrum = generate_magnitude_spectrum(img_gaussian_noise_gray_float32)
img_origin_gray_magnitude_spectrum = generate_magnitude_spectrum(img_origin_gray_float32)
img_pepper_noise_gray_magnitude_spectrum = generate_magnitude_spectrum(img_pepper_noise_gray_float32)

# 2.2 调用上述函数接口（0.2）进行幅值谱可视化
img_gaussian_noise_gray_magnitude_spectrum_for_visualization = magnitude_spectrum_visualization(
    img_gaussian_noise_gray_magnitude_spectrum)
img_origin_gray_magnitude_spectrum_for_visualization = magnitude_spectrum_visualization(
    img_origin_gray_magnitude_spectrum)
img_pepper_noise_gray_magnitude_spectrum_for_visualization = magnitude_spectrum_visualization(
    img_pepper_noise_gray_magnitude_spectrum)

# 2.3 储存可视化的幅值谱
# 定义储存路径
path = './results/frequency_domain_filtering/origin_images_magnitude_spectrum/'
if not os.path.exists(path):  # 创建储存路径
    os.makedirs(path)
# 储存图像
cv2.imwrite(path + 'gaussion_noise_magnitude_spectrum.png',
            img_gaussian_noise_gray_magnitude_spectrum_for_visualization)
cv2.imwrite(path + 'origin_magnitude_spectrum.png', img_origin_gray_magnitude_spectrum_for_visualization)
cv2.imwrite(path + 'pepper_noise_magnitude_spectrum.png', img_pepper_noise_gray_magnitude_spectrum_for_visualization)


# 3 构建滤波器
# 3.1 构建空域滤波器（float32格式）
# 获取滤波器尺寸（=原图）
H = img_gaussian_noise_gray.shape[0]
W = img_gaussian_noise_gray.shape[1]
# 构建空域滤波器（3*3位于H*W的左上角，其余位置补零）
# sobel_x滤波器
sobel_x_filter = np.zeros([H, W], dtype='f')
sobel_x_filter[0:3, 0:3] = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
# sobel_y滤波器
sobel_y_filter = np.zeros([H, W], dtype='f')
sobel_y_filter[0:3, 0:3] = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
# 高斯滤波器
gaussion_filter = np.zeros([H, W], dtype='f')
gaussion_filter[0:3, 0:3] = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
# 拉普拉斯滤波器
laplacian_filter = np.zeros([H, W], dtype='f')
laplacian_filter[0:3, 0:3] = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]

# 3.2 调用上述函数接口（0.1）计算滤波器的幅值谱
sobel_x_filter_magnitude_spectrum = generate_magnitude_spectrum(sobel_x_filter)  # sobel_x滤波器幅值谱
sobel_y_filter_magnitude_spectrum = generate_magnitude_spectrum(sobel_y_filter)  # sobel_y滤波器幅值谱
gaussion_filter_magnitude_spectrum = generate_magnitude_spectrum(gaussion_filter)  # 高斯滤波器幅值谱
laplacian_filter_magnitude_spectrum = generate_magnitude_spectrum(laplacian_filter)  # 拉普拉斯滤波器幅值谱

# 3.3 调用上述函数接口（0.2）进行滤波器幅值谱可视化
sobel_x_filter_magnitude_spectrum_for_visualization = magnitude_spectrum_visualization(
    sobel_x_filter_magnitude_spectrum)  # sobel_x滤波器可视化幅值谱
sobel_y_filter_magnitude_spectrum_for_visualization = magnitude_spectrum_visualization(
    sobel_y_filter_magnitude_spectrum)  # sobel_y滤波器可视化幅值谱
gaussion_filter_magnitude_spectrum_for_visualization = magnitude_spectrum_visualization(
    gaussion_filter_magnitude_spectrum)  # 高斯滤波器可视化幅值谱
laplacian_filter_magnitude_spectrum_for_visualization = magnitude_spectrum_visualization(
    laplacian_filter_magnitude_spectrum)  # 拉普拉斯滤波器可视化幅值谱

# 3.4 储存可视化的滤波器幅值谱
# 定义储存路径
path = './results/frequency_domain_filtering/filters_magnitude_spectrum/'
if not os.path.exists(path):  # 创建储存路径
    os.makedirs(path)
# 储存图像
cv2.imwrite(path + 'sobel_x_filter_magnitude_spectrum.png', sobel_x_filter_magnitude_spectrum_for_visualization)
cv2.imwrite(path + 'sobel_y_filter_magnitude_spectrum.png', sobel_y_filter_magnitude_spectrum_for_visualization)
cv2.imwrite(path + 'gaussion_filter_magnitude_spectrum.png', gaussion_filter_magnitude_spectrum_for_visualization)
cv2.imwrite(path + 'laplacian_filter_magnitude_spectrum.png', laplacian_filter_magnitude_spectrum_for_visualization)


# 4 滤波操作
# 4.1 调用上述函数接口（0.3）进行滤波操作（返回结果为频谱）
# 对原图用sobel_x滤波器滤波
origin_sobel_x = frequency_spectrum_multiply(img_origin_gray_magnitude_spectrum, sobel_x_filter_magnitude_spectrum)
# 对原图用sobel_y滤波器滤波
origin_sobel_y = frequency_spectrum_multiply(img_origin_gray_magnitude_spectrum, sobel_y_filter_magnitude_spectrum)
# 对原图用高斯滤波器滤波
origin_gaussion = frequency_spectrum_multiply(img_origin_gray_magnitude_spectrum, gaussion_filter_magnitude_spectrum)
# 对原图用拉普拉斯滤波器滤波
origin_laplacian = frequency_spectrum_multiply(img_origin_gray_magnitude_spectrum, laplacian_filter_magnitude_spectrum)
# 对高斯噪声图像用sobel_x滤波器滤波
gaussion_sobel_x = frequency_spectrum_multiply(img_gaussian_noise_gray_magnitude_spectrum, sobel_x_filter_magnitude_spectrum)
# 对高斯噪声图像用sobel_y滤波器滤波
gaussion_sobel_y = frequency_spectrum_multiply(img_gaussian_noise_gray_magnitude_spectrum, sobel_y_filter_magnitude_spectrum)
# 对高斯噪声图像用高斯滤波器滤波
gaussion_gaussion = frequency_spectrum_multiply(img_gaussian_noise_gray_magnitude_spectrum, gaussion_filter_magnitude_spectrum)
# 对高斯噪声图像用拉普拉斯滤波器滤波
gaussion_laplacian = frequency_spectrum_multiply(img_gaussian_noise_gray_magnitude_spectrum, laplacian_filter_magnitude_spectrum)

# 4.2 调用上述函数接口（0.2）将滤波结果频谱可视化
origin_sobel_x_for_visualization = magnitude_spectrum_visualization(origin_sobel_x)
origin_sobel_y_for_visualization = magnitude_spectrum_visualization(origin_sobel_y)
origin_gaussion_for_visualization = magnitude_spectrum_visualization(origin_gaussion)
origin_laplacian_for_visualization = magnitude_spectrum_visualization(origin_laplacian)
gaussion_sobel_x_for_visualization = magnitude_spectrum_visualization(gaussion_sobel_x)
gaussion_sobel_y_for_visualization = magnitude_spectrum_visualization(gaussion_sobel_y)
gaussion_gaussion_for_visualization = magnitude_spectrum_visualization(gaussion_gaussion)
gaussion_laplacian_for_visualization = magnitude_spectrum_visualization(gaussion_laplacian)

# 4.3 储存可视化的滤波结果频谱
# 定义储存路径
path = './results/frequency_domain_filtering/after_filtering/magnitude_spectrums/'
if not os.path.exists(path):  # 创建储存路径
    os.makedirs(path)
# 储存图像
cv2.imwrite(path + 'origin_sobel_x_magnitude_spectrum.png', origin_sobel_x_for_visualization)
cv2.imwrite(path + 'origin_sobel_y_magnitude_spectrum.png', origin_sobel_y_for_visualization)
cv2.imwrite(path + 'origin_gaussion_magnitude_spectrum.png', origin_gaussion_for_visualization)
cv2.imwrite(path + 'origin_laplacian_magnitude_spectrum.png', origin_laplacian_for_visualization)
cv2.imwrite(path + 'gaussion_sobel_x_magnitude_spectrum.png', gaussion_sobel_x_for_visualization)
cv2.imwrite(path + 'gaussion_sobel_y_magnitude_spectrum.png', gaussion_sobel_y_for_visualization)
cv2.imwrite(path + 'gaussion_gaussion_magnitude_spectrum.png', gaussion_gaussion_for_visualization)
cv2.imwrite(path + 'gaussion_laplacian_magnitude_spectrum.png', gaussion_laplacian_for_visualization)


# 5 利用傅里叶反变换（IDFT）将滤波结果还原到空域
# 5.1 调用上述函数接口（0.4）还原滤波结果
origin_sobel_x_result = restore_filtering_results(origin_sobel_x)
origin_sobel_y_result = restore_filtering_results(origin_sobel_y)
origin_gaussion_result = restore_filtering_results(origin_gaussion)
origin_laplacian_result = restore_filtering_results(origin_laplacian)
gaussion_sobel_x_result = restore_filtering_results(gaussion_sobel_x)
gaussion_sobel_y_result = restore_filtering_results(gaussion_sobel_y)
gaussion_gaussion_result = restore_filtering_results(gaussion_gaussion)
gaussion_laplacian_result = restore_filtering_results(gaussion_laplacian)

# 5.2 储存已还原的滤波结果（空域图像）
# 定义储存路径
path = './results/frequency_domain_filtering/after_filtering/filtering_results/'
if not os.path.exists(path):  # 创建储存路径
    os.makedirs(path)
# 储存图像
cv2.imwrite(path + 'origin_sobel_x_filtering_result.png', origin_sobel_x_result)
cv2.imwrite(path + 'origin_sobel_y_filtering_result.png', origin_sobel_y_result)
cv2.imwrite(path + 'origin_gaussion_filtering_result.png', origin_gaussion_result)
cv2.imwrite(path + 'origin_laplacian_filtering_result.png', origin_laplacian_result)
cv2.imwrite(path + 'gaussion_sobel_x_filtering_result.png', gaussion_sobel_x_result)
cv2.imwrite(path + 'gaussion_sobel_y_filtering_result.png', gaussion_sobel_y_result)
cv2.imwrite(path + 'gaussion_gaussion_filtering_result.png', gaussion_gaussion_result)
cv2.imwrite(path + 'gaussion_laplacian_filtering_result.png', gaussion_laplacian_result)

