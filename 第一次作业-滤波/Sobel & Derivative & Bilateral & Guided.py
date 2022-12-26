# @Author  : 贾林轩 1853688
# @IDE: PyCharm
# @python：3.9.7
# function：调用自己写的滤波函数接口进行滤波操作的程序（包括3x3sobel滤波、3x3导数滤波、7x7双边滤波及7x7引导滤波）

import cv2
import os
import Filters
import matplotlib.pyplot as plt
import time


# 依次读入图像并转化为灰度图
img_gaussian_noise_gray = cv2.imread('./source_images/gaussian_noise.png', 0)
img_origin_gray = cv2.imread('./source_images/origin_image.png', 0)
img_pepper_noise_gray = cv2.imread('./source_images/pepper_noise.png', 0)

# 3*3sobel滤波
# 开始计时
start = time.time()
print('Sobel 开始计时')
# 定义储存路径
path='./results/3x3 sobel/'
if not os.path.exists(path):   # 创建储存路径
    os.makedirs(path)
# 调用写好的sobel滤波函数接口进行滤波操作（horizontal表示滤出横向边界，vertical表示滤出纵向边界）
sobel_gaussion_horizontal = Filters.Sobel(img_gaussian_noise_gray,"sobel_horizontal")
sobel_gaussion_vertical = Filters.Sobel(img_gaussian_noise_gray,"sobel_vertical")
sobel_origin_horizontal = Filters.Sobel(img_origin_gray,"sobel_horizontal")
sobel_origin_vertical = Filters.Sobel(img_origin_gray,"sobel_vertical")
sobel_pepper_horizontal = Filters.Sobel(img_pepper_noise_gray,"sobel_horizontal")
sobel_pepper_vertical = Filters.Sobel(img_pepper_noise_gray,"sobel_vertical")
# 保存图像
cv2.imwrite(path+'3x3_sobel_gaussion_horizontal.png',sobel_gaussion_horizontal)
cv2.imwrite(path+'3x3_sobel_gaussion_vertical.png',sobel_gaussion_vertical)
cv2.imwrite(path+'3x3_sobel_origin_horizontal.png',sobel_origin_horizontal)
cv2.imwrite(path+'3x3_sobel_origin_vertical.png',sobel_origin_vertical)
cv2.imwrite(path+'3x3_sobel_pepper_horizontal.png',sobel_pepper_horizontal)
cv2.imwrite(path+'3x3_sobel_pepper_vertical.png',sobel_pepper_vertical)
# 展示图像
plt.subplot(321), plt.title("sobel_horizontal_gaussian_noise"), plt.imshow(sobel_gaussion_horizontal, cmap="gray")
plt.subplot(322), plt.title("sobel_vertical_gaussian_noise"), plt.imshow(sobel_gaussion_vertical, cmap="gray")
plt.subplot(323), plt.title("sobel_horizontal_origin"), plt.imshow(sobel_origin_horizontal, cmap="gray")
plt.subplot(324), plt.title("sobel_vertical_origin"), plt.imshow(sobel_origin_vertical, cmap="gray")
plt.subplot(325), plt.title("sobel_horizontal_pepper_noise"), plt.imshow(sobel_pepper_horizontal, cmap="gray")
plt.subplot(326), plt.title("sobel_vertical_pepper_noise"), plt.imshow(sobel_pepper_vertical, cmap="gray")
end = time.time()   # 计时结束
print('Sobel 执行时间 = {} min {} s'.format(int((end - start) / 60), int((end - start) % 60)))
plt.show()

# 3*3 derivative滤波
# 开始计时
start = time.time()
print('Derivative 开始计时')
# 定义储存路径
path = './results/3x3 derivative/'
if not os.path.exists(path):  # 创建储存路径
    os.makedirs(path)
# 调用写好的导数滤波函数接口进行滤波操作（horizontal表示滤出横向边界，vertical表示滤出纵向边界）
derivative_gaussion_horizontal = Filters.Derivative(img_gaussian_noise_gray, "derivative_horizontal")
derivative_gaussion_vertical = Filters.Derivative(img_gaussian_noise_gray, "derivative_vertical")
derivative_origin_horizontal = Filters.Derivative(img_origin_gray, "derivative_horizontal")
derivative_origin_vertical = Filters.Derivative(img_origin_gray, "derivative_vertical")
derivative_pepper_horizontal = Filters.Derivative(img_pepper_noise_gray, "derivative_horizontal")
derivative_pepper_vertical = Filters.Derivative(img_pepper_noise_gray, "derivative_vertical")
# 保存图像
cv2.imwrite(path + '3x3_derivative_gaussion_horizontal.png', derivative_gaussion_horizontal)
cv2.imwrite(path + '3x3_derivative_gaussion_vertical.png', derivative_gaussion_vertical)
cv2.imwrite(path + '3x3_derivative_origin_horizontal.png', derivative_origin_horizontal)
cv2.imwrite(path + '3x3_derivative_origin_vertical.png', derivative_origin_vertical)
cv2.imwrite(path + '3x3_derivative_pepper_horizontal.png', derivative_pepper_horizontal)
cv2.imwrite(path + '3x3_derivative_pepper_vertical.png', derivative_pepper_vertical)
# 展示图像
plt.subplot(321), plt.title("derivative_horizontal_gaussian_noise"), plt.imshow(derivative_gaussion_horizontal, cmap="gray")
plt.subplot(322), plt.title("derivative_vertical_gaussian_noise"), plt.imshow(derivative_gaussion_vertical, cmap="gray")
plt.subplot(323), plt.title("derivative_horizontal_origin"), plt.imshow(derivative_origin_horizontal, cmap="gray")
plt.subplot(324), plt.title("derivative_vertical_origin"), plt.imshow(derivative_origin_vertical, cmap="gray")
plt.subplot(325), plt.title("derivative_horizontal_pepper_noise"), plt.imshow(derivative_pepper_horizontal, cmap="gray")
plt.subplot(326), plt.title("derivative_vertical_pepper_noise"), plt.imshow(derivative_pepper_vertical, cmap="gray")
end = time.time()   # 计时结束
print('Derivative 执行时间 = {} min {} s'.format(int((end - start) / 60), int((end - start) % 60)))
plt.show()

# 7*7 双边滤波
# 开始计时
start = time.time()
print('Bilateral 开始计时')
# 定义储存路径
path = './results/7x7 bilateral/'
if not os.path.exists(path):  # 创建储存路径
    os.makedirs(path)
# 调用写好的双边滤波函数接口进行滤波操作
bilateral_gaussion_noise = Filters.Bilateral(img_gaussian_noise_gray, radius=3, sigma_color=10, sigma_space=10)
bilateral_origin = Filters.Bilateral(img_origin_gray, radius=3, sigma_color=10, sigma_space=10)
bilateral_pepper_noise = Filters.Bilateral(img_pepper_noise_gray, radius=3, sigma_color=10, sigma_space=10)
# 保存图像
cv2.imwrite(path + '7x7_bilateral_gaussion.png', bilateral_gaussion_noise)
cv2.imwrite(path + '7x7_bilateral_origin.png', bilateral_origin)
cv2.imwrite(path + '7x7_bilateral_pepper.png', bilateral_pepper_noise)
# 展示图像
plt.subplot(311), plt.title("bilateral_gaussion_noise"), plt.imshow(bilateral_gaussion_noise, cmap="gray")
plt.subplot(312), plt.title("bilateral_origin"), plt.imshow(bilateral_origin, cmap="gray")
plt.subplot(313), plt.title("bilateral_pepper_noise"), plt.imshow(bilateral_pepper_noise, cmap="gray")
end = time.time()   # 计时结束
print('Bilateral 执行时间 = {} min {} s'.format(int((end - start) / 60), int((end - start) % 60)))
plt.show()

# 7*7 导向滤波
# 开始计时
start = time.time()
print('Guided 开始计时')
# 定义储存路径
path = './results/7x7 guided/'
if not os.path.exists(path):  # 创建储存路径
    os.makedirs(path)
# 调用写好的导向滤波函数接口进行滤波操作
guided_gaussion_noise = Filters.Guided(img_gaussian_noise_gray, img_gaussian_noise_gray, 7, 0)
guided_origin = Filters.Guided(img_origin_gray, img_origin_gray, 7, 0)
guided_pepper_noise = Filters.Guided(img_pepper_noise_gray, img_pepper_noise_gray, 7, 0)
# 保存图像
cv2.imwrite(path + '7x7_guided_gaussion.png', guided_gaussion_noise)
cv2.imwrite(path + '7x7_guided_origin.png', guided_origin)
cv2.imwrite(path + '7x7_guided_pepper.png', guided_pepper_noise)
# 展示图像
plt.subplot(311), plt.title("guided_gaussion_noise"), plt.imshow(guided_gaussion_noise, cmap="gray")
plt.subplot(312), plt.title("guided_origin"), plt.imshow(guided_origin, cmap="gray")
plt.subplot(313), plt.title("guided_pepper_noise"), plt.imshow(guided_pepper_noise, cmap="gray")
end = time.time()   # 计时结束
print('Guided 执行时间 = {} min {} s'.format(int((end - start) / 60), int((end - start) % 60)))
plt.show()


