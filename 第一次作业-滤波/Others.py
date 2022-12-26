# @Author  : 贾林轩 1853688
# @IDE: PyCharm
# @python：3.9.7
# @function：调用opencv现成接口滤波的程序（包括3x3均值、中值、高斯滤波及7x7均值、高斯滤波的操作）

import cv2
import os
import matplotlib.pyplot as plt

# 依次读入图像并转化为灰度图
img_gaussian_noise_gray = cv2.imread('./source_images/gaussian_noise.png', 0)
img_origin_gray = cv2.imread('./source_images/origin_image.png', 0)
img_pepper_noise_gray = cv2.imread('./source_images/pepper_noise.png', 0)

# 3*3均值滤波
# 定义储存路径
path = './results/3x3 mean/'
if not os.path.exists(path):  # 创建储存路径
    os.makedirs(path)
# 调用均值滤波函数接口处理图像（以复制边缘像素的方式padding）
mean_gaussion_noise = cv2.blur(img_gaussian_noise_gray, (3, 3), borderType=cv2.BORDER_REPLICATE)
mean_origin = cv2.blur(img_origin_gray, (3, 3), borderType=cv2.BORDER_REPLICATE)
mean_pepper_noise = cv2.blur(img_pepper_noise_gray, (3, 3), borderType=cv2.BORDER_REPLICATE)
# 储存图像
cv2.imwrite(path + '3x3_mean_gaussion.png', mean_gaussion_noise)
cv2.imwrite(path + '3x3_mean_origin.png', mean_origin)
cv2.imwrite(path + '3x3_mean_pepper.png', mean_pepper_noise)
# 展示图像
plt.figure('3*3 mean')
plt.subplot(311), plt.title("3x3_mean_gaussion_noise"), plt.imshow(mean_gaussion_noise, cmap="gray")
plt.subplot(312), plt.title("3x3_mean_origin"), plt.imshow(mean_origin, cmap="gray")
plt.subplot(313), plt.title("3x3_mean_pepper_noise"), plt.imshow(mean_pepper_noise, cmap="gray")
plt.show()

# 3*3高斯滤波
# 定义储存路径
path = './results/3x3 gaussion/'
if not os.path.exists(path):  # 创建储存路径
    os.makedirs(path)
# 调用均值滤波函数接口处理图像（以复制边缘像素的方式padding）
gaussion_gaussion_noise = cv2.blur(img_gaussian_noise_gray, (3, 3), borderType=cv2.BORDER_REPLICATE)
gaussion_origin = cv2.blur(img_origin_gray, (3, 3), borderType=cv2.BORDER_REPLICATE)
gaussion_pepper_noise = cv2.blur(img_pepper_noise_gray, (3, 3), borderType=cv2.BORDER_REPLICATE)
# 储存图像
cv2.imwrite(path + '3x3_gaussion_gaussion.png', gaussion_gaussion_noise)
cv2.imwrite(path + '3x3_gaussion_origin.png', gaussion_origin)
cv2.imwrite(path + '3x3_gaussion_pepper.png', gaussion_pepper_noise)
# 展示图像
plt.figure('3*3 gaussion')
plt.subplot(311), plt.title("3x3_gaussion_gaussion_noise"), plt.imshow(gaussion_gaussion_noise, cmap="gray")
plt.subplot(312), plt.title("3x3_gaussion_origin"), plt.imshow(gaussion_origin, cmap="gray")
plt.subplot(313), plt.title("3x3_gaussion_pepper_noise"), plt.imshow(gaussion_pepper_noise, cmap="gray")
plt.show()

# 3*3 中值滤波
# 定义储存路径
path = './results/3x3 median/'
if not os.path.exists(path):  # 创建储存路径
    os.makedirs(path)
# 调用中值滤波函数接口处理图像（以复制边缘像素的方式padding）
median_gaussion_noise = cv2.medianBlur(img_gaussian_noise_gray, 3)
median_origin = cv2.medianBlur(img_origin_gray, 3)
median_pepper_noise = cv2.medianBlur(img_pepper_noise_gray, 3)
# 储存图像
cv2.imwrite(path + '3x3_median_gaussion.png', median_gaussion_noise)
cv2.imwrite(path + '3x3_median_origin.png', median_origin)
cv2.imwrite(path + '3x3_median_pepper.png', median_pepper_noise)
# 展示图像
plt.figure('3*3 median')
plt.subplot(311), plt.title("median_gaussion_noise"), plt.imshow(median_gaussion_noise, cmap="gray")
plt.subplot(312), plt.title("median_origin"), plt.imshow(median_origin, cmap="gray")
plt.subplot(313), plt.title("median_pepper_noise"), plt.imshow(median_pepper_noise, cmap="gray")
plt.show()

# 7*7高斯滤波
# 定义储存路径
path = './results/7x7 gaussion/'
if not os.path.exists(path):  # 创建储存路径
    os.makedirs(path)
# 调用均值滤波函数接口处理图像（以复制边缘像素的方式padding）
gaussion_gaussion_noise = cv2.blur(img_gaussian_noise_gray, (7, 7), 0, borderType=cv2.BORDER_REPLICATE)
gaussion_origin = cv2.blur(img_origin_gray, (7, 7), 0, borderType=cv2.BORDER_REPLICATE)
gaussion_pepper_noise = cv2.blur(img_pepper_noise_gray, (7, 7), 0, borderType=cv2.BORDER_REPLICATE)
# 储存图像
cv2.imwrite(path + '7x7_gaussion_gaussion.png', gaussion_gaussion_noise)
cv2.imwrite(path + '7x7_gaussion_origin.png', gaussion_origin)
cv2.imwrite(path + '7x7_gaussion_pepper.png', gaussion_pepper_noise)
# 展示图像
plt.figure('7*7 gaussion')
plt.subplot(311), plt.title("7x7_gaussion_gaussion_noise"), plt.imshow(gaussion_gaussion_noise, cmap="gray")
plt.subplot(312), plt.title("7x7_gaussion_origin"), plt.imshow(gaussion_origin, cmap="gray")
plt.subplot(313), plt.title("7x7_gaussion_pepper_noise"), plt.imshow(gaussion_pepper_noise, cmap="gray")
plt.show()

# 7*7均值滤波
# 定义储存路径
path = './results/7x7 mean/'
if not os.path.exists(path):  # 创建储存路径
    os.makedirs(path)
# 调用均值滤波函数接口处理图像（以复制边缘像素的方式padding）
mean_gaussion_noise = cv2.blur(img_gaussian_noise_gray, (7, 7), borderType=cv2.BORDER_REPLICATE)
mean_origin = cv2.blur(img_origin_gray, (7, 7), borderType=cv2.BORDER_REPLICATE)
mean_pepper_noise = cv2.blur(img_pepper_noise_gray, (7, 7), borderType=cv2.BORDER_REPLICATE)
# 储存图像
cv2.imwrite(path + '7x7_mean_gaussion.png', mean_gaussion_noise)
cv2.imwrite(path + '7x7_mean_origin.png', mean_origin)
cv2.imwrite(path + '7x7_mean_pepper.png', mean_pepper_noise)
# 展示图像
plt.figure('7*7 mean')
plt.subplot(311), plt.title("7x7_mean_gaussion_noise"), plt.imshow(mean_gaussion_noise, cmap="gray")
plt.subplot(312), plt.title("7x7_mean_origin"), plt.imshow(mean_origin, cmap="gray")
plt.subplot(313), plt.title("7x7_mean_pepper_noise"), plt.imshow(mean_pepper_noise, cmap="gray")
plt.show()
