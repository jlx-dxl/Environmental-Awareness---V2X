# @Author  : 贾林轩 1853688
# @IDE: PyCharm
# @python：3.9.7
# @catalogue：
#   0 定义函数接口
#      0.1 腐蚀操作函数接口
#      0.2 膨胀操作函数接口
#   1 依次读入图像并转化为灰度图
#   2 腐蚀操作
#   3 膨胀操作
#   4 顶帽运算
#   5 黑帽运算
#   6 展示结果


import copy
import os
import cv2
import matplotlib.pyplot as plt


# 0 定义函数接口
# 0.1 腐蚀操作函数接口（参数：img：输入图片；kernel_size：窗口大小）
def Erode(img, kernel_size):
    new_image = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REPLICATE)  # 按复制边缘像素的方式padding
    result = copy.copy(img)  # result用于储存腐蚀结果
    # 遍历整个图片进行腐蚀操作
    for i in range(0, result.shape[0]):
        for j in range(0, result.shape[1]):
            window = new_image[i:i + kernel_size, j:j + kernel_size]  # 3*3窗口
            result[i, j] = window.min()  # 取窗口内最小值作为对应位置像素值
    return result  # 返回结果


# 0.2 膨胀操作函数接口（参数：img：输入图片；kernel_size：窗口大小）
def Dilate(img, kernel_size):
    new_image = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REPLICATE)  # 按复制边缘像素的方式padding
    result = copy.copy(img)  # result用于储存膨胀结果
    # 遍历整个图片进行膨胀操作
    for i in range(0, result.shape[0]):
        for j in range(0, result.shape[1]):
            window = new_image[i:i + kernel_size, j:j + kernel_size]  # 3*3窗口
            result[i, j] = window.max()  # 取窗口内最大值作为对应位置像素值
    return result  # 返回结果


# 1 依次读入图像并转化为灰度图
img_gaussian_noise_gray = cv2.imread('./source_images/gaussian_noise.png', 0)
img_origin_gray = cv2.imread('./source_images/origin_image.png', 0)
img_pepper_noise_gray = cv2.imread('./source_images/pepper_noise.png', 0)
print('Program start!')

# 2 腐蚀操作
# 定义储存路径
path = './results/morphological_filtering/erosion/'
if not os.path.exists(path):  # 创建储存路径
    os.makedirs(path)
# 调用上述腐蚀函数接口（0.1）进行腐蚀操作
img_gaussian_noise_gray_eroded = Erode(img_gaussian_noise_gray, 3)
img_origin_gray_eroded = Erode(img_origin_gray, 3)
img_pepper_noise_gray_eroded = Erode(img_pepper_noise_gray, 3)
print('Images successfully eroded!')
# 储存图像
cv2.imwrite(path + 'gaussion_noise_eroded.png', img_gaussian_noise_gray_eroded)
cv2.imwrite(path + 'origin_eroded.png', img_origin_gray_eroded)
cv2.imwrite(path + 'pepper_noise_eroded.png', img_pepper_noise_gray_eroded)

# 3 膨胀操作
# 定义储存路径
path = './results/morphological_filtering/dilation/'
if not os.path.exists(path):  # 创建储存路径
    os.makedirs(path)
# 调用上述膨胀函数接口（0.2）进行膨胀操作
img_gaussian_noise_gray_dilated = Dilate(img_gaussian_noise_gray, 3)
img_origin_gray_dilated = Dilate(img_origin_gray, 3)
img_pepper_noise_gray_dilated = Dilate(img_pepper_noise_gray, 3)
print('Images successfully dilated!')
# 储存图像
cv2.imwrite(path + 'gaussion_noise_dilated.png', img_gaussian_noise_gray_dilated)
cv2.imwrite(path + 'origin_dilated.png', img_origin_gray_dilated)
cv2.imwrite(path + 'pepper_noise_dilated.png', img_pepper_noise_gray_dilated)

# 4 顶帽运算
# 定义储存路径
path = './results/morphological_filtering/top_hat/'
if not os.path.exists(path):  # 创建储存路径
    os.makedirs(path)
# 首先对三张图片进行开运算，即对三张图片的腐蚀操作结果进行膨胀操作
img_gaussian_noise_gray_opening = Dilate(img_gaussian_noise_gray_eroded, 3)
img_origin_gray_opening = Dilate(img_origin_gray_eroded, 3)
img_pepper_noise_gray_opening = Dilate(img_pepper_noise_gray_eroded, 3)
# 顶帽运算：原图 - 开运算结果
img_gaussian_noise_gray_th = img_gaussian_noise_gray - img_gaussian_noise_gray_opening
img_origin_gray_th = img_origin_gray - img_origin_gray_opening
img_pepper_noise_gray_th = img_pepper_noise_gray - img_pepper_noise_gray_opening
print('Images successfully top_hatted!')
# 储存图像
cv2.imwrite(path + 'gaussion_noise_top_hat.png', img_gaussian_noise_gray_th)
cv2.imwrite(path + 'origin_top_hat.png', img_origin_gray_th)
cv2.imwrite(path + 'pepper_noise_top_hat.png', img_pepper_noise_gray_th)

# 5 黑帽运算
# 定义储存路径
path = './results/morphological_filtering/black_hat/'
if not os.path.exists(path):  # 创建储存路径
    os.makedirs(path)
# 首先对三张图片进行闭运算，即对三张图片的膨胀操作结果进行腐蚀操作
img_gaussian_noise_gray_closing = Erode(img_gaussian_noise_gray_dilated, 3)
img_origin_gray_closing = Erode(img_origin_gray_dilated, 3)
img_pepper_noise_gray_closing = Erode(img_pepper_noise_gray_dilated, 3)
# 黑帽运算：闭运算结果 - 原图
img_gaussian_noise_gray_bh = img_gaussian_noise_gray_closing - img_gaussian_noise_gray
img_origin_gray_bh = img_origin_gray_closing - img_origin_gray
img_pepper_noise_gray_bh = img_pepper_noise_gray_closing - img_pepper_noise_gray
print('Images successfully black_hatted!')
# 储存图像
cv2.imwrite(path + 'gaussion_noise_black_hat.png', img_gaussian_noise_gray_bh)
cv2.imwrite(path + 'origin_black_hat.png', img_origin_gray_bh)
cv2.imwrite(path + 'pepper_noise_black_hat.png', img_pepper_noise_gray_bh)

# 6 展示结果
plt.figure('morphological filtering', frameon=False)
plt.subplot(5, 3, 1), plt.title("gaussion_noise_origin"), plt.imshow(img_gaussian_noise_gray, cmap="gray"), plt.axis(
    'off')
plt.subplot(5, 3, 2), plt.title("origin_origin"), plt.imshow(img_origin_gray, cmap="gray"), plt.axis('off')
plt.subplot(5, 3, 3), plt.title("pepper_noise_origin"), plt.imshow(img_pepper_noise_gray, cmap="gray"), plt.axis('off')
plt.subplot(5, 3, 4), plt.title("gaussion_noise after erodsion"), plt.imshow(img_gaussian_noise_gray_eroded,
                                                                             cmap="gray"), plt.axis('off')
plt.subplot(5, 3, 5), plt.title("origin after erodsion"), plt.imshow(img_origin_gray_eroded, cmap="gray"), plt.axis(
    'off')
plt.subplot(5, 3, 6), plt.title("pepper_noise after erodsion"), plt.imshow(img_pepper_noise_gray_eroded,
                                                                           cmap="gray"), plt.axis('off')
plt.subplot(5, 3, 7), plt.title("gaussion_noise after dilation"), plt.imshow(img_gaussian_noise_gray_dilated,
                                                                             cmap="gray"), plt.axis('off')
plt.subplot(5, 3, 8), plt.title("origin after dilation"), plt.imshow(img_origin_gray_dilated, cmap="gray"), plt.axis(
    'off')
plt.subplot(5, 3, 9), plt.title("pepper_noise after dilation"), plt.imshow(img_pepper_noise_gray_dilated,
                                                                           cmap="gray"), plt.axis('off')
plt.subplot(5, 3, 10), plt.title("gaussion_noise after top_hat"), plt.imshow(img_gaussian_noise_gray_th,
                                                                             cmap="gray"), plt.axis('off')
plt.subplot(5, 3, 11), plt.title("origin after top_hat"), plt.imshow(img_gaussian_noise_gray_th, cmap="gray"), plt.axis(
    'off')
plt.subplot(5, 3, 12), plt.title("pepper_noise after top_hat"), plt.imshow(img_gaussian_noise_gray_th,
                                                                           cmap="gray"), plt.axis('off')
plt.subplot(5, 3, 13), plt.title("gaussion_noise after black_hat"), plt.imshow(img_gaussian_noise_gray_bh,
                                                                               cmap="gray"), plt.axis('off')
plt.subplot(5, 3, 14), plt.title("origin after black_hat"), plt.imshow(img_gaussian_noise_gray_bh,
                                                                       cmap="gray"), plt.axis('off')
plt.subplot(5, 3, 15), plt.title("pepper_noise after black_hat"), plt.imshow(img_gaussian_noise_gray_bh,
                                                                             cmap="gray"), plt.axis('off')
plt.show()

print('The results can be found in the folder "./results/morphological_filtering/"')
