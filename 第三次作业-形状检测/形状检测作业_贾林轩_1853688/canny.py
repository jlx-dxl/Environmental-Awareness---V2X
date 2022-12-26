# @Author  : 贾林轩 1853688
# @IDE: PyCharm
# @python：3.9.7

import cv2
import numpy as np
import os
import math


def canny(img, high_limit, low_limit, savepath, ifsave_origin_magnitude, ifsave_non_maximum_suppression,
          ifsave_double_threshold_operate):
    # 对图像进行高斯模糊
    img_after_gaussion = cv2.GaussianBlur(img, (3, 3), 1, 1)
    # 分别进行x, y方向的sobel滤波
    sobelx = cv2.Sobel(img_after_gaussion, cv2.CV_64F, 1, 0, ksize=3)
    sobely = -cv2.Sobel(img_after_gaussion, cv2.CV_64F, 0, 1, ksize=3)
    # 求出幅值
    mag = cv2.magnitude(sobelx, sobely)  # 画出幅值图
    if ifsave_origin_magnitude is True:
        cv2.imwrite(savepath + 'sobel_magnitude.png', mag)
    # 求出梯度方向（角度值）
    sobelx = np.maximum(sobelx, 1e-10)  # 避免分母为零
    tan = sobely / sobelx  # 求出正切值
    angle = np.arctan2(sobely, sobelx)  # 求出梯度方向角
    # 对角度进行离散化
    # 映射到-pi/2 - pi/2
    angle[np.where(angle > np.pi / 2)] = angle[np.where(angle > np.pi / 2)] - np.pi
    angle[np.where(angle <= -np.pi / 2)] = angle[np.where(angle <= -np.pi / 2)] + np.pi
    angle1 = -angle + np.pi / 2
    # 将theta分为四个区间：
    #    索引   角度区间
    #    3   (-pi/2,-pi/4]
    #    2   (-pi/4,0]
    #    1   (0,pi/4]
    #    0   (pi/4,pi/2]
    for i in range(angle1.shape[0]):
        for j in range(angle1.shape[1]):
            angle1[i, j] = math.floor(angle1[i, j] / np.pi * 4)
    # 非极大值抑制
    # padding会导致扩充图与原图索引不一致，当在原图中像素索引为（i，j）时，该像素在padding图中的索引为（i+1，j+1）
    #        NW        N        NE
    # NW   (i,j)    (i,j+1)    (i,j+2)    NE
    # N    (i+1,j)  (i+1,j+1)  (i+1,j+2)  E
    # SW   (i+2,j)  (i+2,j+1)  (i+2,j+2)  SE
    #        SW        S        SE
    mag_padding = cv2.copyMakeBorder(mag, 1, 1, 1, 1, cv2.BORDER_REPLICATE)  # padding
    Gp1 = np.zeros(mag.shape)  # 正方向梯度值
    Gp2 = np.zeros(mag.shape)  # 负方向梯度值
    for i in range(mag.shape[0]):
        for j in range(mag.shape[1]):
            if angle1[i, j] == 0:
                Gp1[i, j] = (1 / abs(tan[i, j])) * mag_padding[i, j + 2] + (1 - (1 / abs(tan[i, j]))) * mag_padding[
                    i, j + 1]
                Gp2[i, j] = (1 / abs(tan[i, j])) * mag_padding[i + 2, j] + (1 - (1 / abs(tan[i, j]))) * mag_padding[
                    i + 2, j + 1]
            elif angle1[i, j] == 1:
                Gp1[i, j] = (1 - abs(tan[i, j])) * mag_padding[i + 1, j + 2] + abs(tan[i, j]) * mag_padding[i, j + 2]
                Gp2[i, j] = (1 - abs(tan[i, j])) * mag_padding[i + 1, j] + abs(tan[i, j]) * mag_padding[i + 2, j]
            elif angle1[i, j] == 2:
                Gp1[i, j] = abs(tan[i, j]) * mag_padding[i + 2, j + 2] + (1 - abs(tan[i, j])) * mag_padding[i + 1, j + 2]
                Gp2[i, j] = abs(tan[i, j]) * mag_padding[i, j] + (1 - abs(tan[i, j])) * mag_padding[i +1, j]
            elif angle1[i, j] == 3:
                Gp1[i, j] = (1 - (1 / abs(tan[i, j]))) * mag_padding[i + 2, j + 1] + (1 / abs(tan[i, j])) * mag_padding[
                    i + 2, j + 2]
                Gp2[i, j] = (1 - (1 / abs(tan[i, j]))) * mag_padding[i, j + 1] + (1 / abs(tan[i, j])) * mag_padding[
                    i, j]

            if mag[i, j] < Gp1[i, j] or mag[i, j] < Gp2[i, j]:  # 只有Gp在Gp，Gp1，Gp2中最大时才保留
                mag[i, j] = 0
    mag = (mag / mag.max()) * 255  # 归一化
    nms_result = mag
    if ifsave_non_maximum_suppression is True:
        cv2.imwrite(savepath + 'after_non_maximum_suppression.png', mag)
    # 双阈值操作
    mag[np.where(mag < low_limit)] = 0  # 背景
    mag[np.where(mag > high_limit)] = 255  # 强边缘
    # 弱边缘： 8邻域区间，如果8邻域内有像素联通强边缘的像素，则置为强边缘
    nn = np.array(((1., 1., 1.), (1., 0., 1.), (1., 1., 1.)), dtype=np.float32)
    mag_padding = cv2.copyMakeBorder(mag, 1, 1, 1, 1, cv2.BORDER_REPLICATE)  # padding
    for y in range(1, mag_padding.shape[0] - 1):
        for x in range(1, mag_padding.shape[1] - 1):   # 遍历图像
            if mag_padding[y, x] < low_limit or mag_padding[y, x] > high_limit:
                continue   # 如果为强边缘点或背景，则不做任何操作
            if np.max(mag_padding[y - 1:y + 2, x - 1:x + 2] * nn) >= high_limit:   # 如果8邻域中存在强边缘点
                mag[y - 1, x - 1] = 255   # 则置为强边缘点
            else:
                mag[y - 1, x - 1] = 0   # 否则置为背景
    cv2.imwrite(savepath + 'binary_result.png', mag)
    result = (mag / 255) * nms_result
    if ifsave_double_threshold_operate is True:
        cv2.imwrite(savepath + 'after_threshold_operate.png', result)
    return mag, angle

# 定义储存路径
path = './results/canny/'
if not os.path.exists(path):  # 创建储存路径
    os.makedirs(path)

# 读入原图并转化为灰度图像
lanes_gray = cv2.imread('./source_images/lanes.png', 0)

# 对车道线图执行canny检测
canny(lanes_gray, 70, 60, path, ifsave_origin_magnitude=True, ifsave_non_maximum_suppression=True, ifsave_double_threshold_operate=True)
