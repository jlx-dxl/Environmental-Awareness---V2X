# @Author  : 贾林轩 1853688
# @IDE: PyCharm
# @python：3.6.15
# @numpy -V : 1.19.5
# @opencv-python -V : 3.4.2.16
# @opencv-contrib-python -V : 3.4.2.16


import numpy as np
import cv2
import os

# 读入图片
A = cv2.imread('./source-images/A.png')
B = cv2.imread('./source-images/B.png')
# 因两图亮度不同，故通过线性加权修正两图亮度，权值为k，考虑到B图中黑色区域较多，故两图均除去黑色部分再做平均
A_mean = np.mean(np.where(A[:, :, :] != 0))
B_mean = np.mean(np.where(B[:, :, :] != 0))
k = A_mean / B_mean
print(k)


# SIFT
# 定义储存路径
path = './results/SIFT/'
if not os.path.exists(path):  # 创建储存路径
    os.makedirs(path)
# 创建SIFT类的实例
sift = cv2.xfeatures2d.SIFT_create()
# 计算特征点和描述子
keypoints_1, descriptors_1 = sift.detectAndCompute(A, None)
keypoints_2, descriptors_2 = sift.detectAndCompute(B, None)
# 特征点匹配（不加筛选，显示所有匹配）
matcher = cv2.BFMatcher()   # 创建匹配器
matches = matcher.match(descriptors_1, descriptors_2)   # 进行特征匹配
matches = sorted(matches, key=lambda x: x.distance)   # 将特征匹配按欧氏距离排序
print(len(matches))
SIFT_match = cv2.drawMatches(A, keypoints_1, B, keypoints_2, matches[:len(matches)], B, flags=2)   # 将匹配的特征点进行连线
cv2.imwrite(path + 'SIFT_match.png',SIFT_match)
# RANSAC滤去离群点
# 获取相匹配特征点的坐标
ptsA = np.float32([keypoints_1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
ptsB = np.float32([keypoints_2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
# 采用RANSAC方法拟合并计算Homography矩阵（mask为二进制掩码）
(H, mask) = cv2.findHomography(ptsB, ptsA, cv2.RANSAC, ransacReprojThreshold=5.0)
print(np.float16(H))   # 打印Homography矩阵
good = []   # good用于保存经过RANSAC处理得到的良好匹配点
for (m, s) in zip(matches, mask):
    # 当二进制掩码为1时（即RANSAC中的internal点）将其添加到良好匹配点中
    if s == 1:
        good.append(m)
print(len(good))
SIFT_ransac = cv2.drawMatches(A, keypoints_1, B, keypoints_2, good[:len(good)], B, flags=2)   # 将匹配的特征点进行连线
cv2.imwrite(path + 'SIFT_ransac.png',SIFT_ransac)
# 图像拼接
wrap = cv2.warpPerspective(B, H, (A.shape[1] + B.shape[1], A.shape[0] + B.shape[0]))   # 根据Homography矩阵进行单应变换（适当扩大画布）
wrap = k * wrap   # 通过线性加权调整两图亮度差异
wrap[0:A.shape[0], 0:A.shape[1]] = A   # 左图不变，放入左端
# 得到有效图像区域
rows, cols = np.where(wrap[:, :, 0] != 0)
min_row, max_row = min(rows), max(rows) + 1
min_col, max_col = min(cols), max(cols) + 1
# 去除黑色无用部分
result = wrap[min_row:max_row, min_col:max_col, :]
cv2.imwrite(path + 'SIFT_stitch.png',result)


# SURF
# 定义储存路径
path = './results/SURF/'
if not os.path.exists(path):  # 创建储存路径
    os.makedirs(path)
# 创建surf类的实例，检测阈值为7000（阈值越大检测到的特征点数越少）
surf = cv2.xfeatures2d.SURF_create(hessianThreshold=7000)
# 计算特征点和描述子
keypoints_1, descriptors_1 = surf.detectAndCompute(A, None)
keypoints_2, descriptors_2 = surf.detectAndCompute(B, None)
# 特征点匹配
matcher = cv2.BFMatcher()   # 创建匹配器
matches = matcher.match(descriptors_1, descriptors_2)   # 进行特征匹配
matches = sorted(matches, key=lambda x: x.distance)   # 将特征匹配按欧氏距离排序
print(len(matches))
SIFT_match = cv2.drawMatches(A, keypoints_1, B, keypoints_2, matches[:len(matches)], B, flags=2)   # 将匹配的特征点进行连线
cv2.imwrite(path + 'SURF_match.png',SIFT_match)
# RANSAC滤去离群点
# 获得相匹配特征点的坐标
ptsA = np.float32([keypoints_1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
ptsB = np.float32([keypoints_2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
# 采用RANSAC方法拟合并计算Homography矩阵（mask为二进制掩码）
(H, mask) = cv2.findHomography(ptsB, ptsA, cv2.RANSAC, ransacReprojThreshold=5.0)
print(np.float16(H))   # 打印Homography矩阵
good = []   # good用于保存经过RANSAC处理得到的良好匹配点
for (m, s) in zip(matches, mask):
    # 当二进制掩码为1时（即RANSAC中的internal点）将其添加到良好匹配点中
    if s == 1:
        good.append(m)
print(len(good))
SURF_ransac = cv2.drawMatches(A, keypoints_1, B, keypoints_2, good[:len(good)], B, flags=2)   # 将匹配的特征点进行连线
cv2.imwrite(path + 'SURF_ransac.png',SURF_ransac)
# 图像拼接
wrap = cv2.warpPerspective(B, H, (A.shape[1] + B.shape[1], A.shape[0] + B.shape[0]))   # 根据Homography矩阵进行单应变换（适当扩大画布）
wrap = k * wrap   # 通过线性加权调整两图亮度差异
wrap[0:A.shape[0], 0:A.shape[1]] = A   # 左图不变，放入左端
# 得到有效图像区域
rows, cols = np.where(wrap[:, :, 0] != 0)
min_row, max_row = min(rows), max(rows) + 1
min_col, max_col = min(cols), max(cols) + 1
# 去除黑色无用部分
result = wrap[min_row:max_row, min_col:max_col, :]
cv2.imwrite(path + 'SURF_stitch.png',result)


# ORB
# 定义储存路径
path = './results/ORB/'
if not os.path.exists(path):  # 创建储存路径
    os.makedirs(path)
# 创建ORB类的实例
orb = cv2.ORB_create()
# 计算特征点和描述子
keypoints_1, descriptors_1 = orb.detectAndCompute(A, None)
keypoints_2, descriptors_2 = orb.detectAndCompute(B, None)
# 特征点匹配（不加筛选，显示所有匹配）
matcher = cv2.BFMatcher()   # 创建匹配器
matches = matcher.match(descriptors_1, descriptors_2)   # 进行特征匹配
print(len(matches))
matches = sorted(matches, key=lambda x: x.distance)   # 将特征匹配按欧氏距离排序
ORB_match = cv2.drawMatches(A, keypoints_1, B, keypoints_2, matches[:len(matches)], B, flags=2)   # 将匹配的特征点进行连线
cv2.imwrite(path + 'ORB_match.png',ORB_match)
# RANSAC滤去离群点
# 获得相匹配特征点的坐标
ptsA = np.float32([keypoints_1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
ptsB = np.float32([keypoints_2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
# 采用RANSAC方法拟合并计算Homography矩阵（mask为二进制掩码）
(H, mask) = cv2.findHomography(ptsB, ptsA, cv2.RANSAC, ransacReprojThreshold=15.0)
print(np.float16(H))   # 打印Homography矩阵
good = []   # good用于保存经过RANSAC处理得到的良好匹配点
for (m, s) in zip(matches, mask):
    # 当二进制掩码为1时（即RANSAC中的internal点）将其添加到良好匹配点中
    if s == 1:
        good.append(m)
print(len(good))
ORB_ransac = cv2.drawMatches(A, keypoints_1, B, keypoints_2, good[:len(good)], B, flags=2)   # 将匹配的特征点进行连线
cv2.imwrite(path + 'ORB_ransac.png',ORB_ransac)
# 图像拼接
wrap = cv2.warpPerspective(B, H, (A.shape[1] + B.shape[1], A.shape[0] + B.shape[0]))   # 根据Homography矩阵进行单应变换（适当扩大画布）
wrap = k * wrap   # 通过线性加权调整两图亮度差异
wrap[0:A.shape[0], 0:A.shape[1]] = A   # 左图不变，放入左端
# 得到有效图像区域
rows, cols = np.where(wrap[:, :, 0] != 0)
min_row, max_row = min(rows), max(rows) + 1
min_col, max_col = min(cols), max(cols) + 1
# 去除黑色无用部分
result = wrap[min_row:max_row, min_col:max_col, :]
cv2.imwrite(path + 'ORB_stitch.png',result)