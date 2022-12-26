# @Author  : 贾林轩 1853688
# @IDE: PyCharm
# @python：3.9.7

import cv2
import numpy as np
import os
import canny

# 定义储存路径
path = './/results/hough//'
if not os.path.exists(path):  # 创建储存路径
    os.makedirs(path)

# 读入原图并转化为灰度图像
wheel_gray = cv2.imread('./source_images/wheel.png', 0)

# 对车轮图进行canny检测
result, angle = canny.canny(wheel_gray, 40, 25, path, ifsave_origin_magnitude=False, ifsave_double_threshold_operate=False,
            ifsave_non_maximum_suppression=False)

# 对车轮图中的强边缘点信息进行汇总
m, n = result.shape   # 图片长宽
x = []   # 强边缘点横坐标
y = []   # 强边缘点纵坐标
thetas = []   # 强边缘点梯度方向
for i in range(m):
    for j in range(n):
        if result[i, j] == 255:   # 遍历所有强边缘点
            x.append(j)
            y.append(i)
            thetas.append(angle[i, j])   # 得到强边缘坐标矩阵序列

# 投票找到圆心
# 定义参数空间
pspace = np.zeros([m, n])
# 栅格化横坐标
v = np.linspace(0, n - 1, n)
for i in range(len(thetas)):   # 遍历所有强边缘点
    vindx = []   #
    vindy = []
    if thetas[i] != np.pi / 2:   # 如果梯度角不等于pi/2
        u = -np.tan(thetas[i]) * (v - x[i]) + y[i]   # 由像素空间到参数空间的直线方程
        u = np.rint(u)   # 取整以适应栅格化的参数空间
        for j in range(len(u)):   # 遍历直线上所有点
            if u[j] >= 0 and u[j] < m:   # 截取在图像大小范围内的部分
                vindy.append(u[j])   # 纵坐标索引
                vindx.append(j)   # 横坐标索引
    else:   # 如果梯度角不等于pi/2，即斜率不存在
        u = np.linspace(0, m - 1, m)   # 参数空间中的直线为一条竖直线
        for j in range(len(u)):   # 遍历线上所有点
            vindy.append(u[j])   # 纵坐标索引
            vindx.append(x[i])   # 横坐标索引
    # 开始投票
    for j in range(len(vindx)):   # 遍历直线上所有点
        pspace[int(vindy[j]),int(vindx[j])] = pspace[int(vindy[j]),int(vindx[j])] + 1   # 进行投票
cv2.imwrite(path + 'parameter_space.PNG', (pspace / pspace.max()) * 255)   # 储存参数空间投票结果

# 寻找圆心坐标
pspace = cv2.blur(pspace, (9, 9))   # 进行均值模糊以减少奇异值对检测的影响
index1 = np.argmax(pspace)   # 找到投票结果最大值的索引
u_max1, v_max1 = divmod(index1, pspace.shape[1])   # 投票结果最大值的坐标，即第一个圆的圆心坐标
print(u_max1, v_max1)

pspace[u_max1 - 10:u_max1 + 10, v_max1 - 10:v_max1 + 10] = 0   # 将第一个圆心周围一片区域置零，起到屏蔽效果

index2 = np.argmax(pspace)   # 找到当前最大值所在的索引（实际是原结果中的第二大）
u_max2, v_max2 = divmod(index2, pspace.shape[1])   # 第二个圆的圆心坐标
print(u_max2, v_max2)

# 投票寻找半径
# 寻找第一个圆的半径
rs = []
for i in range(len(y)):   # 遍历所有强边界点
    d = np.sqrt((y[i] - u_max1) ** 2 + (x[i] - v_max1) ** 2)   # 求出到圆心的距离
    d = np.rint(d)   # 取整
    rs.append(int(d))
rs = np.array(rs)

rspace = np.zeros([1, rs.max()])   # 定义参数空间
for i in range(len(rs)):   # 遍历所有强边界点
    rspace[0, rs[i]-1] = rspace[0, rs[i]-1] + 1   # 根据半径进行投票

r1 = np.argmax(rspace)   # 投票最大值即为所求的半径

# 寻找第二个圆的半径
rs = []
for i in range(len(y)):   # 遍历所有强边界点
    d = np.sqrt((y[i] - u_max2) ** 2 + (x[i] - v_max2) ** 2)   # 求出到圆心的距离
    d = np.rint(d)   # 取整
    rs.append(int(d))
rs = np.array(rs)

rspace = np.zeros([1, rs.max()])   # 定义参数空间
for i in range(len(rs)):   # 遍历所有强边界点
    rspace[0, rs[i]-1] = rspace[0, rs[i]-1] + 1   # 根据半径进行投票

r2 = np.argmax(rspace)   # 投票最大值即为所求的半径

# 结果可视化
wheel = cv2.imread('./source_images/wheel.png')   # 读入彩色图像
# 在图像上画圆（参数分别为：画板图像，圆心坐标，半径，线色）
cv2.circle(wheel, [v_max1, u_max1], r1, (0, 0, 255))
cv2.circle(wheel, [v_max2, u_max2], r2, (0, 0, 255))
cv2.imwrite(path + 'with_circles.png', wheel)
