# 贾林轩 1853688 图像匹配检测作业

## 一. 运行环境

### 1. 硬件信息

GPU：NVIDIA GeForce GTX 1660 Ti with Max-Q Design 
CPU：Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz
RAM：16GB；
OS：WindowsPE 64bit Version：Windows-10-10.0.19043-SP0

### 2. 软件环境

IDE：pycharm
python Version：3.6.15
software packages：opencv-python(3.4.2.16)，opencv-contrib-python(3.4.2.16), numpy(1.19.5), os
.md编辑器：typora

*说明：高版本的python及opencv包不支持SIFT及SURF接口的调用，因此需要采用较低版本的环境*

### 3.环境配置

在Anaconda Prompt中运行：

conda create -n new_env python=3.6.15

conda activate new_env

conda install opencv-python=3.4.2.16

conda install opencv-contrib-python=3.4.2.16

conda install numpy=1.19.5

## 二. 文件结构

1. Stitching.py：主程序，按照特征检测器类型组织（在一种特征检测器下完成所有出图和拼接任务后转向另一特征检测器）

   *说明：使用RANSAC优化前后特征点个数：*

   *SIFT：前：1359；后：395（RANSAC（阈值5.0）前未采用诸如knn等特征点筛选算法，以做对比）*

   *SURF：前：76；后：29（RANSAC（阈值5.0）前即有检测阈值hessianThreshold=7000，阈值越大，特征点数越少）*

   *ORB：前483；后：77（RANSAC（阈值15.0）前未采用诸如knn等特征点筛选算法，以做对比）*

1. results文件夹：按照特征检测器类型分类，分为SIFT，SURF，ORB三个文件夹，其中分别保存直接匹配结果(xxx_match)，RANSAC滤去离群点后的匹配结果(xxx_ransac)，Homography矩阵(xxx_homography)，图像拼接结果(xxx_stitch)

1. source_images：储存两张原图

1. new_env：环境文件夹



