# 贾林轩 1853688 形状检测作业
## 一. 运行环境
### 1. 硬件信息
GPU：NVIDIA GeForce GTX 1660 Ti with Max-Q Design 
CPU：Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz
RAM：16GB；
OS：WindowsPE 64bit Version：Windows-10-10.0.19043-SP0

### 2. 软件环境
IDE：pycharm
python Version：3.9.7
software packages：cv2, numpy, os, math
.md阅读器：typora

## 二. 文件结构
1. canny.py：包括canny函数接口及对lanes.png所做的处理，结果储存在‘./results/canny/’中

1. hough.py：包括调用canny.py中的canny函数接口处理wheel.png，以及霍夫圆检测，结果储存在‘./results/hough/’中

1. results文件夹：储存结果（4+3=7张图）

1. resource_images：储存两张原图




