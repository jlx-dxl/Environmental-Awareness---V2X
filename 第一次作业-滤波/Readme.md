# 贾林轩 1853688 滤波作业
## 一. 运行环境
### 1. 硬件信息
GPU：NVIDIA GeForce GTX 1660 Ti with Max-Q Design 
<br />CPU：Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz
<br />RAM：16GB；
<br />OS：WindowsPE 64bit Version：Windows-10-10.0.19043-SP0
### 2. 软件环境
IDE：pycharm
python Version：3.9.7
<br />software packages：cv2, numpy, os, matplotlib.pyplot（用于展示图像）, time（用于计时）
## 二. 文件结构
1. Sobel & Derivative & Bilateral & Guided.py：包括3x3sobel滤波、3x3导数滤波、7x7双边滤波及7x7引导滤波的运行程序（通过调用**自己写的**函数接口实现，具体滤波函数可以在Filters.py中查阅） 

    | 滤波器类型 | Sobel | Derivative | Bilateral | Guided |
    |:---------|-------|------------|-----------|--------|
    | 程序运行时间   |0min35s|0min29s|2min33s|0min7s|
2. Others.py：包括3x3均值、中值、高斯滤波及7x7均值、高斯滤波的操作（均通过调用opencv现有接口实现）
3. source_images: 存放三张原图
4. results: 储存滤波结果
<br />（执行滤波请运行Sobel & Derivative & Bilateral & Guided.py及Others.py，主要原理性注释集中在Filters.py中）
