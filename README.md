# 基于树莓派的巡线小车/ ESP32 的避障小车

## 简介

本仓库为 上海交通大学 **智能车感知与控制实践(AU2601)** 课程的全部代码，完成于2024年秋季学期，基于 **`树莓派`** / 
**`ESP32`** 实现小车的巡线和避障功能。

欢迎任何的 **`Issues`** 以及 **`Pull requests`**!

## 目录

- **Car**
  - **课程报告**：视觉感知和激光雷达部分的实验报告
  - **课件**：视觉感知和激光雷达部分的课件
  - **Main**：视觉感知部分代码
    - **`main_single.py`**：巡单线代码
    - **`main_double.py`**：巡双线代码
    - **`main_traffic.py`**：识别指示牌代码
  - **RPLidarDriver**：激光雷达部分代码
    - **circle**：环形赛道代码
    - **maze**：迷宫赛道代码
    - **mecanum**：测试用
    - **obstacle_avoidance**：原地避障代码
  - **Train**：识别指示牌的模型训练
    - **CNN**
      - **data**：训练、测试数据集
      - **`four.pth`**：助教提供的模型
      - **`recognize_test_2.py`**：测试用（适用于树莓派32位系统）
      - **`recognize_test_3.py`**：测试用
      - **`recognize_test_4.py`**：测试用
      - **`recognize_test.py`**：测试用（仅适用于本地64位系统）
      - **`traffic_sign.pth`**：自己训练出来的模型
      - **`train.py`**：训练代码

***注意：*** 
  1. **Main**即对应树莓派上的Main文件夹，需在树莓派上运行；
  2. 运行训练代码 **`train.py`** 需以**CNN**文件夹为根目录。

## 邮箱

任何疑问，欢迎邮件交流：**`passerby_zzz@sjtu.edu.cn`** !

## **Wish for your Star⭐!**
