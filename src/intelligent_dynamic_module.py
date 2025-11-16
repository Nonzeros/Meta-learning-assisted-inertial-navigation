# 梯形积分
## 2025-6-6 引入了质量补偿
## 2025-6-22 引入了和传统模型合外力的比较绘图
## 2025-6-22 引入了双模型作差计算误差（一个模型是a=3，另一个模型是a=6，认为误差等于不那么精确的减去精确的）

import math
import os

import matplotlib
import matplotlib.pyplot as plt

import mlmodel
import numpy as np
import utils
import pandas as pd
import time

# 设置全局字体为支持中文的字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def intelligent_dynamic_module(inputdata, outputlabel, delta_v,delta_p, dynamic_a,dynamic_P, dim_a,hover_throttle,T_sp,Ri,
                               vt_minus1,pt_minus1,px_minus1,pw_minus1,options):


    # 训练好的模型
    dataset = 'neural-fly'
    dataset_folder = 'data/experiment'
    features = ['v', 'q', 'pwm']  # 定义一个列表，包含三个特征名称
    modelname = f"{dataset}_dim-a-{dim_a}_{'-'.join(features)}"

    stopping_epoch = 900 # 取训练第900轮的模型最为最终模型
    final_model = mlmodel.load_model(modelname = modelname + '-epoch-' + str(stopping_epoch)) # 导入最终模型

    # 利用滤波的思路实时预测
    # 从options中获取q0和r0参数
    q0 = options.get('q0', 0.1)
    r0 = options.get('r0', 0.1)
    Y_bars,dynamic_a,dynamic_P,px,pw = mlmodel.validation_realtime(phi_net=final_model.phi, h_net=final_model.h, inputdata=inputdata,
                                         outputlabel=outputlabel,vel_data=delta_v, pos_data=delta_p, options=options,
                                             dynamic_a=dynamic_a,dynamic_P=dynamic_P,px=px_minus1,pw=pw_minus1, q0=q0, r0=r0)

    # 根据适应阶段和预测阶段的数据进行纯动力学导航解算
    deltat = 0.02 # 数据是每0.02s更新一次
    m0 = 2.6 # 原文文献提到他们自定义无人机的质量为2.6kg
    g_ = 9.8
    # g = np.array([0,0,-g_])
    g = np.array([0,0,-g_])
    # g = np.array([0,0,-9.8])
    g = g.reshape((3,1))
    # 用神经网络拟合结果来积分
    neural_f = Y_bars

    # 简单的矩形积分
    m = m0
    neural_f = neural_f.reshape(3, 1)  # 转成列向量
    fT = np.array([0, 0, float(T_sp / hover_throttle) * 9.8 * m0])
    fT = fT.reshape((3, 1))

    fT = np.array([0, 0, float(T_sp / hover_throttle) * 9.8 * m0])
    fT = fT.reshape((3,1))

    v_dot = g + ( Ri @ fT + neural_f ) / m # 3*1 # 神经网络模型

    ## 梯形公式积分
    # 对加速度进行积分
    vt =v_dot * deltat + vt_minus1

    # 对速度进行积分
    # 矩形积分
    pt = vt_minus1 * deltat + pt_minus1
    dynamic_pos = pt
    # 返回更多信息用于日志记录
    return dynamic_pos,dynamic_P,dynamic_a,px,pw,neural_f,vt,v_dot

