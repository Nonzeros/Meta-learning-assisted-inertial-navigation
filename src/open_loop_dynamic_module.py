"""
开环动力学模型模块
实现纯动力学模型的开环计算（不经过UKF）
用于验证闭环VDM辅助导航的问题
"""

import numpy as np
import mlmodel


def open_loop_dynamic_step_intelligent(
    inputdata,
    outputlabel,
    delta_v,
    delta_p,
    dynamic_a,
    dynamic_P,
    dim_a,
    hover_throttle,
    T_sp,
    Ri,
    vt_minus1,
    pt_minus1,
    px_minus1,
    pw_minus1,
    options,
    deltat=0.02,
):
    """
    元学习模型的开环动力学计算
    
    参数:
        inputdata: 当前输入数据 [v, q, pwm] (1xdim_x)
        outputlabel: 当前标签（用于元学习模型，但开环计算不使用）
        delta_v: 速度误差（用于元学习模型）
        delta_p: 位置误差（用于元学习模型）
        dynamic_a: 动力学参数a（用于元学习模型）
        dynamic_P: 动力学参数P（用于元学习模型）
        dim_a: 动力学参数维度
        hover_throttle: 悬停油门
        T_sp: 推力设定值
        Ri: 旋转矩阵（从真实姿态计算得到）
        vt_minus1: 上一时刻的开环速度 (3x1)
        pt_minus1: 上一时刻的开环位置 (3x1)
        px_minus1: 粒子滤波的粒子状态（用于元学习模型）
        pw_minus1: 粒子滤波的粒子权重（用于元学习模型）
        options: 模型选项
        deltat: 时间步长（默认0.02秒）
    
    返回:
        dynamic_pos: 预测位置 (3x1)
        dynamic_vel: 预测速度 (3x1)
        dynamic_vdot: 加速度 (3x1)
        neural_fa: 气动力 (3x1)
        dynamic_a: 更新后的动力学参数a
        dynamic_P: 更新后的动力学参数P
        px: 更新后的粒子状态
        pw: 更新后的粒子权重
    """
    # 训练好的模型
    dataset = "neural-fly"
    features = ["v", "q", "pwm"]
    modelname = f"{dataset}_dim-a-{dim_a}_{'-'.join(features)}"
    stopping_epoch = 900
    final_model = mlmodel.load_model(
        modelname=modelname + "-epoch-" + str(stopping_epoch)
    )
    
    # 利用滤波的思路实时预测
    Y_bars, dynamic_a, dynamic_P, px, pw = mlmodel.validation_realtime(
        phi_net=final_model.phi,
        h_net=final_model.h,
        inputdata=inputdata,
        outputlabel=outputlabel,
        vel_data=delta_v,
        pos_data=delta_p,
        options=options,
        dynamic_a=dynamic_a,
        dynamic_P=dynamic_P,
        px=px_minus1,
        pw=pw_minus1,
    )
    
    # 纯动力学导航解算
    m0 = 2.6
    g_ = 9.8
    g = np.array([0, 0, -g_]).reshape((3, 1))
    neural_f = Y_bars.reshape(3, 1)
    fT = np.array([0, 0, float(T_sp / hover_throttle) * 9.8 * m0]).reshape((3, 1))
    
    # 加速度计算
    v_dot = g + (Ri @ fT + neural_f) / m0
    
    # 矩形积分
    vt = v_dot * deltat + vt_minus1
    pt = vt * deltat + pt_minus1
    
    return pt, vt, v_dot, neural_f, dynamic_a, dynamic_P, px, pw


def open_loop_dynamic_step_baseline(
    vt_minus1,
    pt_minus1,
    Ri,
    hover_throttle,
    T_sp,
    deltat=0.02,
):
    """
    零气动力模型的开环动力学计算
    
    参数:
        vt_minus1: 上一时刻的开环速度 (3x1)
        pt_minus1: 上一时刻的开环位置 (3x1)
        Ri: 旋转矩阵（从真实姿态计算得到）
        hover_throttle: 悬停油门
        T_sp: 推力设定值
        deltat: 时间步长（默认0.02秒）
    
    返回:
        dynamic_pos: 预测位置 (3x1)
        dynamic_vel: 预测速度 (3x1)
        dynamic_vdot: 加速度 (3x1)
        traditional_fa: 气动力（零向量）(3x1)
    """
    m0 = 2.6
    g_ = 9.8
    g = np.array([0, 0, -g_]).reshape((3, 1))
    fT = np.array([0, 0, float(T_sp / hover_throttle) * 9.8 * m0]).reshape((3, 1))
    
    # 零气动力
    traditional_fa = np.zeros((3, 1))
    
    # 加速度计算
    v_dot = g + (Ri @ fT + traditional_fa) / m0
    
    # 矩形积分
    vt = v_dot * deltat + vt_minus1
    pt = vt * deltat + pt_minus1
    
    return pt, vt, v_dot, traditional_fa


def open_loop_dynamic_step_linear_drag(
    vt_minus1,
    pt_minus1,
    Ri,
    hover_throttle,
    T_sp,
    drag_coefficients,
    deltat=0.02,
):
    """
    线性阻力模型的开环动力学计算
    
    参数:
        vt_minus1: 上一时刻的开环速度 (3x1)
        pt_minus1: 上一时刻的开环位置 (3x1)
        Ri: 旋转矩阵（从真实姿态计算得到）
        hover_throttle: 悬停油门
        T_sp: 推力设定值
        drag_coefficients: 阻力系数 [kx, ky, kz] (3x1或1x3数组)
        deltat: 时间步长（默认0.02秒）
    
    返回:
        dynamic_pos: 预测位置 (3x1)
        dynamic_vel: 预测速度 (3x1)
        dynamic_vdot: 加速度 (3x1)
        traditional_fa: 气动力 (3x1)
    """
    m0 = 2.6
    g_ = 9.8
    g = np.array([0, 0, -g_]).reshape((3, 1))
    fT = np.array([0, 0, float(T_sp / hover_throttle) * 9.8 * m0]).reshape((3, 1))
    
    # 线性阻力模型：F = -k * v
    drag_coefficients = np.array(drag_coefficients).reshape((3, 1))
    traditional_fa = -drag_coefficients * vt_minus1
    
    # 加速度计算
    v_dot = g + (Ri @ fT + traditional_fa) / m0
    
    # 矩形积分
    vt = v_dot * deltat + vt_minus1
    pt = vt * deltat + pt_minus1
    
    return pt, vt, v_dot, traditional_fa


def open_loop_dynamic_step_linear_fit(
    inputdata,
    adaptinput,
    adaptlabel,
    vt_minus1,
    pt_minus1,
    Ri,
    hover_throttle,
    T_sp,
    W_fitted=None,
    deltat=0.02,
):
    """
    线性拟合模型的开环动力学计算
    
    参数:
        inputdata: 当前输入数据 [v, q, pwm] (1xdim_x)
        adaptinput: 适应阶段输入数据 (Nxdim_x)
        adaptlabel: 适应阶段标签（气动力）(Nx3)
        vt_minus1: 上一时刻的开环速度 (3x1)
        pt_minus1: 上一时刻的开环位置 (3x1)
        Ri: 旋转矩阵（从真实姿态计算得到）
        hover_throttle: 悬停油门
        T_sp: 推力设定值
        W_fitted: 已拟合的权重矩阵（可选，如果提供则使用，否则重新计算）
        deltat: 时间步长（默认0.02秒）
    
    返回:
        dynamic_pos: 预测位置 (3x1)
        dynamic_vel: 预测速度 (3x1)
        dynamic_vdot: 加速度 (3x1)
        traditional_fa: 气动力 (3x1)
        W_fitted: 拟合的权重矩阵（用于后续调用）
    """
    m0 = 2.6
    g_ = 9.8
    g = np.array([0, 0, -g_]).reshape((3, 1))
    fT = np.array([0, 0, float(T_sp / hover_throttle) * 9.8 * m0]).reshape((3, 1))
    
    # 如果提供了已拟合的W，使用它；否则重新计算
    if W_fitted is None:
        # 使用最小二乘法拟合线性模型：Y = X * W^T
        X = np.column_stack([adaptinput, np.ones(len(adaptinput))])
        W = np.linalg.lstsq(X, adaptlabel, rcond=None)[0]
    else:
        W = W_fitted
    
    # 使用拟合的模型预测当前时刻的气动力
    X_current = np.append(inputdata, 1)
    traditional_fa = (W.T @ X_current).reshape((3, 1))
    
    # 加速度计算
    v_dot = g + (Ri @ fT + traditional_fa) / m0
    
    # 矩形积分
    vt = v_dot * deltat + vt_minus1
    pt = vt * deltat + pt_minus1
    
    return pt, vt, v_dot, traditional_fa, W


