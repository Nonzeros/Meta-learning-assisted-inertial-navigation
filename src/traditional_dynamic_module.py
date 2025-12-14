"""
传统动力学模型模块
实现三种传统动力学模型用于与元学习模型对比
"""

import numpy as np


def traditional_dynamic_module_baseline(
    vt_minus1, pt_minus1, Ri, hover_throttle, T_sp, deltat=0.02
):
    """
    传统动力学模型1：零气动力模型（Baseline）
    假设气动力为零，只考虑推力和重力

    参数:
        vt_minus1: 上一时刻的速度 (3x1)
        pt_minus1: 上一时刻的位置 (3x1)
        Ri: 旋转矩阵（从机体坐标系到惯性坐标系）
        hover_throttle: 悬停油门
        T_sp: 推力设定值
        deltat: 时间步长（默认0.02秒）

    返回:
        dynamic_pos: 预测位置 (3x1)
        traditional_fa: 气动力（零向量）(3x1)
        dynamic_vel: 预测速度 (3x1)
        dynamic_vdot: 加速度 (3x1)
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

    return pt, traditional_fa, vt, v_dot


def traditional_dynamic_module_linear_drag(
    vt_minus1, pt_minus1, Ri, hover_throttle, T_sp, drag_coefficients, deltat=0.02
):
    """
    传统动力学模型2：线性阻力模型
    气动力 = -k * v，其中k是阻力系数（3x1向量）

    参数:
        vt_minus1: 上一时刻的速度 (3x1)
        pt_minus1: 上一时刻的位置 (3x1)
        Ri: 旋转矩阵
        hover_throttle: 悬停油门
        T_sp: 推力设定值
        drag_coefficients: 阻力系数 [kx, ky, kz] (3x1或1x3数组)
        deltat: 时间步长（默认0.02秒）

    返回:
        dynamic_pos: 预测位置 (3x1)
        traditional_fa: 气动力 (3x1)
        dynamic_vel: 预测速度 (3x1)
        dynamic_vdot: 加速度 (3x1)
    """
    m0 = 2.6
    g_ = 9.8
    g = np.array([0, 0, -g_]).reshape((3, 1))
    fT = np.array([0, 0, float(T_sp / hover_throttle) * 9.8 * m0]).reshape((3, 1))

    # 线性阻力模型：F = -k * v
    drag_coefficients = np.array(drag_coefficients).reshape((3, 1))
    traditional_fa = -drag_coefficients * Ri.T @ vt_minus1
    traditional_fa = Ri @ traditional_fa
    # 加速度计算
    v_dot = g + (Ri @ fT + Ri @ traditional_fa) / m0

    # 矩形积分
    vt = v_dot * deltat + vt_minus1
    pt = vt * deltat + pt_minus1

    return pt, traditional_fa, vt, v_dot


def traditional_dynamic_module_linear_regression(
    inputdata,
    adaptinput,
    adaptlabel,
    vt_minus1,
    pt_minus1,
    Ri,
    hover_throttle,
    T_sp,
    deltat=0.02,
    W_fitted=None,
):
    """
    传统动力学模型3：适应阶段线性回归模型
    在适应阶段使用线性回归拟合：F_aero = W * [v, q, pwm] + b
    预测阶段使用固定的W和b

    参数:
        inputdata: 当前输入数据 [v, q, pwm] (1xdim_x)
        adaptinput: 适应阶段输入数据 (Nxdim_x)
        adaptlabel: 适应阶段标签（气动力）(Nx3)
        vt_minus1: 上一时刻的速度 (3x1)
        pt_minus1: 上一时刻的位置 (3x1)
        Ri: 旋转矩阵
        hover_throttle: 悬停油门
        T_sp: 推力设定值
        deltat: 时间步长（默认0.02秒）
        W_fitted: 已拟合的权重矩阵（可选，如果提供则使用，否则重新计算）

    返回:
        dynamic_pos: 预测位置 (3x1)
        traditional_fa: 气动力 (3x1)
        dynamic_vel: 预测速度 (3x1)
        dynamic_vdot: 加速度 (3x1)
        W_fitted: 拟合的权重矩阵（用于后续调用）
    """
    m0 = 2.6
    g_ = 9.8
    g = np.array([0, 0, -g_]).reshape((3, 1))
    fT = np.array([0, 0, float(T_sp / hover_throttle) * 9.8 * m0]).reshape((3, 1))

    # 如果提供了已拟合的W，使用它；否则重新计算
    if W_fitted is None:
        # 使用最小二乘法拟合线性模型：Y = X * W^T
        # adaptinput: (N, dim_x), adaptlabel: (N, 3)
        # 添加偏置项
        X = np.column_stack([adaptinput, np.ones(len(adaptinput))])  # (N, dim_x+1)

        # 最小二乘解：W = (X^T * X)^(-1) * X^T * Y
        # W的形状：(dim_x+1, 3)
        W = np.linalg.lstsq(X, adaptlabel, rcond=None)[0]  # (dim_x+1, 3)
    else:
        W = W_fitted

    # 使用拟合的模型预测当前时刻的气动力
    X_current = np.append(inputdata, 1)  # 添加偏置项 (dim_x+1,)
    traditional_fa = (W.T @ X_current).reshape((3, 1))  # (3, 1)

    # 加速度计算
    v_dot = g + (Ri @ fT + traditional_fa) / m0

    # 矩形积分
    vt = v_dot * deltat + vt_minus1
    pt = vt * deltat + pt_minus1

    return pt, traditional_fa, vt, v_dot, W


def estimate_drag_coefficients_from_adaptation(Fbs, vbs):
    """
    从适应阶段的数据估计线性阻力系数
    使用最小二乘法：F_aero = -k * v

    参数:
        Fbs: 适应阶段机体系下的剩余气动力 (Nx3) numpy数组
        vbs: 适应阶段机体坐标系下的速度 (Nx3) numpy数组
    返回:
        drag_coefficients: 阻力系数 [kx, ky, kz] (3x1)
    """
    # 确保输入是 numpy 数组
    vbs = np.array(vbs)  # (N, 3)
    Fbs = np.array(Fbs)  # (N, 3)
    
    # 提取速度部分
    velocities = vbs  # (N, 3)，已经是速度数据，不需要再切片

    # 对于每个方向，拟合 F = -k * v
    # 即：F = k * (-v)，使用最小二乘法
    drag_coefficients = np.zeros(3)

    for i in range(3):
        # F_i = -k_i * v_i
        # 即：F_i = k_i * (-v_i)
        v_i = velocities[:, i].reshape(-1, 1)  # (N, 1)
        F_i = Fbs[:, i].reshape(-1, 1)  # (N, 1)

        # 最小二乘：F_i = -k_i * v_i，即 k_i = -F_i / v_i
        # 使用最小二乘法：k_i = -(v_i^T * v_i)^(-1) * v_i^T * F_i
        if np.sum(v_i**2) > 1e-10:  # 避免除零
            # 使用最小二乘法求解 k_i，使得 F_i ≈ -k_i * v_i
            # 即求解：F_i = k_i * (-v_i)，所以 k_i = (v_i^T * v_i)^(-1) * v_i^T * (-F_i)
            neg_v_i = -v_i
            k_i_result = np.linalg.lstsq(neg_v_i, F_i, rcond=None)[0]
            k_i = k_i_result[0, 0] if k_i_result.ndim > 0 else k_i_result
            # 确保阻力系数为正（阻力应该与速度方向相反）
            drag_coefficients[i] = max(0, k_i)
        else:
            drag_coefficients[i] = 0.0

    return drag_coefficients
