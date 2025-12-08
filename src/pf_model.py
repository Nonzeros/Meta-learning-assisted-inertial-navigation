import math

import numpy as np


def calc_covariance(x_est, px, pw):
    """
    calculate covariance matrix
    """
    cov = np.zeros((3, 3))
    n_particle = px.shape[0]
    for i in range(n_particle):
        dx = px[i] - x_est
        cov += pw[i] * dx @ dx.T
    # cov *= 1.0 / (1.0 - pw.T @ pw)

    return cov


def re_sampling(px, pw, NP):
    """
    low variance re-sampling
    """
    w_cum = np.cumsum(pw)
    base = np.arange(0.0, 1.0, 1 / NP)
    re_sample_id = base + np.random.uniform(0, 1 / NP)
    indexes = []
    ind = 0
    for ip in range(NP):
        while re_sample_id[ip] > w_cum[ind]:
            ind += 1
        indexes.append(ind)

    px = px[indexes, :, :]
    pw = np.zeros((NP, 1)) + 1.0 / NP  # init weight

    return px, pw


def pf_core(px, pw, Y, phi, lambda1, deltaT, numPar, s2, filter_config=None):
    # 初始化 - 从配置文件读取参数
    if filter_config is None:
        # 默认值
        R = 0.1
        q0 = 0.1
    else:
        pf_config = filter_config.get("particle_filter", {})
        R = pf_config.get("R", 0.1)
        q0 = pf_config.get("q0", 0.1)

    Nth = 50
    randQs = np.random.normal(0, q0, (numPar, 3, 3))

    # x_est_bar = np.zeros((3, 3)) # 先验均值

    # 预测
    for i in range(0, numPar):
        px[i] = (1 - lambda1 * deltaT) * px[i] + randQs[i]
        # 计算先验均值
        # x_est_bar = x_est_bar + pw[i] * px[i]

    # 计算先验的方差
    # Ck_bar = calc_covariance(x_est_bar,px,pw)

    # 更新
    for i in range(0, numPar):
        Y_bar = phi @ px[i]
        Y_d = Y - Y_bar
        pw[i] = (
            pw[i]
            * (1 / np.sqrt((2 * math.pi) ** 3 * R))
            * math.exp(-0.5 * Y_d @ np.transpose(Y_d) / R)
            + 1e-199
        )
        # pw[i] = pw[i] * (1 / np.sqrt((2 * math.pi) ** 3 * R)) * math.exp(-0.5 * Y_d @ np.transpose(Y_d) / R)

    # 权重归一化
    sumw = np.sum(pw)
    pw = pw / sumw

    x_est = np.zeros((3, 3))
    # 计算单步滤波均值
    for i in range(0, numPar):
        x_est = x_est + pw[i] * px[i]

    # s2 = np.reshape(s2,(1,3))
    # x_est = x_est + Ck_bar @ np.transpose(phi) @ s2

    # 判断是否重采样
    N_eff = 1.0 / (np.transpose(pw) @ pw)
    if N_eff < Nth:
        px, pw = re_sampling(px, pw, numPar)

    return x_est, px, pw


def pf_core2(
    px, pw, Y, phi, lambda1, deltaT, numPar, s2, p_hso, w_hso, filter_config=None
):
    # 粒子滤波核心代码，适用于HSO改进
    # 初始化 - 从配置文件读取参数
    if filter_config is None:
        # 默认值
        R = 0.1
        q0 = 0.1
    else:
        pf_config = filter_config.get("particle_filter", {})
        R = pf_config.get("R", 0.1)
        q0 = pf_config.get("q0", 0.1)

    Nth = 50
    # randQs = np.random.normal(0,q0,(numPar,3,3))
    # x_est_bar = np.zeros((3, 3)) # 先验均值

    # 预测
    for i in range(0, numPar):
        px[i] = (1 - lambda1 * deltaT) * px[i] + p_hso[i]
        # px[i] = (1 - lambda1 * deltaT) * px[i] + randQs[i]
        # 计算先验均值
        # x_est_bar = x_est_bar + pw[i] * px[i]

    # 计算先验的方差
    # Ck_bar = calc_covariance(x_est_bar,px,pw)

    # 更新
    for i in range(0, numPar):
        Y_bar = phi @ px[i]
        Y_d = Y - Y_bar
        # pw[i] = pw[i] * ( 1/ np.sqrt( (2*math.pi)**3 * R) ) * math.exp(-0.5 * Y_d @ np.transpose(Y_d) / R) + 1e-199
        pw[i] = (
            w_hso[i]
            * (1 / np.sqrt((2 * math.pi) ** 3 * R))
            * math.exp(-0.5 * Y_d @ np.transpose(Y_d) / R)
            + 1e-199
        )
        # pw[i] = pw[i] * (1 / np.sqrt((2 * math.pi) ** 3 * R)) * math.exp(-0.5 * Y_d @ np.transpose(Y_d) / R)

    # 权重归一化
    sumw = np.sum(pw)
    pw = pw / sumw

    x_est = np.zeros((3, 3))
    # 计算单步滤波均值
    for i in range(0, numPar):
        x_est = x_est + pw[i] * px[i]

    # s2 = np.reshape(s2,(1,3))
    # x_est = x_est + Ck_bar @ np.transpose(phi) @ s2

    # 判断是否重采样
    N_eff = 1.0 / (np.transpose(pw) @ pw)
    if N_eff < Nth:
        px, pw = re_sampling(px, pw, numPar)

    return x_est, px, pw
