"""
绘图工具函数
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from typing import List, Tuple, Optional, Dict

# 设置全局字体为支持中文的字体
matplotlib.rcParams["font.sans-serif"] = ["SimHei"]  # 使用黑体
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


def plot_comparison(
    time_data: np.ndarray,
    calc_data: np.ndarray,
    real_data: np.ndarray,
    labels: List[str],
    title: str = "对比图",
    figsize: Tuple[int, int] = (15, 12),
    baseline_data: Optional[np.ndarray] = None,
):
    """
    绘制计算值与真实值的对比图，支持添加baseline模型对比

    参数:
        time_data: 时间轴数据
        calc_data: 计算值数据 (3 x N) - 元学习模型
        real_data: 真实值数据 (3 x N)
        labels: 各分量的标签列表
        title: 图表标题
        figsize: 图表大小
        baseline_data: baseline模型数据 (3 x N)，可选
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize)

    for i in range(3):
        ax = axes[i]
        # 真实值（最先绘制，作为参考）
        ax.plot(
            time_data,
            real_data[i, :],
            label="真实值",
            linewidth=2.0,
            alpha=0.9,
            linestyle="--",
            color="#06A77D",
        )
        # 元学习模型（计算值）
        ax.plot(
            time_data, 
            calc_data[i, :], 
            label="元学习模型", 
            linewidth=2.0, 
            alpha=0.9,
            color="#2E86AB",
        )
        # Baseline模型（如果提供）
        if baseline_data is not None:
            ax.plot(
                time_data,
                baseline_data[i, :],
                label="零气动力模型",
                linewidth=2.0,
                alpha=0.9,
                linestyle=":",
                color="#F77F00",
        )
        ax.set_xlabel("时间 /s", fontsize=10)
        ax.set_ylabel(labels[i], fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    return fig


def plot_rmse_comparison(
    ukf_metrics: Dict[str, float],
    pure_ins_metrics: Dict[str, float],
    title: str = "RMSE对比",
    baseline_metrics: Optional[Dict[str, float]] = None,
    linear_drag_metrics: Optional[Dict[str, float]] = None,
    fit_metrics: Optional[Dict[str, float]] = None,
):
    """
    绘制所有模型的RMSE对比图

    参数:
        ukf_metrics: 元学习模型UKF的RMSE指标字典
        pure_ins_metrics: 纯惯导的RMSE指标字典
        title: 图表标题
        baseline_metrics: 零气动力模型UKF的RMSE指标字典，可选
        linear_drag_metrics: 线性阻力模型UKF的RMSE指标字典，可选
        fit_metrics: 线性拟合模型UKF的RMSE指标字典，可选
    """
    categories = [
        "速度东向",
        "速度北向",
        "速度天向",
        "位置东向",
        "位置北向",
        "位置天向",
    ]
    ukf_values = [
        ukf_metrics.get("ukf_vel_rmse_east", ukf_metrics.get("vel_rmse_east", 0)),
        ukf_metrics.get("ukf_vel_rmse_north", ukf_metrics.get("vel_rmse_north", 0)),
        ukf_metrics.get("ukf_vel_rmse_up", ukf_metrics.get("vel_rmse_up", 0)),
        ukf_metrics.get("ukf_pos_rmse_east", ukf_metrics.get("pos_rmse_east", 0)),
        ukf_metrics.get("ukf_pos_rmse_north", ukf_metrics.get("pos_rmse_north", 0)),
        ukf_metrics.get("ukf_pos_rmse_up", ukf_metrics.get("pos_rmse_up", 0)),
    ]
    pure_ins_values = [
        pure_ins_metrics.get("pure_ins_vel_rmse_east", pure_ins_metrics.get("vel_rmse_east", 0)),
        pure_ins_metrics.get("pure_ins_vel_rmse_north", pure_ins_metrics.get("vel_rmse_north", 0)),
        pure_ins_metrics.get("pure_ins_vel_rmse_up", pure_ins_metrics.get("vel_rmse_up", 0)),
        pure_ins_metrics.get("pure_ins_pos_rmse_east", pure_ins_metrics.get("pos_rmse_east", 0)),
        pure_ins_metrics.get("pure_ins_pos_rmse_north", pure_ins_metrics.get("pos_rmse_north", 0)),
        pure_ins_metrics.get("pure_ins_pos_rmse_up", pure_ins_metrics.get("pos_rmse_up", 0)),
    ]
    
    # 计算有多少个模型
    num_models = 2  # 元学习模型和纯惯导
    if baseline_metrics is not None:
        num_models += 1
    if linear_drag_metrics is not None:
        num_models += 1
    if fit_metrics is not None:
        num_models += 1
    
    x = np.arange(len(categories))
    
    # 根据模型数量调整柱状图宽度和布局
    if num_models == 5:
        width = 0.15
        fig, ax = plt.subplots(figsize=(16, 6))
        offset = -2 * width
        ax.bar(x + offset, ukf_values, width, label="元学习模型UKF", alpha=0.9, color="#2E86AB")
        offset += width
        ax.bar(x + offset, pure_ins_values, width, label="纯惯导", alpha=0.9, color="#F24236")
        offset += width
        baseline_values = [
            baseline_metrics.get("baseline_vel_rmse_east", baseline_metrics.get("vel_rmse_east", 0)),
            baseline_metrics.get("baseline_vel_rmse_north", baseline_metrics.get("vel_rmse_north", 0)),
            baseline_metrics.get("baseline_vel_rmse_up", baseline_metrics.get("vel_rmse_up", 0)),
            baseline_metrics.get("baseline_pos_rmse_east", baseline_metrics.get("pos_rmse_east", 0)),
            baseline_metrics.get("baseline_pos_rmse_north", baseline_metrics.get("pos_rmse_north", 0)),
            baseline_metrics.get("baseline_pos_rmse_up", baseline_metrics.get("pos_rmse_up", 0)),
        ]
        ax.bar(x + offset, baseline_values, width, label="零气动力模型UKF", alpha=0.9, color="#F77F00")
        offset += width
        linear_drag_values = [
            linear_drag_metrics.get("linear_drag_vel_rmse_east", linear_drag_metrics.get("vel_rmse_east", 0)),
            linear_drag_metrics.get("linear_drag_vel_rmse_north", linear_drag_metrics.get("vel_rmse_north", 0)),
            linear_drag_metrics.get("linear_drag_vel_rmse_up", linear_drag_metrics.get("vel_rmse_up", 0)),
            linear_drag_metrics.get("linear_drag_pos_rmse_east", linear_drag_metrics.get("pos_rmse_east", 0)),
            linear_drag_metrics.get("linear_drag_pos_rmse_north", linear_drag_metrics.get("pos_rmse_north", 0)),
            linear_drag_metrics.get("linear_drag_pos_rmse_up", linear_drag_metrics.get("pos_rmse_up", 0)),
        ]
        ax.bar(x + offset, linear_drag_values, width, label="线性阻力模型UKF", alpha=0.9, color="#9B59B6")
        offset += width
        fit_values = [
            fit_metrics.get("fit_vel_rmse_east", fit_metrics.get("vel_rmse_east", 0)),
            fit_metrics.get("fit_vel_rmse_north", fit_metrics.get("vel_rmse_north", 0)),
            fit_metrics.get("fit_vel_rmse_up", fit_metrics.get("vel_rmse_up", 0)),
            fit_metrics.get("fit_pos_rmse_east", fit_metrics.get("pos_rmse_east", 0)),
            fit_metrics.get("fit_pos_rmse_north", fit_metrics.get("pos_rmse_north", 0)),
            fit_metrics.get("fit_pos_rmse_up", fit_metrics.get("pos_rmse_up", 0)),
        ]
        ax.bar(x + offset, fit_values, width, label="线性拟合模型UKF", alpha=0.9, color="#E67E22")
    elif num_models == 4:
        width = 0.2
        fig, ax = plt.subplots(figsize=(15, 6))
        offset = -1.5 * width
        ax.bar(x + offset, ukf_values, width, label="元学习模型UKF", alpha=0.9, color="#2E86AB")
        offset += width
        ax.bar(x + offset, pure_ins_values, width, label="纯惯导", alpha=0.9, color="#F24236")
        offset += width
        if baseline_metrics is not None:
            baseline_values = [
                baseline_metrics.get("baseline_vel_rmse_east", baseline_metrics.get("vel_rmse_east", 0)),
                baseline_metrics.get("baseline_vel_rmse_north", baseline_metrics.get("vel_rmse_north", 0)),
                baseline_metrics.get("baseline_vel_rmse_up", baseline_metrics.get("vel_rmse_up", 0)),
                baseline_metrics.get("baseline_pos_rmse_east", baseline_metrics.get("pos_rmse_east", 0)),
                baseline_metrics.get("baseline_pos_rmse_north", baseline_metrics.get("pos_rmse_north", 0)),
                baseline_metrics.get("baseline_pos_rmse_up", baseline_metrics.get("pos_rmse_up", 0)),
            ]
            ax.bar(x + offset, baseline_values, width, label="零气动力模型UKF", alpha=0.9, color="#F77F00")
            offset += width
        if linear_drag_metrics is not None:
            linear_drag_values = [
                linear_drag_metrics.get("linear_drag_vel_rmse_east", linear_drag_metrics.get("vel_rmse_east", 0)),
                linear_drag_metrics.get("linear_drag_vel_rmse_north", linear_drag_metrics.get("vel_rmse_north", 0)),
                linear_drag_metrics.get("linear_drag_vel_rmse_up", linear_drag_metrics.get("vel_rmse_up", 0)),
                linear_drag_metrics.get("linear_drag_pos_rmse_east", linear_drag_metrics.get("pos_rmse_east", 0)),
                linear_drag_metrics.get("linear_drag_pos_rmse_north", linear_drag_metrics.get("pos_rmse_north", 0)),
                linear_drag_metrics.get("linear_drag_pos_rmse_up", linear_drag_metrics.get("pos_rmse_up", 0)),
            ]
            ax.bar(x + offset, linear_drag_values, width, label="线性阻力模型UKF", alpha=0.9, color="#9B59B6")
            offset += width
        if fit_metrics is not None:
            fit_values = [
                fit_metrics.get("fit_vel_rmse_east", fit_metrics.get("vel_rmse_east", 0)),
                fit_metrics.get("fit_vel_rmse_north", fit_metrics.get("vel_rmse_north", 0)),
                fit_metrics.get("fit_vel_rmse_up", fit_metrics.get("vel_rmse_up", 0)),
                fit_metrics.get("fit_pos_rmse_east", fit_metrics.get("pos_rmse_east", 0)),
                fit_metrics.get("fit_pos_rmse_north", fit_metrics.get("pos_rmse_north", 0)),
                fit_metrics.get("fit_pos_rmse_up", fit_metrics.get("pos_rmse_up", 0)),
            ]
            ax.bar(x + offset, fit_values, width, label="线性拟合模型UKF", alpha=0.9, color="#E67E22")
    elif baseline_metrics is not None:
        width = 0.25
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.bar(x - width, ukf_values, width, label="元学习模型UKF", alpha=0.9, color="#2E86AB")
        ax.bar(x, pure_ins_values, width, label="纯惯导", alpha=0.9, color="#F24236")
        baseline_values = [
            baseline_metrics.get("baseline_vel_rmse_east", baseline_metrics.get("vel_rmse_east", 0)),
            baseline_metrics.get("baseline_vel_rmse_north", baseline_metrics.get("vel_rmse_north", 0)),
            baseline_metrics.get("baseline_vel_rmse_up", baseline_metrics.get("vel_rmse_up", 0)),
            baseline_metrics.get("baseline_pos_rmse_east", baseline_metrics.get("pos_rmse_east", 0)),
            baseline_metrics.get("baseline_pos_rmse_north", baseline_metrics.get("pos_rmse_north", 0)),
            baseline_metrics.get("baseline_pos_rmse_up", baseline_metrics.get("pos_rmse_up", 0)),
        ]
        ax.bar(x + width, baseline_values, width, label="零气动力模型UKF", alpha=0.9, color="#F77F00")
    else:
        width = 0.35
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width / 2, ukf_values, width, label="元学习模型UKF", alpha=0.9, color="#2E86AB")
        ax.bar(x + width / 2, pure_ins_values, width, label="纯惯导", alpha=0.9, color="#F24236")

    ax.set_xlabel("指标", fontsize=12)
    ax.set_ylabel("RMSE", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    return fig
