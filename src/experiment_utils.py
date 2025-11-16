"""
实验工具函数：用于读取验证集数据和配置
"""
import os
import yaml
import numpy as np
import pandas as pd
from ast import literal_eval
from typing import Dict, Tuple, Optional


def load_config(config_path: str) -> Dict:
    """
    加载实验配置文件
    
    参数:
        config_path: 配置文件路径
    
    返回:
        配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def load_validation_csv(csv_path: str) -> Dict:
    """
    从CSV文件加载验证集数据
    
    参数:
        csv_path: CSV文件路径
    
    返回:
        包含p, q, v, t等数据的字典
    """
    df = pd.read_csv(csv_path)
    
    # 解析向量数据（p, q, v等是字符串形式的列表）
    data = {}
    for field in df.columns[1:]:  # 跳过第一列（索引列）
        if isinstance(df[field][0], str):
            # 尝试解析为列表
            try:
                data[field] = np.array([literal_eval(x) for x in df[field]], dtype=float)
            except:
                # 如果解析失败，保持原样
                data[field] = df[field].values
        else:
            data[field] = df[field].values
    
    # 确保时间数据存在
    if 't' in data:
        data['t'] = np.array(data['t'], dtype=float)
    else:
        # 如果没有时间列，从索引推断
        data['t'] = np.arange(len(df)) * 0.02
    
    return data


def prepare_matlab_data(validation_data: Dict, adapt_end_index: int) -> Tuple[np.ndarray, float]:
    """
    准备传给MATLAB的数据
    
    参数:
        validation_data: 验证集数据字典
        adapt_end_index: 适应阶段结束索引
    
    返回:
        (pavq_data, dt): pavq_data是N x 11的矩阵，dt是时间步长
    """
    # 提取数据
    p = validation_data['p']  # N x 3
    q = validation_data['q']  # N x 4
    v = validation_data['v']  # N x 3
    t = validation_data['t']  # N
    
    # 确保数据是2D数组
    if p.ndim == 1:
        p = p.reshape(-1, 1)
    if q.ndim == 1:
        q = q.reshape(-1, 1)
    if v.ndim == 1:
        v = v.reshape(-1, 1)
    
    # 从adapt_end_index开始到文件末尾
    p_seg = p[adapt_end_index:, :]
    q_seg = q[adapt_end_index:, :]
    v_seg = v[adapt_end_index:, :]
    t_seg = t[adapt_end_index:]
    
    # 计算时间步长（取前两个时间点的差值）
    if len(t_seg) > 1:
        dt = float(t_seg[1] - t_seg[0])
    else:
        dt = 0.02  # 默认时间步长
    
    # 组合数据：[p(3列), q(4列), v(3列), t(1列)]
    pavq_data = np.hstack([p_seg, q_seg, v_seg, t_seg.reshape(-1, 1)])
    
    return pavq_data, dt


def calculate_data_length(validation_data: Dict, adapt_end_index: int, 
                         loops: Optional[int] = None, max_duration: Optional[float] = None) -> int:
    """
    计算验证集数据长度
    
    参数:
        validation_data: 验证集数据字典
        adapt_end_index: 适应阶段结束索引
        loops: 指定的循环次数（如果为-1或None，则使用全部数据）
        max_duration: 最大时长（秒）
    
    返回:
        数据长度
    """
    t = validation_data['t']
    total_length = len(t) - adapt_end_index
    
    if loops is not None and loops > 0:
        return min(loops, total_length)
    
    if max_duration is not None:
        # 计算时间步长
        if len(t) > 1:
            dt = float(t[1] - t[0])
        else:
            dt = 0.02
        max_points = int(max_duration / dt)
        return min(max_points, total_length)
    
    return total_length

