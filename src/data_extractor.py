"""
数据提取工具：从CSV文件提取数据并生成MATLAB需要的Excel文件
"""
import os
import pandas as pd
import numpy as np
from ast import literal_eval


def extract_csv_to_excel(csv_file_path, excel_file_path):
    """
    从CSV文件提取p, q, v, t数据并生成Excel文件
    
    参数:
        csv_file_path: CSV文件路径
        excel_file_path: 输出的Excel文件路径
    
    返回:
        success: 是否成功
        error_msg: 错误信息（如果失败）
    """
    try:
        # 读取CSV文件
        df = pd.read_csv(csv_file_path)
        
        # 解析字符串形式的列表数据
        # p: [x, y, z] -> 展开为 x, y, z 三列
        # q: [q1, q2, q3, q4] -> 展开为 q1, q2, q3, q4 四列
        # v: [vx, vy, vz] -> 展开为 vx, vy, vz 三列
        # t: 时间
        
        # 提取p (位置)
        p_data = df['p'].apply(literal_eval)
        x_data = [p[0] for p in p_data]
        y_data = [p[1] for p in p_data]
        z_data = [p[2] for p in p_data]
        
        # 提取q (四元数)
        q_data = df['q'].apply(literal_eval)
        q1_data = [q[0] for q in q_data]
        q2_data = [q[1] for q in q_data]
        q3_data = [q[2] for q in q_data]
        q4_data = [q[3] for q in q_data]
        
        # 提取v (速度)
        v_data = df['v'].apply(literal_eval)
        vx_data = [v[0] for v in v_data]
        vy_data = [v[1] for v in v_data]
        vz_data = [v[2] for v in v_data]
        
        # 提取t (时间)
        t_data = df['t'].values
        
        # 创建新的DataFrame，列顺序：x y z q1 q2 q3 q4 vx vy vz t
        excel_df = pd.DataFrame({
            'x': x_data,
            'y': y_data,
            'z': z_data,
            'q1': q1_data,
            'q2': q2_data,
            'q3': q3_data,
            'q4': q4_data,
            'vx': vx_data,
            'vy': vy_data,
            'vz': vz_data,
            't': t_data
        })
        
        # 确保输出目录存在
        excel_dir = os.path.dirname(excel_file_path)
        if not os.path.exists(excel_dir):
            os.makedirs(excel_dir)
        
        # 保存为Excel文件
        excel_df.to_excel(excel_file_path, index=False, engine='openpyxl')
        
        return True, None
        
    except Exception as e:
        return False, str(e)


def get_excel_filename_from_csv(csv_filename):
    """
    根据CSV文件名生成对应的Excel文件名
    
    参数:
        csv_filename: CSV文件名（如 custom_figure8_baseline_35wind.csv）
    
    返回:
        excel_filename: Excel文件名（如 apts_expi_baseline_35wind.xlsx）
    """
    # 移除.csv扩展名
    base_name = csv_filename.replace('.csv', '')
    
    # 生成Excel文件名
    # 将custom_figure8_baseline_35wind -> apts_expi_baseline_35wind
    # 或者保持原样，只是改扩展名
    excel_filename = f"apts_expi_{base_name.replace('custom_figure8_', '')}.xlsx"
    
    return excel_filename

