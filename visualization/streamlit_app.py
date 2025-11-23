"""
Streamlit应用，用于可视化MLflow记录的实验数据
"""
import streamlit as st
import pandas as pd
import mlflow
import mlflow.tracking
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os
import glob
import ast
import json
import shutil


def rerun_app():
    """
    重新运行应用（兼容不同版本的Streamlit）
    """
    try:
        if hasattr(st, 'rerun'):
            st.rerun()
        elif hasattr(st, 'experimental_rerun'):
            st.experimental_rerun()
        else:
            # 如果都不存在，尝试使用其他方法
            st.cache_data.clear()
    except (AttributeError, Exception):
        # 如果所有方法都失败，至少清除缓存
        st.cache_data.clear()


def load_mlflow_experiments(tracking_uri: str = "./mlruns"):
    """
    加载MLflow实验数据
    
    参数:
        tracking_uri: MLflow跟踪URI
    """
    # 转换URI格式（如果是绝对路径）
    if os.path.isabs(tracking_uri):
        if os.name == 'nt':  # Windows
            normalized_path = tracking_uri.replace('\\', '/')
            if ':' in normalized_path:
                parts = normalized_path.split(':', 1)
                normalized_path = f"/{parts[0]}:{parts[1]}"
            tracking_uri = f"file://{normalized_path}"
        else:
            tracking_uri = f"file://{tracking_uri}"
    
    mlflow.set_tracking_uri(tracking_uri)
    
    experiments = []
    try:
        client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
        experiment_list = client.search_experiments()
        
        for exp in experiment_list:
            runs = client.search_runs(experiment_ids=[exp.experiment_id])
            for run in runs:
                run_data = {
                    'experiment_name': exp.name,
                    'run_id': run.info.run_id,
                    'run_name': run.info.run_name,
                    'start_time': datetime.fromtimestamp(run.info.start_time / 1000),
                    'status': run.info.status,
                }
                
                # 添加参数
                for key, value in run.data.params.items():
                    run_data[f'param_{key}'] = value
                
                # 添加指标
                for key, value in run.data.metrics.items():
                    run_data[f'metric_{key}'] = value
                
                # 获取日志文件路径（如果存在）
                log_file_param = run.data.params.get('log_file', None)
                if log_file_param:
                    run_data['log_file'] = log_file_param
                    # 尝试从artifact获取完整路径
                    artifact_uri = run.info.artifact_uri
                    if artifact_uri:
                        run_data['artifact_uri'] = artifact_uri
                
                experiments.append(run_data)
    except Exception as e:
        st.error(f"加载MLflow数据时出错: {str(e)}")
    
    return pd.DataFrame(experiments)


def load_log_file(log_file_name: str, project_root: str = ".", task_batch_folder: str = None):
    """
    加载导航日志文件
    
    参数:
        log_file_name: 日志文件名
        project_root: 项目根目录
        task_batch_folder: 大任务文件夹名称（可选）
    """
    # 确保 task_batch_folder 是字符串或 None（处理 pandas NaN 的情况）
    if task_batch_folder is not None:
        if pd.isna(task_batch_folder):
            task_batch_folder = None
        else:
            task_batch_folder = str(task_batch_folder)
    
    # 确保 log_file_name 是字符串
    if log_file_name is not None:
        if pd.isna(log_file_name):
            log_file_name = None
        else:
            log_file_name = str(log_file_name)
    
    if not log_file_name:
        return None
    
    # 尝试多个可能的路径
    possible_paths = []
    
    # 如果指定了大任务文件夹，优先在大任务文件夹中查找
    if task_batch_folder:
        possible_paths.append(os.path.join(project_root, "navigation_logs", task_batch_folder, log_file_name))
    
    # 添加其他可能的路径
    possible_paths.extend([
        os.path.join(project_root, "navigation_logs", log_file_name),
        os.path.join(project_root, log_file_name),
        log_file_name,
    ])
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                return df
            except Exception as e:
                st.error(f"读取日志文件失败 {path}: {str(e)}")
                return None
    
    # 如果找不到，尝试在navigation_logs目录及其子目录中搜索
    log_dir = os.path.join(project_root, "navigation_logs")
    if os.path.exists(log_dir):
        # 先在大任务文件夹中搜索
        if task_batch_folder:
            task_batch_path = os.path.join(log_dir, task_batch_folder)
            if os.path.exists(task_batch_path):
                pattern = os.path.join(task_batch_path, f"*{log_file_name}*")
                matches = glob.glob(pattern)
                if matches:
                    try:
                        df = pd.read_csv(matches[0])
                        return df
                    except Exception as e:
                        st.error(f"读取日志文件失败 {matches[0]}: {str(e)}")
        
        # 在所有大任务文件夹中搜索
        task_batch_pattern = os.path.join(log_dir, "task_batch_*")
        task_batch_dirs = glob.glob(task_batch_pattern)
        for task_dir in task_batch_dirs:
            pattern = os.path.join(task_dir, f"*{log_file_name}*")
            matches = glob.glob(pattern)
            if matches:
                try:
                    df = pd.read_csv(matches[0])
                    return df
                except Exception as e:
                    st.error(f"读取日志文件失败 {matches[0]}: {str(e)}")
        
        # 最后在整个navigation_logs目录中搜索
        pattern = os.path.join(log_dir, f"*{log_file_name}*")
        matches = glob.glob(pattern)
        if matches:
            try:
                df = pd.read_csv(matches[0])
                return df
            except Exception as e:
                st.error(f"读取日志文件失败 {matches[0]}: {str(e)}")
    
    return None


def get_task_batch_notes_file(project_root: str = ".") -> str:
    """
    获取任务批次备注文件路径
    
    参数:
        project_root: 项目根目录
    
    返回:
        备注文件路径
    """
    return os.path.join(project_root, "navigation_logs", "task_batch_notes.json")


def load_task_batch_notes(project_root: str = ".") -> dict:
    """
    加载任务批次备注
    
    参数:
        project_root: 项目根目录
    
    返回:
        备注字典 {task_batch_folder: note}
    """
    notes_file = get_task_batch_notes_file(project_root)
    if os.path.exists(notes_file):
        try:
            with open(notes_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"读取备注文件失败: {str(e)}")
            return {}
    return {}


def save_task_batch_note(task_batch_folder: str, note: str, project_root: str = ".") -> bool:
    """
    保存任务批次备注
    
    参数:
        task_batch_folder: 任务批次文件夹名称
        note: 备注内容
        project_root: 项目根目录
    
    返回:
        是否保存成功
    """
    try:
        notes_file = get_task_batch_notes_file(project_root)
        notes = load_task_batch_notes(project_root)
        
        if note.strip():
            notes[task_batch_folder] = note.strip()
        else:
            # 如果备注为空，删除该条目
            notes.pop(task_batch_folder, None)
        
        # 确保目录存在
        os.makedirs(os.path.dirname(notes_file), exist_ok=True)
        
        # 保存到文件
        with open(notes_file, 'w', encoding='utf-8') as f:
            json.dump(notes, f, ensure_ascii=False, indent=2)
        
        return True
    except Exception as e:
        st.error(f"保存备注失败: {str(e)}")
        return False


def delete_task_batch_folder(task_batch_folder: str, project_root: str = ".", tracking_uri: str = "./mlruns") -> bool:
    """
    删除任务批次文件夹及其相关的MLflow运行记录
    
    参数:
        task_batch_folder: 任务批次文件夹名称
        project_root: 项目根目录
        tracking_uri: MLflow跟踪URI
    
    返回:
        是否删除成功
    """
    try:
        # 1. 删除navigation_logs下的文件夹
        folder_path = os.path.join(project_root, "navigation_logs", task_batch_folder)
        folder_deleted = False
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            shutil.rmtree(folder_path)
            folder_deleted = True
        
        # 2. 删除相关的MLflow运行记录
        mlflow_runs_deleted = 0
        try:
            # 转换URI格式
            if os.path.isabs(tracking_uri):
                if os.name == 'nt':  # Windows
                    normalized_path = tracking_uri.replace('\\', '/')
                    if ':' in normalized_path:
                        parts = normalized_path.split(':', 1)
                        normalized_path = f"/{parts[0]}:{parts[1]}"
                    tracking_uri_normalized = f"file://{normalized_path}"
                else:
                    tracking_uri_normalized = f"file://{tracking_uri}"
            else:
                # 相对路径，转换为绝对路径
                abs_tracking_uri = os.path.abspath(tracking_uri)
                if os.name == 'nt':  # Windows
                    normalized_path = abs_tracking_uri.replace('\\', '/')
                    if ':' in normalized_path:
                        parts = normalized_path.split(':', 1)
                        normalized_path = f"/{parts[0]}:{parts[1]}"
                    tracking_uri_normalized = f"file://{normalized_path}"
                else:
                    tracking_uri_normalized = f"file://{abs_tracking_uri}"
            
            client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri_normalized)
            
            # 搜索所有包含该task_batch_folder的运行
            experiment_list = client.search_experiments()
            for exp in experiment_list:
                # 搜索该实验下所有包含task_batch_folder参数的运行
                runs = client.search_runs(
                    experiment_ids=[exp.experiment_id],
                    filter_string=f"params.task_batch_folder = '{task_batch_folder}'"
                )
                
                for run in runs:
                    try:
                        # 删除运行（包括artifacts）
                        client.delete_run(run.info.run_id)
                        mlflow_runs_deleted += 1
                    except Exception as e:
                        # 如果删除失败，记录错误但继续
                        print(f"删除MLflow运行 {run.info.run_id} 失败: {str(e)}")
        
        except Exception as e:
            # MLflow删除失败不影响整体流程，但记录错误
            print(f"删除MLflow运行记录时出错: {str(e)}")
        
        # 3. 同时删除备注
        notes = load_task_batch_notes(project_root)
        notes.pop(task_batch_folder, None)
        notes_file = get_task_batch_notes_file(project_root)
        if os.path.exists(notes_file):
            with open(notes_file, 'w', encoding='utf-8') as f:
                json.dump(notes, f, ensure_ascii=False, indent=2)
        
        # 返回结果和详细信息
        if folder_deleted or mlflow_runs_deleted > 0:
            if mlflow_runs_deleted > 0:
                st.info(f"已删除 {mlflow_runs_deleted} 个MLflow运行记录")
            return True
        else:
            st.error(f"文件夹不存在: {folder_path}")
            return False
            
    except Exception as e:
        st.error(f"删除文件夹失败: {str(e)}")
        return False


def extract_param_value(row, param_name):
    """
    从MLflow数据行中提取参数值
    
    参数:
        row: MLflow数据行
        param_name: 参数名称（如 'Rk', 'lambda1', 'Q', 'R'）
    
    返回:
        参数值（字符串或数值）
    """
    if param_name == 'Rk':
        # Rk可能存储在ukf_Rk中，可能是字符串格式的列表
        rk_str = row.get('param_ukf_Rk', '')
        if rk_str:
            try:
                # 尝试解析字符串列表，如 "[10.0, 10.0, 10.0]"
                rk_list = ast.literal_eval(rk_str)
                if isinstance(rk_list, list) and len(rk_list) > 0:
                    # 如果三个值相同，只返回一个值
                    if len(rk_list) == 3 and rk_list[0] == rk_list[1] == rk_list[2]:
                        return rk_list[0]
                    else:
                        return rk_list[0]  # 返回第一个值作为代表
            except:
                pass
        return 'N/A'
    elif param_name == 'lambda1':
        return row.get('param_filter_lambda1', 'N/A')
    elif param_name == 'Q':
        return row.get('param_filter_Q', 'N/A')
    elif param_name == 'R':
        return row.get('param_filter_R', 'N/A')
    elif param_name == 'numPar':
        return row.get('param_filter_numPar', 'N/A')
    else:
        # 尝试直接获取
        param_key = f'param_{param_name}'
        return row.get(param_key, 'N/A')


def create_summary_table(df):
    """
    创建任务汇总表，显示参数和指标
    """
    # 创建汇总表
    summary_data = []
    for idx, row in df.iterrows():
        summary_row = {
            'run_id': row.get('run_id', ''),  # 保存run_id用于跳转
            'run_name_key': row.get('run_name', 'Unknown'),  # 保存run_name用于跳转
            '实验时间': row.get('start_time', ''),
        }
        
        # CSV文件名称
        csv_filename = row.get('param_csv_filename', '')
        if not csv_filename:
            # 尝试从run_name中提取
            run_name = row.get('run_name', '')
            if run_name:
                # run_name格式可能是：custom_figure8_baseline_35wind_PF_R0.1_q00.1_...
                parts = run_name.split('_')
                if 'custom' in parts:
                    csv_idx = parts.index('custom')
                    csv_filename = '_'.join(parts[csv_idx:csv_idx+5]) if len(parts) > csv_idx+4 else run_name
                else:
                    csv_filename = run_name
        summary_row['CSV文件'] = csv_filename if csv_filename else 'Unknown'
        
        # 计算速度改善率（UKF相比纯惯导的改善百分比）
        # 速度东方向改善率
        ukf_vel_east = row.get('metric_ukf_vel_rmse_east', None)
        pure_vel_east = row.get('metric_pure_ins_vel_rmse_east', None)
        if pd.notna(ukf_vel_east) and pd.notna(pure_vel_east) and pure_vel_east > 0:
            vel_east_improvement = ((pure_vel_east - ukf_vel_east) / pure_vel_east * 100)
            summary_row['速度改善率_东(%)'] = f"{vel_east_improvement:.2f}"
        else:
            summary_row['速度改善率_东(%)'] = 'N/A'
        
        # 速度北方向改善率
        ukf_vel_north = row.get('metric_ukf_vel_rmse_north', None)
        pure_vel_north = row.get('metric_pure_ins_vel_rmse_north', None)
        if pd.notna(ukf_vel_north) and pd.notna(pure_vel_north) and pure_vel_north > 0:
            vel_north_improvement = ((pure_vel_north - ukf_vel_north) / pure_vel_north * 100)
            summary_row['速度改善率_北(%)'] = f"{vel_north_improvement:.2f}"
        else:
            summary_row['速度改善率_北(%)'] = 'N/A'
        
        # 速度天方向改善率
        ukf_vel_up = row.get('metric_ukf_vel_rmse_up', None)
        pure_vel_up = row.get('metric_pure_ins_vel_rmse_up', None)
        if pd.notna(ukf_vel_up) and pd.notna(pure_vel_up) and pure_vel_up > 0:
            vel_up_improvement = ((pure_vel_up - ukf_vel_up) / pure_vel_up * 100)
            summary_row['速度改善率_天(%)'] = f"{vel_up_improvement:.2f}"
        else:
            summary_row['速度改善率_天(%)'] = 'N/A'
        
        # 速度总RMSE改善率
        ukf_vel_total = row.get('metric_ukf_vel_rmse_total', None)
        pure_vel_total = row.get('metric_pure_ins_vel_rmse_total', None)
        if pd.notna(ukf_vel_total) and pd.notna(pure_vel_total) and pure_vel_total > 0:
            vel_total_improvement = ((pure_vel_total - ukf_vel_total) / pure_vel_total * 100)
            summary_row['速度改善率_总(%)'] = f"{vel_total_improvement:.2f}"
        else:
            summary_row['速度改善率_总(%)'] = 'N/A'
        
        # 实验参数
        filter_type = row.get('param_filter_solve_type', '')
        if filter_type == '1':
            summary_row['滤波器类型'] = 'KF'
        elif filter_type == '2':
            summary_row['滤波器类型'] = 'PF'
        else:
            summary_row['滤波器类型'] = filter_type if filter_type else 'Unknown'
        
        summary_row['lambda1'] = f"{row.get('param_filter_lambda1', 'N/A')}"
        summary_row['numPar'] = f"{row.get('param_filter_numPar', 'N/A')}"
        summary_row['Q'] = f"{row.get('param_filter_Q', 'N/A')}"
        summary_row['R'] = f"{row.get('param_filter_R', 'N/A')}"
        
        # 提取Rk参数值（用于参数分析）
        rk_value = extract_param_value(row, 'Rk')
        summary_row['Rk'] = f"{rk_value}" if rk_value != 'N/A' else 'N/A'
        
        # 所有RMSE（除了姿态）
        # UKF速度RMSE
        summary_row['UKF速度RMSE_东'] = f"{row.get('metric_ukf_vel_rmse_east', 0):.6f}" if pd.notna(row.get('metric_ukf_vel_rmse_east')) else 'N/A'
        summary_row['UKF速度RMSE_北'] = f"{row.get('metric_ukf_vel_rmse_north', 0):.6f}" if pd.notna(row.get('metric_ukf_vel_rmse_north')) else 'N/A'
        summary_row['UKF速度RMSE_天'] = f"{row.get('metric_ukf_vel_rmse_up', 0):.6f}" if pd.notna(row.get('metric_ukf_vel_rmse_up')) else 'N/A'
        summary_row['UKF速度RMSE_总'] = f"{row.get('metric_ukf_vel_rmse_total', 0):.6f}" if pd.notna(row.get('metric_ukf_vel_rmse_total')) else 'N/A'
        
        # UKF位置RMSE
        summary_row['UKF位置RMSE_东'] = f"{row.get('metric_ukf_pos_rmse_east', 0):.6f}" if pd.notna(row.get('metric_ukf_pos_rmse_east')) else 'N/A'
        summary_row['UKF位置RMSE_北'] = f"{row.get('metric_ukf_pos_rmse_north', 0):.6f}" if pd.notna(row.get('metric_ukf_pos_rmse_north')) else 'N/A'
        summary_row['UKF位置RMSE_天'] = f"{row.get('metric_ukf_pos_rmse_up', 0):.6f}" if pd.notna(row.get('metric_ukf_pos_rmse_up')) else 'N/A'
        summary_row['UKF位置RMSE_总'] = f"{row.get('metric_ukf_pos_rmse_total', 0):.6f}" if pd.notna(row.get('metric_ukf_pos_rmse_total')) else 'N/A'
        
        # 纯惯导速度RMSE
        summary_row['纯惯导速度RMSE_东'] = f"{row.get('metric_pure_ins_vel_rmse_east', 0):.6f}" if pd.notna(row.get('metric_pure_ins_vel_rmse_east')) else 'N/A'
        summary_row['纯惯导速度RMSE_北'] = f"{row.get('metric_pure_ins_vel_rmse_north', 0):.6f}" if pd.notna(row.get('metric_pure_ins_vel_rmse_north')) else 'N/A'
        summary_row['纯惯导速度RMSE_天'] = f"{row.get('metric_pure_ins_vel_rmse_up', 0):.6f}" if pd.notna(row.get('metric_pure_ins_vel_rmse_up')) else 'N/A'
        summary_row['纯惯导速度RMSE_总'] = f"{row.get('metric_pure_ins_vel_rmse_total', 0):.6f}" if pd.notna(row.get('metric_pure_ins_vel_rmse_total')) else 'N/A'
        
        # 纯惯导位置RMSE
        summary_row['纯惯导位置RMSE_东'] = f"{row.get('metric_pure_ins_pos_rmse_east', 0):.6f}" if pd.notna(row.get('metric_pure_ins_pos_rmse_east')) else 'N/A'
        summary_row['纯惯导位置RMSE_北'] = f"{row.get('metric_pure_ins_pos_rmse_north', 0):.6f}" if pd.notna(row.get('metric_pure_ins_pos_rmse_north')) else 'N/A'
        summary_row['纯惯导位置RMSE_天'] = f"{row.get('metric_pure_ins_pos_rmse_up', 0):.6f}" if pd.notna(row.get('metric_pure_ins_pos_rmse_up')) else 'N/A'
        summary_row['纯惯导位置RMSE_总'] = f"{row.get('metric_pure_ins_pos_rmse_total', 0):.6f}" if pd.notna(row.get('metric_pure_ins_pos_rmse_total')) else 'N/A'
        
        # 气动力RMSE
        summary_row['气动力RMSE_x'] = f"{row.get('metric_fa_rmse_x', 0):.6f}" if pd.notna(row.get('metric_fa_rmse_x')) else 'N/A'
        summary_row['气动力RMSE_y'] = f"{row.get('metric_fa_rmse_y', 0):.6f}" if pd.notna(row.get('metric_fa_rmse_y')) else 'N/A'
        summary_row['气动力RMSE_z'] = f"{row.get('metric_fa_rmse_z', 0):.6f}" if pd.notna(row.get('metric_fa_rmse_z')) else 'N/A'
        summary_row['气动力RMSE_总'] = f"{row.get('metric_fa_rmse_total', 0):.6f}" if pd.notna(row.get('metric_fa_rmse_total')) else 'N/A'
        
        summary_data.append(summary_row)
    
    return pd.DataFrame(summary_data)


def plot_time_series(log_df, run_name: str):
    """
    绘制时间序列对比图：位置、速度、姿态
    对于速度和位置，xyz三个方向分别各一张图
    每张图包含：纯惯导、UKF融合结果、真实结果
    
    参考main.py中的绘图逻辑：
    - 排除最后几个数据点（exclude_last）
    - UKF融合结果：蓝色实线 (#2E86AB)
    - 纯惯导解：红色虚线 (#F24236, dash)
    - 真实值：绿色点划线 (#06A77D, dot)
    """
    if log_df is None or log_df.empty:
        return [], [], []
    
    time_col = 'time'
    if time_col not in log_df.columns:
        st.warning("日志文件中没有找到时间列")
        return [], [], []
    
    # 排除最后几个数据点（与main.py保持一致）
    exclude_last = min(5, len(log_df) - 1)
    if exclude_last > 0:
        log_df_plot = log_df.iloc[:-exclude_last].copy()
    else:
        log_df_plot = log_df.copy()
    
    # 位置对比图（X、Y、Z各一张）
    pos_figs = []
    pos_directions = ['x', 'y', 'z']
    pos_labels = ['东向', '北向', '天向']
    
    for dir, label in zip(pos_directions, pos_labels):
        fig = go.Figure()
        
        # 真实位置 - 使用日志文件中的real_px/py/pz（先绘制真实值）
        real_col = f'real_p{dir}'
        if real_col in log_df_plot.columns:
            valid_mask = pd.notna(log_df_plot[real_col]) & (log_df_plot[real_col] != 0)
            if valid_mask.any():
                fig.add_trace(go.Scatter(
                    x=log_df_plot.loc[valid_mask, time_col],
                    y=log_df_plot.loc[valid_mask, real_col],
                    name='真实值',
                    mode='lines',
                    line=dict(color='#06A77D', width=2, dash='dot')
                ))
        
        # UKF融合位置 - 使用日志文件中的ukf_fused_px/py/pz
        ukf_col = f'ukf_fused_p{dir}'
        if ukf_col in log_df_plot.columns:
            # 过滤掉无效值（NaN或0）
            valid_mask = pd.notna(log_df_plot[ukf_col]) & (log_df_plot[ukf_col] != 0)
            if valid_mask.any():
                fig.add_trace(go.Scatter(
                    x=log_df_plot.loc[valid_mask, time_col],
                    y=log_df_plot.loc[valid_mask, ukf_col],
                    name='UKF融合结果',
                    mode='lines',
                    line=dict(color='#2E86AB', width=2)
                ))
        
        # 纯惯导位置 - 使用日志文件中的pure_ins_px/py/pz
        pure_col = f'pure_ins_p{dir}'
        if pure_col in log_df_plot.columns:
            valid_mask = pd.notna(log_df_plot[pure_col]) & (log_df_plot[pure_col] != 0)
            if valid_mask.any():
                fig.add_trace(go.Scatter(
                    x=log_df_plot.loc[valid_mask, time_col],
                    y=log_df_plot.loc[valid_mask, pure_col],
                    name='纯惯导',
                    mode='lines',
                    line=dict(color='#F24236', width=2, dash='dash')
                ))
        
        fig.update_layout(
            title=f'{label}位置对比',
            xaxis_title='时间 (s)',
            yaxis_title=f'{label}位置 (m)',
            height=400,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        pos_figs.append(fig)
    
    # 速度对比图（X、Y、Z各一张）
    vel_figs = []
    vel_directions = ['x', 'y', 'z']
    vel_labels = ['东向', '北向', '天向']
    
    for dir, label in zip(vel_directions, vel_labels):
        fig = go.Figure()
        
        # 真实速度 - 使用日志文件中的real_vx/vy/vz（先绘制真实值）
        real_col = f'real_v{dir}'
        if real_col in log_df_plot.columns:
            valid_mask = pd.notna(log_df_plot[real_col])
            if valid_mask.any():
                fig.add_trace(go.Scatter(
                    x=log_df_plot.loc[valid_mask, time_col],
                    y=log_df_plot.loc[valid_mask, real_col],
                    name='真实值',
                    mode='lines',
                    line=dict(color='#06A77D', width=2, dash='dot')
                ))
        
        # UKF融合速度 - 使用日志文件中的ukf_fused_vx/vy/vz
        ukf_col = f'ukf_fused_v{dir}'
        if ukf_col in log_df_plot.columns:
            valid_mask = pd.notna(log_df_plot[ukf_col])
            if valid_mask.any():
                fig.add_trace(go.Scatter(
                    x=log_df_plot.loc[valid_mask, time_col],
                    y=log_df_plot.loc[valid_mask, ukf_col],
                    name='UKF融合结果',
                    mode='lines',
                    line=dict(color='#2E86AB', width=2)
                ))
        
        # 纯惯导速度 - 使用日志文件中的pure_ins_vx/vy/vz
        pure_col = f'pure_ins_v{dir}'
        if pure_col in log_df_plot.columns:
            valid_mask = pd.notna(log_df_plot[pure_col])
            if valid_mask.any():
                fig.add_trace(go.Scatter(
                    x=log_df_plot.loc[valid_mask, time_col],
                    y=log_df_plot.loc[valid_mask, pure_col],
                    name='纯惯导',
                    mode='lines',
                    line=dict(color='#F24236', width=2, dash='dash')
                ))
        
        fig.update_layout(
            title=f'{label}速度对比',
            xaxis_title='时间 (s)',
            yaxis_title=f'{label}速度 (m/s)',
            height=400,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        vel_figs.append(fig)
    
    # 姿态对比图（X、Y、Z各一张）
    att_figs = []
    att_directions = ['x', 'y', 'z']
    att_labels = ['X', 'Y', 'Z']
    
    for dir, label in zip(att_directions, att_labels):
        fig = go.Figure()
        
        # 真实姿态 - 使用日志文件中的real_att_x/y/z（先绘制真实值）
        real_col = f'real_att_{dir}'
        if real_col in log_df_plot.columns:
            valid_mask = pd.notna(log_df_plot[real_col])
            if valid_mask.any():
                fig.add_trace(go.Scatter(
                    x=log_df_plot.loc[valid_mask, time_col],
                    y=log_df_plot.loc[valid_mask, real_col],
                    name='真实值',
                    mode='lines',
                    line=dict(color='#06A77D', width=2, dash='dot')
                ))
        
        # UKF融合姿态 - 使用日志文件中的ukf_fused_att_x/y/z
        ukf_col = f'ukf_fused_att_{dir}'
        if ukf_col in log_df_plot.columns:
            valid_mask = pd.notna(log_df_plot[ukf_col])
            if valid_mask.any():
                fig.add_trace(go.Scatter(
                    x=log_df_plot.loc[valid_mask, time_col],
                    y=log_df_plot.loc[valid_mask, ukf_col],
                    name='UKF融合结果',
                    mode='lines',
                    line=dict(color='#2E86AB', width=2)
                ))
        
        # 纯惯导姿态 - 使用日志文件中的pure_ins_att_x/y/z
        pure_col = f'pure_ins_att_{dir}'
        if pure_col in log_df_plot.columns:
            valid_mask = pd.notna(log_df_plot[pure_col])
            if valid_mask.any():
                fig.add_trace(go.Scatter(
                    x=log_df_plot.loc[valid_mask, time_col],
                    y=log_df_plot.loc[valid_mask, pure_col],
                    name='纯惯导',
                    mode='lines',
                    line=dict(color='#F24236', width=2, dash='dash')
                ))
        
        fig.update_layout(
            title=f'姿态{label}对比',
            xaxis_title='时间 (s)',
            yaxis_title=f'姿态{label} (度)',
            height=400,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        att_figs.append(fig)
    
    return pos_figs, vel_figs, att_figs


def plot_aerodynamic_force(log_df, run_name: str):
    """
    绘制气动力时间序列对比图
    1. neural_f vs real_fa（X、Y、Z三个方向）
    2. neural_f_total vs real_fa_total（X、Y、Z三个方向）
    """
    if log_df is None or log_df.empty:
        return [], []
    
    time_col = 'time'
    if time_col not in log_df.columns:
        st.warning("日志文件中没有找到时间列")
        return [], []
    
    # 排除最后几个数据点（与main.py保持一致）
    exclude_last = min(5, len(log_df) - 1)
    if exclude_last > 0:
        log_df_plot = log_df.iloc[:-exclude_last].copy()
    else:
        log_df_plot = log_df.copy()
    
    # 图1：neural_f vs real_fa（X、Y、Z各一张）
    fa_figs = []
    fa_directions = ['x', 'y', 'z']
    fa_labels = ['X', 'Y', 'Z']
    
    for dir, label in zip(fa_directions, fa_labels):
        fig = go.Figure()
        
        # 真实气动力 - real_fa_x/y/z
        real_col = f'real_fa_{dir}'
        if real_col in log_df_plot.columns:
            valid_mask = pd.notna(log_df_plot[real_col])
            if valid_mask.any():
                fig.add_trace(go.Scatter(
                    x=log_df_plot.loc[valid_mask, time_col],
                    y=log_df_plot.loc[valid_mask, real_col],
                    name='真实气动力',
                    mode='lines',
                    line=dict(color='#06A77D', width=2, dash='dot')
                ))
        
        # 神经网络预测气动力 - neural_fa_x/y/z
        neural_col = f'neural_fa_{dir}'
        if neural_col in log_df_plot.columns:
            valid_mask = pd.notna(log_df_plot[neural_col])
            if valid_mask.any():
                fig.add_trace(go.Scatter(
                    x=log_df_plot.loc[valid_mask, time_col],
                    y=log_df_plot.loc[valid_mask, neural_col],
                    name='神经网络预测',
                    mode='lines',
                    line=dict(color='#2E86AB', width=2)
                ))
        
        fig.update_layout(
            title=f'气动力{label}方向对比（neural_f vs real_fa）',
            xaxis_title='时间 (s)',
            yaxis_title=f'气动力{label} (N)',
            height=400,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        fa_figs.append(fig)
    
    # 图2：neural_f_total vs real_fa_total（X、Y、Z各一张）
    fa_total_figs = []
    
    for dir, label in zip(fa_directions, fa_labels):
        fig = go.Figure()
        
        # 真实总力 - real_fa_total_x/y/z
        real_total_col = f'real_fa_total_{dir}'
        if real_total_col in log_df_plot.columns:
            valid_mask = pd.notna(log_df_plot[real_total_col])
            if valid_mask.any():
                fig.add_trace(go.Scatter(
                    x=log_df_plot.loc[valid_mask, time_col],
                    y=log_df_plot.loc[valid_mask, real_total_col],
                    name='参考总气动力',
                    mode='lines',
                    line=dict(color='#06A77D', width=2, dash='dot')
                ))
        
        # 神经网络预测总力 - neural_f_total_x/y/z
        neural_total_col = f'neural_f_total_{dir}'
        if neural_total_col in log_df_plot.columns:
            valid_mask = pd.notna(log_df_plot[neural_total_col])
            if valid_mask.any():
                fig.add_trace(go.Scatter(
                    x=log_df_plot.loc[valid_mask, time_col],
                    y=log_df_plot.loc[valid_mask, neural_total_col],
                    name='神经网络预测总气动力',
                    mode='lines',
                    line=dict(color='#2E86AB', width=2)
                ))
        
        fig.update_layout(
            title=f'总力{label}方向对比（neural_f_total vs real_fa_total）',
            xaxis_title='时间 (s)',
            yaxis_title=f'总力{label} (N)',
            height=400,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        fa_total_figs.append(fig)
    
    return fa_figs, fa_total_figs


def main():
    st.set_page_config(
        page_title="导航融合实验可视化",
        page_icon="📊",
        layout="wide"
    )
    
    # 初始化session state
    if 'selected_run_name' not in st.session_state:
        st.session_state['selected_run_name'] = None
    if 'jump_to_tab' not in st.session_state:
        st.session_state['jump_to_tab'] = None
    
    st.title("📊 导航融合实验可视化")
    
    # 侧边栏：配置
    st.sidebar.header("配置")
    tracking_uri = st.sidebar.text_input("MLflow跟踪URI", value="./mlruns")
    project_root = st.sidebar.text_input("项目根目录", value=".")
    
    # 加载数据
    if st.sidebar.button("刷新数据"):
        st.cache_data.clear()
    
    @st.cache_data
    def load_data():
        return load_mlflow_experiments(tracking_uri)
    
    df = load_data()
    
    if df.empty:
        st.warning("没有找到实验数据。请确保MLflow跟踪URI正确，并且已有实验记录。")
        return
    
    # 筛选选项
    st.sidebar.header("筛选")
    
    # 大任务文件夹筛选（优先显示）
    if 'param_task_batch_folder' in df.columns:
        task_batch_folders = df['param_task_batch_folder'].dropna().unique()
        if len(task_batch_folders) > 0:
            # 加载备注，用于显示
            notes = load_task_batch_notes(project_root)
            
            # 创建带备注的选项列表
            folder_options = []
            for folder in sorted(task_batch_folders, reverse=True):
                note = notes.get(folder, "")
                if note:
                    folder_options.append(f"{folder} 📝 {note}")
                else:
                    folder_options.append(folder)
            
            selected_task_batches_with_notes = st.sidebar.multiselect(
                "选择大任务文件夹",
                options=folder_options,
                default=[],  # 默认不选择任何文件夹，让用户自己选择
                help="带📝标记的文件夹有备注信息。请至少选择一个文件夹以查看数据。"
            )
            
            # 提取实际的文件夹名称（去掉备注部分）
            selected_task_batches = []
            for item in selected_task_batches_with_notes:
                # 如果包含"📝"，提取前面的文件夹名
                if "📝" in item:
                    folder_name = item.split("📝")[0].strip()
                else:
                    folder_name = item
                selected_task_batches.append(folder_name)
            
            if len(selected_task_batches) > 0:
                df = df[df['param_task_batch_folder'].isin(selected_task_batches)]
                st.sidebar.info(f"已选择 {len(selected_task_batches)} 个大任务文件夹")
            else:
                # 如果没有选择任何文件夹，显示提示并清空数据
                st.sidebar.warning("⚠️ 请至少选择一个文件夹以查看数据")
                df = pd.DataFrame()  # 清空数据框
    
    # 大任务文件夹管理
    st.sidebar.header("📁 大任务文件夹管理")
    
    # 获取所有大任务文件夹（从文件系统）
    navigation_logs_dir = os.path.join(project_root, "navigation_logs")
    if os.path.exists(navigation_logs_dir):
        task_batch_dirs = [d for d in os.listdir(navigation_logs_dir) 
                          if os.path.isdir(os.path.join(navigation_logs_dir, d)) 
                          and d.startswith("task_batch_")]
        task_batch_dirs = sorted(task_batch_dirs, reverse=True)
        
        if len(task_batch_dirs) > 0:
            # 加载备注
            notes = load_task_batch_notes(project_root)
            
            # 创建标签页：单个管理和批量删除
            tab_single, tab_batch = st.sidebar.tabs(["📝 单个管理", "🗑️ 批量删除"])
            
            with tab_single:
                # 选择要管理的文件夹
                selected_manage_folder = st.selectbox(
                    "选择要管理的文件夹",
                    options=task_batch_dirs,
                    index=0 if task_batch_dirs else None,
                    key="manage_task_batch_folder"
                )
                
                if selected_manage_folder:
                    # 重新加载备注（确保显示最新）
                    notes = load_task_batch_notes(project_root)
                    current_note = notes.get(selected_manage_folder, "")
                    
                    st.markdown("**当前备注:**")
                    if current_note:
                        st.info(current_note)
                    else:
                        st.info("（无备注）")
                    
                    # 备注输入框 - 使用动态key确保每次文件夹切换时都重新加载
                    note_input_key = f"note_input_{selected_manage_folder}"
                    new_note = st.text_area(
                        "编辑备注",
                        value=current_note,  # 直接使用当前备注值
                        height=100,
                        key=note_input_key,
                        help="输入备注信息，用于标识这个任务批次的内容"
                    )
                    
                    # 保存备注按钮
                    if st.button("💾 保存备注", key=f"save_note_{selected_manage_folder}"):
                        if save_task_batch_note(selected_manage_folder, new_note, project_root):
                            st.success("备注已保存！")
                            st.cache_data.clear()  # 清除缓存，刷新显示
                            # 清除该输入框的session_state，强制重新加载
                            if note_input_key in st.session_state:
                                del st.session_state[note_input_key]
                            rerun_app()
                    
                    # 删除文件夹功能
                    delete_key = f"delete_confirm_{selected_manage_folder}"
                    if delete_key not in st.session_state:
                        st.session_state[delete_key] = False
                    
                    if not st.session_state[delete_key]:
                        if st.button("🗑️ 删除文件夹", key=f"delete_{selected_manage_folder}", type="secondary"):
                            st.session_state[delete_key] = True
                            rerun_app()
                    else:
                        st.warning(f"⚠️ 确定要删除文件夹 '{selected_manage_folder}' 吗？")
                        st.markdown("**此操作不可恢复！**")
                        col_confirm, col_cancel = st.columns(2)
                        with col_confirm:
                            if st.button("✅ 确认删除", key=f"confirm_delete_{selected_manage_folder}", type="primary"):
                                if delete_task_batch_folder(selected_manage_folder, project_root, tracking_uri):
                                    st.session_state[delete_key] = False
                                    st.success("文件夹及MLflow记录已删除！")
                                    st.cache_data.clear()  # 清除缓存，刷新显示
                                    rerun_app()
                        with col_cancel:
                            if st.button("❌ 取消", key=f"cancel_delete_{selected_manage_folder}"):
                                st.session_state[delete_key] = False
                                rerun_app()
                    
                    # 显示文件夹信息
                    folder_path = os.path.join(navigation_logs_dir, selected_manage_folder)
                    if os.path.exists(folder_path):
                        file_count = len([f for f in os.listdir(folder_path) 
                                         if os.path.isfile(os.path.join(folder_path, f))])
                        folder_size = sum(os.path.getsize(os.path.join(folder_path, f))
                                         for f in os.listdir(folder_path)
                                         if os.path.isfile(os.path.join(folder_path, f)))
                        folder_size_mb = folder_size / (1024 * 1024)
                        
                        st.markdown("**文件夹信息:**")
                        st.text(f"文件数量: {file_count}")
                        st.text(f"文件夹大小: {folder_size_mb:.2f} MB")
            
            with tab_batch:
                st.markdown("**批量删除文件夹**")
                st.markdown("选择要删除的文件夹，然后点击批量删除按钮。")
                
                # 创建带备注的选项列表
                batch_folder_options = []
                for folder in task_batch_dirs:
                    note = notes.get(folder, "")
                    if note:
                        batch_folder_options.append(f"{folder} 📝 {note}")
                    else:
                        batch_folder_options.append(folder)
                
                selected_batch_folders_with_notes = st.multiselect(
                    "选择要删除的文件夹",
                    options=batch_folder_options,
                    default=[],
                    help="⚠️ 删除操作不可恢复，请谨慎选择"
                )
                
                # 提取实际的文件夹名称
                selected_batch_folders = []
                for item in selected_batch_folders_with_notes:
                    if "📝" in item:
                        folder_name = item.split("📝")[0].strip()
                    else:
                        folder_name = item
                    selected_batch_folders.append(folder_name)
                
                if len(selected_batch_folders) > 0:
                    st.warning(f"⚠️ 将删除 {len(selected_batch_folders)} 个文件夹，此操作不可恢复！")
                    
                    # 显示将要删除的文件夹列表
                    with st.expander("查看将要删除的文件夹列表", expanded=False):
                        for folder in selected_batch_folders:
                            note = notes.get(folder, "")
                            if note:
                                st.text(f"• {folder} 📝 {note}")
                            else:
                                st.text(f"• {folder}")
                    
                    # 批量删除确认
                    batch_delete_key = "batch_delete_confirm"
                    if batch_delete_key not in st.session_state:
                        st.session_state[batch_delete_key] = False
                    
                    if not st.session_state[batch_delete_key]:
                        if st.button("🗑️ 批量删除", key="batch_delete_btn", type="primary"):
                            st.session_state[batch_delete_key] = True
                            rerun_app()
                    else:
                        st.error("⚠️ 最后确认：确定要删除这些文件夹吗？")
                        col_confirm_batch, col_cancel_batch = st.columns(2)
                        with col_confirm_batch:
                            if st.button("✅ 确认批量删除", key="confirm_batch_delete", type="primary"):
                                success_count = 0
                                failed_folders = []
                                for folder in selected_batch_folders:
                                    if delete_task_batch_folder(folder, project_root, tracking_uri):
                                        success_count += 1
                                    else:
                                        failed_folders.append(folder)
                                
                                if success_count > 0:
                                    st.success(f"成功删除 {success_count} 个文件夹及对应的MLflow记录！")
                                if failed_folders:
                                    st.error(f"删除失败: {', '.join(failed_folders)}")
                                
                                st.session_state[batch_delete_key] = False
                                st.cache_data.clear()
                                rerun_app()
                        with col_cancel_batch:
                            if st.button("❌ 取消", key="cancel_batch_delete"):
                                st.session_state[batch_delete_key] = False
                                rerun_app()
                else:
                    st.info("请选择要删除的文件夹")
        else:
            st.sidebar.info("没有找到大任务文件夹")
    
    # 实验名称筛选
    if 'experiment_name' in df.columns:
        experiment_names = df['experiment_name'].unique()
        selected_experiments = st.sidebar.multiselect(
            "选择实验",
            options=experiment_names,
            default=list(experiment_names)
        )
        df = df[df['experiment_name'].isin(selected_experiments)]
    
    # 参数筛选
    param_columns = [col for col in df.columns if col.startswith('param_')]
    if param_columns:
        st.sidebar.subheader("参数筛选")
        for col in param_columns[:5]:  # 只显示前5个参数
            param_name = col.replace('param_', '')
            unique_values = df[col].dropna().unique()
            if len(unique_values) > 0 and len(unique_values) < 20:  # 只显示选项少于20个的参数
                selected = st.sidebar.multiselect(
                    param_name,
                    options=unique_values,
                    default=list(unique_values)
                )
                df = df[df[col].isin(selected)]
    
    # 主内容区域
    tab1, tab2, tab3 = st.tabs(["📋 任务汇总表", "📈 参数分析", "🔍 实验详情"])
    
    # 如果设置了跳转，显示提示信息
    jump_to_tab = st.session_state.get('jump_to_tab')
    if jump_to_tab:
        if jump_to_tab == 'tab3':
            st.info("💡 已选择实验，请切换到 **🔍 实验详情** 标签页查看详细信息")
    
    with tab1:
        st.header("📋 任务汇总表")
        st.markdown("显示所有实验的参数和关键指标。勾选任务后点击跳转按钮查看详情。")
        
        # 创建汇总表
        summary_df = create_summary_table(df)
        
        if not summary_df.empty:
            # 排序
            summary_df_sorted = summary_df.sort_values('实验时间', ascending=False).reset_index(drop=True)
            
            # 初始化选择状态
            if 'selected_run_index' not in st.session_state:
                st.session_state['selected_run_index'] = None
            
            # 准备显示的数据（隐藏内部列）
            display_df = summary_df_sorted.drop(columns=['run_id', 'run_name_key'], errors='ignore').copy()
            
            # 使用下拉框选择任务
            run_names = summary_df_sorted['run_name_key'].tolist()
            run_display_names = []
            for idx, row in summary_df_sorted.iterrows():
                csv_file = row.get('CSV文件', 'Unknown')
                filter_type = row.get('滤波器类型', 'Unknown')
                lambda1 = row.get('lambda1', 'N/A')
                display_name = f"{csv_file} | {filter_type} | λ={lambda1}"
                run_display_names.append(display_name)
            
            # 如果从其他地方跳转过来，使用session state中的选择
            default_run = st.session_state.get('selected_run_name', None)
            if default_run and default_run in run_names:
                default_index = run_names.index(default_run)
            else:
                default_index = 0
            
            # 下拉框选择任务
            col_select, col_btn = st.columns([3, 1])
            with col_select:
                selected_display = st.selectbox(
                    "选择要查看的任务",
                    options=run_display_names,
                    index=default_index if default_index < len(run_display_names) else 0,
                    key="summary_run_selector"
                )
            
            selected_index = run_display_names.index(selected_display) if selected_display in run_display_names else 0
            selected_run_name = run_names[selected_index] if selected_index < len(run_names) else run_names[0]
            st.session_state['selected_run_name'] = selected_run_name
            st.session_state['selected_run_index'] = selected_index
            
            with col_btn:
                st.markdown("<br>", unsafe_allow_html=True)  # 垂直对齐
                if st.button("🔍 查看实验详情", key="jump_to_details_btn", use_container_width=True):
                    if st.session_state.get('selected_run_name'):
                        st.session_state['jump_to_tab'] = 'tab3'
                        rerun_app()
            
            st.markdown("---")
            
            # 显示汇总表格（带复选框）
            st.markdown("### 任务汇总表")
            
            # 改善率颜色编码说明
            st.caption("💡 改善率说明：正值表示UKF相比纯惯导有改善，负值表示性能下降。改善率越高，颜色越绿。")
            
            # 获取所有列名（除了内部列）
            table_columns = [col for col in display_df.columns if col not in ['run_id', 'run_name_key']]
            
            # 改善率说明（速度改善率已显示在CSV文件列右边）
            if '速度改善率_总(%)' in display_df.columns:
                st.caption("💡 改善率说明：正值表示UKF相比纯惯导有改善，负值表示性能下降。改善率越高越好。")
            
            # 初始化复选框列
            checkbox_key_state = f"checkbox_df_{len(display_df)}"
            if checkbox_key_state not in st.session_state:
                # 初始化复选框列，默认选中下拉框选择的行
                checkbox_col = [False] * len(display_df)
                if selected_index < len(display_df):
                    checkbox_col[selected_index] = True
                st.session_state[checkbox_key_state] = checkbox_col
            
            # 准备可编辑的DataFrame
            editable_df = display_df.copy()
            checkbox_col = st.session_state.get(checkbox_key_state, [False] * len(display_df))
            
            # 确保长度匹配
            if len(checkbox_col) != len(display_df):
                checkbox_col = [False] * len(display_df)
                if selected_index < len(display_df):
                    checkbox_col[selected_index] = True
                st.session_state[checkbox_key_state] = checkbox_col
            
            # 如果下拉框选择改变，更新复选框状态
            if selected_index < len(display_df):
                # 清除所有选择
                checkbox_col = [False] * len(display_df)
                # 选中下拉框选择的行
                checkbox_col[selected_index] = True
                st.session_state[checkbox_key_state] = checkbox_col
            
            # 添加复选框列
            editable_df.insert(0, '选择', checkbox_col)
            
            # 配置列：选择列为复选框，其他列不可编辑
            column_config = {
                '选择': st.column_config.CheckboxColumn(
                    label="选择",
                    help="选择要查看的实验",
                    default=False,
                    width="small"
                )
            }
            
            # 禁用其他列的编辑
            disabled_columns = [col for col in editable_df.columns if col != '选择']
            
            # 使用st.data_editor显示可编辑表格（复选框在表格内）
            edited_df = st.data_editor(
                editable_df,
                column_config=column_config,
                disabled=disabled_columns,
                use_container_width=True,
                height=500,
                hide_index=True,
                key="summary_table_editor"
            )
            
            # 保存复选框状态
            selected_indices = []
            if '选择' in edited_df.columns:
                checkbox_values = edited_df['选择'].tolist()
                st.session_state[checkbox_key_state] = checkbox_values
                
                # 获取选中的行索引
                selected_indices = edited_df[edited_df['选择']].index.tolist()
                
                # 如果选中了行，更新选中的任务
                if selected_indices:
                    # 使用第一个选中的行
                    selected_idx = selected_indices[0]
                    if selected_idx < len(run_names):
                        st.session_state['selected_run_name'] = run_names[selected_idx]
                        st.session_state['selected_run_index'] = selected_idx
            
            st.markdown("---")
            
            # 跳转按钮（只有在只有一个复选框被选中时才生效）
            num_selected = len(selected_indices)
            col_btn_jump, col_btn_download = st.columns([1, 1])
            
            with col_btn_jump:
                if num_selected == 1:
                    # 只有一个选中，可以跳转
                    selected_idx = selected_indices[0]
                    if st.button("🔍 跳转到选中任务详情", key="jump_to_selected_task_btn", use_container_width=True):
                        if selected_idx < len(run_names):
                            st.session_state['selected_run_name'] = run_names[selected_idx]
                            st.session_state['selected_run_index'] = selected_idx
                            st.session_state['jump_to_tab'] = 'tab3'
                            rerun_app()
                else:
                    # 没有选中或选中多个，按钮禁用或显示提示
                    if num_selected == 0:
                        st.button("🔍 跳转到选中任务详情", key="jump_to_selected_task_btn", 
                                 use_container_width=True, disabled=True, 
                                 help="请先选择一个任务")
                    else:
                        st.button("🔍 跳转到选中任务详情", key="jump_to_selected_task_btn", 
                                 use_container_width=True, disabled=True, 
                                 help=f"只能选择一个任务（当前选中 {num_selected} 个）")
            
            with col_btn_download:
                # 下载按钮
                csv = display_df.to_csv(index=False)
                st.download_button(
                    label="📥 下载汇总表 (CSV)",
                    data=csv,
                    file_name=f"experiment_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        else:
            st.warning("无法创建汇总表")
        
        # ========== 参数影响分析部分（放在任务汇总表下面）==========
        st.markdown("---")
        st.subheader("📊 参数影响分析")
        
        # 检查是否有汇总表数据
        if 'summary_df' in locals() and summary_df is not None and len(summary_df) > 0:
            # 选择要分析的参数
            available_params = []
            param_display_names = {
                'Rk': 'Rk (UKF观测噪声)',
                'lambda1': 'lambda1 (自适应参数)',
                'Q': 'Q (过程噪声)',
                'R': 'R (观测噪声)',
                'numPar': 'numPar (粒子数)'
            }
            
            # 检查哪些参数有数据
            for param in ['Rk', 'lambda1', 'Q', 'R', 'numPar']:
                param_col = param
                if param_col in summary_df.columns:
                    # 检查是否有非N/A的值
                    non_na_values = summary_df[param_col][summary_df[param_col] != 'N/A']
                    if len(non_na_values) > 0:
                        available_params.append(param)
            
            if available_params:
                col_param, col_metric = st.columns([1, 1])
                
                with col_param:
                    selected_param = st.selectbox(
                        "选择要分析的参数",
                        options=available_params,
                        format_func=lambda x: param_display_names.get(x, x),
                        key="param_analysis_param"
                    )
                
                with col_metric:
                    # 选择要分析的指标类型
                    metric_type = st.selectbox(
                        "选择指标类型",
                        options=['速度RMSE', '位置RMSE', '气动力RMSE', '速度+位置', '全部'],
                        key="param_analysis_metric_type"
                    )
                
                # 按参数值分组数据
                param_col = selected_param
                param_values = summary_df[param_col].unique()
                param_values = [v for v in param_values if v != 'N/A']
                
                if len(param_values) > 0:
                    # 准备数据：按参数值分组，每个参数值下有多个CSV文件的结果
                    analysis_data = []
                    for param_val in param_values:
                        # 获取该参数值下的所有数据
                        param_data = summary_df[summary_df[param_col] == param_val]
                        
                        for _, row in param_data.iterrows():
                            csv_file = row.get('CSV文件', 'Unknown')
                            
                            # 转换为数值
                            def safe_float(val):
                                if val == 'N/A' or pd.isna(val):
                                    return None
                                try:
                                    return float(val)
                                except:
                                    return None
                            
                            # 提取速度RMSE值（xyz方向和总）
                            ukf_vel_east = row.get('UKF速度RMSE_东', 'N/A')
                            ukf_vel_north = row.get('UKF速度RMSE_北', 'N/A')
                            ukf_vel_up = row.get('UKF速度RMSE_天', 'N/A')
                            ukf_vel_total = row.get('UKF速度RMSE_总', 'N/A')
                            pure_vel_east = row.get('纯惯导速度RMSE_东', 'N/A')
                            pure_vel_north = row.get('纯惯导速度RMSE_北', 'N/A')
                            pure_vel_up = row.get('纯惯导速度RMSE_天', 'N/A')
                            pure_vel_total = row.get('纯惯导速度RMSE_总', 'N/A')
                            
                            # 提取位置RMSE值（xyz方向和总）
                            ukf_pos_east = row.get('UKF位置RMSE_东', 'N/A')
                            ukf_pos_north = row.get('UKF位置RMSE_北', 'N/A')
                            ukf_pos_up = row.get('UKF位置RMSE_天', 'N/A')
                            ukf_pos_total = row.get('UKF位置RMSE_总', 'N/A')
                            pure_pos_east = row.get('纯惯导位置RMSE_东', 'N/A')
                            pure_pos_north = row.get('纯惯导位置RMSE_北', 'N/A')
                            pure_pos_up = row.get('纯惯导位置RMSE_天', 'N/A')
                            pure_pos_total = row.get('纯惯导位置RMSE_总', 'N/A')
                            
                            # 提取气动力RMSE值（xyz方向和总）
                            fa_rmse_x = row.get('气动力RMSE_x', 'N/A')
                            fa_rmse_y = row.get('气动力RMSE_y', 'N/A')
                            fa_rmse_z = row.get('气动力RMSE_z', 'N/A')
                            fa_rmse_total = row.get('气动力RMSE_总', 'N/A')
                            
                            analysis_data.append({
                                '参数值': str(param_val),
                                'CSV文件': csv_file,
                                # 速度RMSE
                                'UKF速度RMSE_东': safe_float(ukf_vel_east),
                                'UKF速度RMSE_北': safe_float(ukf_vel_north),
                                'UKF速度RMSE_天': safe_float(ukf_vel_up),
                                'UKF速度RMSE_总': safe_float(ukf_vel_total),
                                '纯惯导速度RMSE_东': safe_float(pure_vel_east),
                                '纯惯导速度RMSE_北': safe_float(pure_vel_north),
                                '纯惯导速度RMSE_天': safe_float(pure_vel_up),
                                '纯惯导速度RMSE_总': safe_float(pure_vel_total),
                                # 位置RMSE
                                'UKF位置RMSE_东': safe_float(ukf_pos_east),
                                'UKF位置RMSE_北': safe_float(ukf_pos_north),
                                'UKF位置RMSE_天': safe_float(ukf_pos_up),
                                'UKF位置RMSE_总': safe_float(ukf_pos_total),
                                '纯惯导位置RMSE_东': safe_float(pure_pos_east),
                                '纯惯导位置RMSE_北': safe_float(pure_pos_north),
                                '纯惯导位置RMSE_天': safe_float(pure_pos_up),
                                '纯惯导位置RMSE_总': safe_float(pure_pos_total),
                                # 气动力RMSE
                                '气动力RMSE_x': safe_float(fa_rmse_x),
                                '气动力RMSE_y': safe_float(fa_rmse_y),
                                '气动力RMSE_z': safe_float(fa_rmse_z),
                                '气动力RMSE_总': safe_float(fa_rmse_total),
                            })
                    
                    analysis_df = pd.DataFrame(analysis_data)
                    
                    # 可视化选项
                    viz_type = st.radio(
                        "选择可视化方式",
                        options=['柱状图', '折线图', '热力图', '组合视图'],
                        horizontal=True,
                        key="param_analysis_viz_type"
                    )
                    
                    # 辅助函数：绘制单个方向的图表
                    def plot_metric_direction(metric_name, direction_key, direction_label, ukf_col, pure_col=None, unit=''):
                        """绘制单个方向的RMSE图表"""
                        st.markdown(f"##### {metric_name} - {direction_label}方向")
                        
                        if viz_type == '柱状图':
                            fig_data = []
                            for _, row in analysis_df.iterrows():
                                if row[ukf_col] is not None:
                                    fig_data.append({
                                        '参数值': row['参数值'],
                                        'CSV文件': row['CSV文件'],
                                        'UKF': row[ukf_col],
                                        '纯惯导': row[pure_col] if pure_col and row[pure_col] is not None else None
                                    })
                            
                            if fig_data:
                                fig_df = pd.DataFrame(fig_data)
                                # 尝试按参数值排序
                                try:
                                    fig_df['参数值_数值'] = fig_df['参数值'].astype(float)
                                    fig_df = fig_df.sort_values('参数值_数值')
                                except:
                                    pass
                                
                                if pure_col:
                                    fig = px.bar(
                                        fig_df,
                                        x='参数值',
                                        y=['UKF', '纯惯导'],
                                        color='CSV文件',
                                        title=f'{param_display_names.get(selected_param, selected_param)} 对{metric_name} {direction_label}方向的影响',
                                        barmode='group',
                                        labels={'value': f'RMSE ({unit})', '参数值': param_display_names.get(selected_param, selected_param)}
                                    )
                                else:
                                    fig = px.bar(
                                        fig_df,
                                        x='参数值',
                                        y='UKF',
                                        color='CSV文件',
                                        title=f'{param_display_names.get(selected_param, selected_param)} 对{metric_name} {direction_label}方向的影响',
                                        barmode='group',
                                        labels={'value': f'RMSE ({unit})', '参数值': param_display_names.get(selected_param, selected_param)}
                                    )
                                fig.update_layout(height=400)
                                st.plotly_chart(fig, use_container_width=True)
                        
                        elif viz_type == '折线图':
                            fig_data = []
                            for _, row in analysis_df.iterrows():
                                if row[ukf_col] is not None:
                                    fig_data.append({
                                        '参数值': row['参数值'],
                                        'CSV文件': row['CSV文件'],
                                        'UKF': row[ukf_col],
                                        '纯惯导': row[pure_col] if pure_col and row[pure_col] is not None else None
                                    })
                            
                            if fig_data:
                                fig_df = pd.DataFrame(fig_data)
                                try:
                                    fig_df['参数值_数值'] = fig_df['参数值'].astype(float)
                                    fig_df = fig_df.sort_values('参数值_数值')
                                except:
                                    pass
                                
                                if pure_col:
                                    fig = px.line(
                                        fig_df,
                                        x='参数值',
                                        y=['UKF', '纯惯导'],
                                        color='CSV文件',
                                        title=f'{param_display_names.get(selected_param, selected_param)} 对{metric_name} {direction_label}方向的影响',
                                        markers=True,
                                        labels={'value': f'RMSE ({unit})', '参数值': param_display_names.get(selected_param, selected_param)}
                                    )
                                else:
                                    fig = px.line(
                                        fig_df,
                                        x='参数值',
                                        y='UKF',
                                        color='CSV文件',
                                        title=f'{param_display_names.get(selected_param, selected_param)} 对{metric_name} {direction_label}方向的影响',
                                        markers=True,
                                        labels={'value': f'RMSE ({unit})', '参数值': param_display_names.get(selected_param, selected_param)}
                                    )
                                fig.update_layout(height=400)
                                st.plotly_chart(fig, use_container_width=True)
                        
                        elif viz_type == '热力图':
                            pivot_data = analysis_df.pivot_table(
                                index='CSV文件',
                                columns='参数值',
                                values=ukf_col,
                                aggfunc='mean'
                            )
                            
                            if not pivot_data.empty:
                                fig = px.imshow(
                                    pivot_data.values,
                                    x=pivot_data.columns,
                                    y=pivot_data.index,
                                    labels=dict(x=param_display_names.get(selected_param, selected_param), 
                                              y='CSV文件', 
                                              color=f'RMSE ({unit})'),
                                    title=f'{param_display_names.get(selected_param, selected_param)} 对{metric_name} {direction_label}方向的影响（热力图）',
                                    color_continuous_scale='Viridis',
                                    aspect='auto'
                                )
                                fig.update_layout(height=400)
                                st.plotly_chart(fig, use_container_width=True)
                        
                        elif viz_type == '组合视图':
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                fig_data = []
                                for _, row in analysis_df.iterrows():
                                    if row[ukf_col] is not None:
                                        fig_data.append({
                                            '参数值': row['参数值'],
                                            'CSV文件': row['CSV文件'],
                                            'UKF': row[ukf_col]
                                        })
                                
                                if fig_data:
                                    fig_df = pd.DataFrame(fig_data)
                                    try:
                                        fig_df['参数值_数值'] = fig_df['参数值'].astype(float)
                                        fig_df = fig_df.sort_values('参数值_数值')
                                    except:
                                        pass
                                    
                                    fig = px.bar(
                                        fig_df,
                                        x='参数值',
                                        y='UKF',
                                        color='CSV文件',
                                        title=f'UKF {metric_name} {direction_label}方向（柱状图）',
                                        barmode='group',
                                        labels={'value': f'RMSE ({unit})'}
                                    )
                                    fig.update_layout(height=350)
                                    st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                fig_data = []
                                for _, row in analysis_df.iterrows():
                                    if row[ukf_col] is not None:
                                        fig_data.append({
                                            '参数值': row['参数值'],
                                            'CSV文件': row['CSV文件'],
                                            'UKF': row[ukf_col]
                                        })
                                
                                if fig_data:
                                    fig_df = pd.DataFrame(fig_data)
                                    try:
                                        fig_df['参数值_数值'] = fig_df['参数值'].astype(float)
                                        fig_df = fig_df.sort_values('参数值_数值')
                                    except:
                                        pass
                                    
                                    fig = px.line(
                                        fig_df,
                                        x='参数值',
                                        y='UKF',
                                        color='CSV文件',
                                        title=f'UKF {metric_name} {direction_label}方向（折线图）',
                                        markers=True,
                                        labels={'value': f'RMSE ({unit})'}
                                    )
                                    fig.update_layout(height=350)
                                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 根据指标类型显示相应的图表
                    if metric_type == '速度RMSE' or metric_type == '速度+位置' or metric_type == '全部':
                        st.markdown("#### 速度RMSE对比")
                        # 显示xyz方向和总RMSE
                        plot_metric_direction('速度', '东', '东', 'UKF速度RMSE_东', '纯惯导速度RMSE_东', 'm/s')
                        plot_metric_direction('速度', '北', '北', 'UKF速度RMSE_北', '纯惯导速度RMSE_北', 'm/s')
                        plot_metric_direction('速度', '天', '天', 'UKF速度RMSE_天', '纯惯导速度RMSE_天', 'm/s')
                        plot_metric_direction('速度', '总', '总', 'UKF速度RMSE_总', '纯惯导速度RMSE_总', 'm/s')
                    
                    if metric_type == '位置RMSE' or metric_type == '速度+位置' or metric_type == '全部':
                        if metric_type == '速度+位置' or metric_type == '全部':
                            st.markdown("---")
                        st.markdown("#### 位置RMSE对比")
                        # 显示xyz方向和总RMSE
                        plot_metric_direction('位置', '东', '东', 'UKF位置RMSE_东', '纯惯导位置RMSE_东', 'm')
                        plot_metric_direction('位置', '北', '北', 'UKF位置RMSE_北', '纯惯导位置RMSE_北', 'm')
                        plot_metric_direction('位置', '天', '天', 'UKF位置RMSE_天', '纯惯导位置RMSE_天', 'm')
                        plot_metric_direction('位置', '总', '总', 'UKF位置RMSE_总', '纯惯导位置RMSE_总', 'm')
                    
                    if metric_type == '气动力RMSE' or metric_type == '全部':
                        if metric_type == '全部':
                            st.markdown("---")
                        st.markdown("#### 气动力RMSE对比")
                        # 显示xyz方向和总RMSE（气动力没有纯惯导对比）
                        plot_metric_direction('气动力', 'x', 'X', '气动力RMSE_x', None, 'N')
                        plot_metric_direction('气动力', 'y', 'Y', '气动力RMSE_y', None, 'N')
                        plot_metric_direction('气动力', 'z', 'Z', '气动力RMSE_z', None, 'N')
                        plot_metric_direction('气动力', '总', '总', '气动力RMSE_总', None, 'N')
                else:
                    st.warning(f"参数 {param_display_names.get(selected_param, selected_param)} 没有有效的数据值")
            else:
                st.info("没有可用的参数数据进行分析")
        else:
            st.info("请先加载实验数据")
    
    with tab2:
        st.header("参数分析")
        
        # 选择参数和指标进行相关性分析
        param_cols = [col for col in df.columns if col.startswith('param_')]
        metric_cols = [col for col in df.columns if col.startswith('metric_')]
        
        if param_cols and metric_cols:
            col1, col2 = st.columns(2)
            
            with col1:
                selected_param = st.selectbox("选择参数", options=param_cols)
            
            with col2:
                selected_metric = st.selectbox("选择指标", options=metric_cols)
            
            if selected_param and selected_metric:
                # 散点图
                scatter_data = df[[selected_param, selected_metric]].dropna()
                if not scatter_data.empty:
                    fig = px.scatter(
                        scatter_data,
                        x=selected_param,
                        y=selected_metric,
                        title=f'{selected_param.replace("param_", "")} vs {selected_metric.replace("metric_", "")}'
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("🔍 实验详情")
        
        if not df.empty:
            # 如果从汇总表跳转过来，使用session state中的选择
            default_run = st.session_state.get('selected_run_name', None)
            run_options = df['run_name'].unique() if 'run_name' in df.columns else df.index.tolist()
            
            if default_run and default_run in run_options:
                default_index = list(run_options).index(default_run)
            else:
                default_index = 0
            
            selected_run = st.selectbox(
                "选择实验运行",
                options=run_options,
                index=default_index if default_index < len(run_options) else 0
            )
            
            selected_row = df[df['run_name'] == selected_run].iloc[0] if 'run_name' in df.columns else df.iloc[selected_run]
            
            # RMSE对比部分
            st.subheader("📊 RMSE对比（UKF vs 纯惯导）")
            
            # 提取RMSE指标
            rmse_metrics = {
                '速度RMSE': {
                    'UKF': {
                        '东向': selected_row.get('metric_ukf_vel_rmse_east', None),
                        '北向': selected_row.get('metric_ukf_vel_rmse_north', None),
                        '天向': selected_row.get('metric_ukf_vel_rmse_up', None),
                        '总RMSE': selected_row.get('metric_ukf_vel_rmse_total', None)
                    },
                    '纯惯导': {
                        '东向': selected_row.get('metric_pure_ins_vel_rmse_east', None),
                        '北向': selected_row.get('metric_pure_ins_vel_rmse_north', None),
                        '天向': selected_row.get('metric_pure_ins_vel_rmse_up', None),
                        '总RMSE': selected_row.get('metric_pure_ins_vel_rmse_total', None)
                    }
                },
                '位置RMSE': {
                    'UKF': {
                        '东向': selected_row.get('metric_ukf_pos_rmse_east', None),
                        '北向': selected_row.get('metric_ukf_pos_rmse_north', None),
                        '天向': selected_row.get('metric_ukf_pos_rmse_up', None),
                        '总RMSE': selected_row.get('metric_ukf_pos_rmse_total', None)
                    },
                    '纯惯导': {
                        '东向': selected_row.get('metric_pure_ins_pos_rmse_east', None),
                        '北向': selected_row.get('metric_pure_ins_pos_rmse_north', None),
                        '天向': selected_row.get('metric_pure_ins_pos_rmse_up', None),
                        '总RMSE': selected_row.get('metric_pure_ins_pos_rmse_total', None)
                    }
                }
            }
            
            # 创建对比表格
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 速度RMSE对比")
                vel_data = []
                for direction in ['东向', '北向', '天向', '总RMSE']:
                    ukf_val = rmse_metrics['速度RMSE']['UKF'][direction]
                    pure_val = rmse_metrics['速度RMSE']['纯惯导'][direction]
                    if ukf_val is not None and pure_val is not None:
                        improvement = ((pure_val - ukf_val) / pure_val * 100) if pure_val != 0 else 0
                        vel_data.append({
                            '方向': direction,
                            'UKF (m/s)': f"{ukf_val:.4f}",
                            '纯惯导 (m/s)': f"{pure_val:.4f}",
                            '改善 (%)': f"{improvement:.2f}%"
                        })
                
                if vel_data:
                    vel_df = pd.DataFrame(vel_data)
                    st.dataframe(vel_df, use_container_width=True, hide_index=True)
                    
                    # 速度RMSE对比图
                    fig_vel = go.Figure()
                    directions = ['东向', '北向', '天向', '总RMSE']
                    ukf_vals = [rmse_metrics['速度RMSE']['UKF'][d] for d in directions if rmse_metrics['速度RMSE']['UKF'][d] is not None]
                    pure_vals = [rmse_metrics['速度RMSE']['纯惯导'][d] for d in directions if rmse_metrics['速度RMSE']['纯惯导'][d] is not None]
                    valid_directions = [d for d in directions if rmse_metrics['速度RMSE']['UKF'][d] is not None and rmse_metrics['速度RMSE']['纯惯导'][d] is not None]
                    
                    if ukf_vals and pure_vals:
                        fig_vel.add_trace(go.Bar(
                            x=valid_directions,
                            y=ukf_vals,
                            name='UKF',
                            marker_color='#2E86AB'
                        ))
                        fig_vel.add_trace(go.Bar(
                            x=valid_directions,
                            y=pure_vals,
                            name='纯惯导',
                            marker_color='#F24236'
                        ))
                        fig_vel.update_layout(
                            title='速度RMSE对比',
                            xaxis_title='方向',
                            yaxis_title='RMSE (m/s)',
                            barmode='group',
                            height=400,
                            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                        )
                        st.plotly_chart(fig_vel, use_container_width=True)
                else:
                    st.info("速度RMSE数据不可用")
            
            with col2:
                st.markdown("### 位置RMSE对比")
                pos_data = []
                for direction in ['东向', '北向', '天向', '总RMSE']:
                    ukf_val = rmse_metrics['位置RMSE']['UKF'][direction]
                    pure_val = rmse_metrics['位置RMSE']['纯惯导'][direction]
                    if ukf_val is not None and pure_val is not None:
                        improvement = ((pure_val - ukf_val) / pure_val * 100) if pure_val != 0 else 0
                        pos_data.append({
                            '方向': direction,
                            'UKF (m)': f"{ukf_val:.4f}",
                            '纯惯导 (m)': f"{pure_val:.4f}",
                            '改善 (%)': f"{improvement:.2f}%"
                        })
                
                if pos_data:
                    pos_df = pd.DataFrame(pos_data)
                    st.dataframe(pos_df, use_container_width=True, hide_index=True)
                    
                    # 位置RMSE对比图
                    fig_pos = go.Figure()
                    directions = ['东向', '北向', '天向', '总RMSE']
                    ukf_vals = [rmse_metrics['位置RMSE']['UKF'][d] for d in directions if rmse_metrics['位置RMSE']['UKF'][d] is not None]
                    pure_vals = [rmse_metrics['位置RMSE']['纯惯导'][d] for d in directions if rmse_metrics['位置RMSE']['纯惯导'][d] is not None]
                    valid_directions = [d for d in directions if rmse_metrics['位置RMSE']['UKF'][d] is not None and rmse_metrics['位置RMSE']['纯惯导'][d] is not None]
                    
                    if ukf_vals and pure_vals:
                        fig_pos.add_trace(go.Bar(
                            x=valid_directions,
                            y=ukf_vals,
                            name='UKF',
                            marker_color='#2E86AB'
                        ))
                        fig_pos.add_trace(go.Bar(
                            x=valid_directions,
                            y=pure_vals,
                            name='纯惯导',
                            marker_color='#F24236'
                        ))
                        fig_pos.update_layout(
                            title='位置RMSE对比',
                            xaxis_title='方向',
                            yaxis_title='RMSE (m)',
                            barmode='group',
                            height=400,
                            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                        )
                        st.plotly_chart(fig_pos, use_container_width=True)
                else:
                    st.info("位置RMSE数据不可用")
            
            # 气动力RMSE显示
            st.markdown("### 气动力RMSE")
            col_fa1, col_fa2 = st.columns(2)
            
            with col_fa1:
                st.markdown("#### 气动力RMSE（neural_f vs real_fa）")
                fa_data = []
                for direction in ['X', 'Y', 'Z', '总RMSE']:
                    metric_key = f'metric_fa_rmse_{direction.lower()}' if direction != '总RMSE' else 'metric_fa_rmse_total'
                    if direction == 'X':
                        metric_key = 'metric_fa_rmse_x'
                    elif direction == 'Y':
                        metric_key = 'metric_fa_rmse_y'
                    elif direction == 'Z':
                        metric_key = 'metric_fa_rmse_z'
                    else:
                        metric_key = 'metric_fa_rmse_total'
                    
                    fa_val = selected_row.get(metric_key, None)
                    if fa_val is not None:
                        fa_data.append({
                            '方向': direction,
                            'RMSE (N)': f"{fa_val:.6f}"
                        })
                
                if fa_data:
                    fa_df = pd.DataFrame(fa_data)
                    st.dataframe(fa_df, use_container_width=True, hide_index=True)
                else:
                    st.info("气动力RMSE数据不可用")
            
            with col_fa2:
                st.markdown("#### 总力RMSE（neural_f_total vs real_fa_total）")
                fa_total_data = []
                for direction in ['X', 'Y', 'Z', '总RMSE']:
                    if direction == 'X':
                        metric_key = 'metric_neural_fa_total_rmse_x'
                    elif direction == 'Y':
                        metric_key = 'metric_neural_fa_total_rmse_y'
                    elif direction == 'Z':
                        metric_key = 'metric_neural_fa_total_rmse_z'
                    else:
                        metric_key = 'metric_neural_fa_total_rmse_total'
                    
                    fa_total_val = selected_row.get(metric_key, None)
                    if fa_total_val is not None:
                        fa_total_data.append({
                            '方向': direction,
                            'RMSE (N)': f"{fa_total_val:.6f}"
                        })
                
                if fa_total_data:
                    fa_total_df = pd.DataFrame(fa_total_data)
                    st.dataframe(fa_total_df, use_container_width=True, hide_index=True)
                else:
                    st.info("总力RMSE数据不可用")
            
            st.markdown("---")
            
            # ========== 时间序列对比部分（放在RMSE下面，参数上面）==========
            st.subheader("📈 时间序列对比")
            st.markdown("显示位置、速度、姿态的时间序列对比图（从日志文件读取）")
            
            # 获取日志文件名和大任务文件夹
            log_file_name = selected_row.get('log_file', None)
            task_batch_folder = selected_row.get('param_task_batch_folder', None)
            
            # 确保类型正确（处理 pandas NaN）
            if log_file_name is not None and pd.notna(log_file_name):
                log_file_name = str(log_file_name)
            else:
                log_file_name = None
            
            if task_batch_folder is not None and pd.notna(task_batch_folder):
                task_batch_folder = str(task_batch_folder)
            else:
                task_batch_folder = None
            
            if log_file_name:
                # 加载日志文件（传入大任务文件夹信息）
                with st.spinner(f"正在加载日志文件: {log_file_name}"):
                    log_df = load_log_file(log_file_name, project_root, task_batch_folder)
                
                if log_df is not None and not log_df.empty:
                    # 绘制时间序列图
                    pos_figs, vel_figs, att_figs = plot_time_series(log_df, selected_run)
                    
                    # 使用标签页分别显示姿态、速度、位置
                    tab_att, tab_vel, tab_pos = st.tabs(["🎯 姿态对比", "⚡ 速度对比", "📍 位置对比"])
                    
                    with tab_att:
                        st.markdown("#### 姿态对比（X、Y、Z方向）")
                        if att_figs:
                            for i, (fig, label) in enumerate(zip(att_figs, ['X', 'Y', 'Z'])):
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("无法生成姿态对比图")
                    
                    with tab_vel:
                        st.markdown("#### 速度对比（东向、北向、天向）")
                        if vel_figs:
                            for i, (fig, label) in enumerate(zip(vel_figs, ['东向', '北向', '天向'])):
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("无法生成速度对比图")
                    
                    with tab_pos:
                        st.markdown("#### 位置对比（东向、北向、天向）")
                        if pos_figs:
                            for i, (fig, label) in enumerate(zip(pos_figs, ['东向', '北向', '天向'])):
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("无法生成位置对比图")
                    
                    # 绘制气动力时间序列对比图
                    fa_figs, fa_total_figs = plot_aerodynamic_force(log_df, selected_run)
                    
                    if fa_figs or fa_total_figs:
                        st.markdown("---")
                        st.markdown("#### 🚁 气动力时间序列对比")
                        
                        # 图1：neural_f vs real_fa
                        st.markdown("##### 气动力对比（neural_f vs real_fa）")
                        tab_fa_x, tab_fa_y, tab_fa_z = st.tabs(["X方向", "Y方向", "Z方向"])
                        
                        with tab_fa_x:
                            if len(fa_figs) > 0:
                                st.plotly_chart(fa_figs[0], use_container_width=True)
                            else:
                                st.warning("无法生成X方向气动力对比图")
                        
                        with tab_fa_y:
                            if len(fa_figs) > 1:
                                st.plotly_chart(fa_figs[1], use_container_width=True)
                            else:
                                st.warning("无法生成Y方向气动力对比图")
                        
                        with tab_fa_z:
                            if len(fa_figs) > 2:
                                st.plotly_chart(fa_figs[2], use_container_width=True)
                            else:
                                st.warning("无法生成Z方向气动力对比图")
                        
                        # 图2：neural_f_total vs real_fa_total
                        st.markdown("##### 总力对比（neural_f_total vs real_fa_total）")
                        st.markdown("总力 = 气动力 + R@fT + m*g")
                        tab_fa_total_x, tab_fa_total_y, tab_fa_total_z = st.tabs(["X方向", "Y方向", "Z方向"])
                        
                        with tab_fa_total_x:
                            if len(fa_total_figs) > 0:
                                st.plotly_chart(fa_total_figs[0], use_container_width=True)
                            else:
                                st.warning("无法生成X方向总力对比图")
                        
                        with tab_fa_total_y:
                            if len(fa_total_figs) > 1:
                                st.plotly_chart(fa_total_figs[1], use_container_width=True)
                            else:
                                st.warning("无法生成Y方向总力对比图")
                        
                        with tab_fa_total_z:
                            if len(fa_total_figs) > 2:
                                st.plotly_chart(fa_total_figs[2], use_container_width=True)
                            else:
                                st.warning("无法生成Z方向总力对比图")
                else:
                    st.warning(f"无法加载日志文件: {log_file_name}")
                    st.info("提示：请确保日志文件在 navigation_logs 目录或对应的大任务文件夹中")
            else:
                st.info("该实验没有关联的日志文件")
            
            st.markdown("---")
            
            # 显示参数（放在时间序列对比下面）
            st.subheader("📋 实验参数")
            param_data = {k.replace('param_', ''): v for k, v in selected_row.items() if k.startswith('param_')}
            st.json(param_data)
            
            # 显示所有指标
            st.subheader("📊 所有指标")
            metric_data = {k.replace('metric_', ''): v for k, v in selected_row.items() if k.startswith('metric_')}
            st.json(metric_data)
    
if __name__ == "__main__":
    main()

