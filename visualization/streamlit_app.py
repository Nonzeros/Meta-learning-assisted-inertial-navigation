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


def load_log_file(log_file_name: str, project_root: str = "."):
    """
    加载导航日志文件
    
    参数:
        log_file_name: 日志文件名
        project_root: 项目根目录
    """
    # 尝试多个可能的路径
    possible_paths = [
        os.path.join(project_root, "navigation_logs", log_file_name),
        os.path.join(project_root, log_file_name),
        log_file_name,
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                return df
            except Exception as e:
                st.error(f"读取日志文件失败 {path}: {str(e)}")
                return None
    
    # 如果找不到，尝试在navigation_logs目录中搜索
    log_dir = os.path.join(project_root, "navigation_logs")
    if os.path.exists(log_dir):
        pattern = os.path.join(log_dir, f"*{log_file_name}*")
        matches = glob.glob(pattern)
        if matches:
            try:
                df = pd.read_csv(matches[0])
                return df
            except Exception as e:
                st.error(f"读取日志文件失败 {matches[0]}: {str(e)}")
    
    return None


def create_summary_table(df):
    """
    创建任务汇总表，显示参数和指标
    """
    # 选择要显示的列
    param_cols = [col for col in df.columns if col.startswith('param_')]
    metric_cols = [col for col in df.columns if col.startswith('metric_')]
    
    # 创建汇总表
    summary_data = []
    for _, row in df.iterrows():
        summary_row = {
            '实验名称': row.get('run_name', 'Unknown'),
            '实验时间': row.get('start_time', ''),
            '状态': row.get('status', ''),
        }
        
        # 添加关键参数
        key_params = ['model_dim_a', 'filter_solve_type', 'filter_numPar', 
                     'dataset_folder', 'adapt_end_index']
        for param in key_params:
            param_col = f'param_{param}'
            if param_col in row:
                summary_row[param.replace('param_', '').replace('_', ' ').title()] = row[param_col]
        
        # 添加关键指标
        key_metrics = ['ukf_vel_rmse_total', 'ukf_pos_rmse_total', 
                      'pure_ins_vel_rmse_total', 'pure_ins_pos_rmse_total',
                      'fa_rmse_total']
        for metric in key_metrics:
            metric_col = f'metric_{metric}'
            if metric_col in row:
                summary_row[metric.replace('metric_', '').replace('_', ' ').title()] = f"{row[metric_col]:.6f}"
        
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
        
        # 真实位置 - 使用日志文件中的real_px/py/pz
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
        
        # 真实速度 - 使用日志文件中的real_vx/vy/vz
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
        
        # 真实姿态 - 使用日志文件中的real_att_x/y/z（如果有）
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
        
        fig.update_layout(
            title=f'姿态{label}对比',
            xaxis_title='时间 (s)',
            yaxis_title=f'姿态{label} (度)',
            height=400,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        att_figs.append(fig)
    
    return pos_figs, vel_figs, att_figs


def main():
    st.set_page_config(
        page_title="导航融合实验可视化",
        page_icon="📊",
        layout="wide"
    )
    
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
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 任务汇总表", "📊 RMSE对比", "📈 参数分析", "🔍 实验详情", "📈 时间序列对比"])
    
    with tab1:
        st.header("📋 任务汇总表")
        st.markdown("显示所有实验的参数和关键指标")
        
        # 创建汇总表
        summary_df = create_summary_table(df)
        
        if not summary_df.empty:
            st.dataframe(
                summary_df.sort_values('实验时间', ascending=False),
                use_container_width=True,
                height=600
            )
            
            # 下载按钮
            csv = summary_df.to_csv(index=False)
            st.download_button(
                label="下载汇总表 (CSV)",
                data=csv,
                file_name=f"experiment_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.warning("无法创建汇总表")
    
    with tab2:
        st.header("RMSE对比")
        
        # 选择要对比的指标
        metric_columns = [col for col in df.columns if col.startswith('metric_')]
        rmse_metrics = [col for col in metric_columns if 'rmse' in col.lower()]
        
        if len(rmse_metrics) > 0:
            selected_metrics = st.multiselect(
                "选择要对比的指标",
                options=rmse_metrics,
                default=rmse_metrics[:6]  # 默认选择前6个
            )
            
            if selected_metrics:
                # 创建对比图
                fig_data = []
                for metric in selected_metrics:
                    metric_name = metric.replace('metric_', '')
                    for _, row in df.iterrows():
                        if pd.notna(row[metric]):
                            fig_data.append({
                                '指标': metric_name,
                                '值': row[metric],
                                '实验': row.get('run_name', 'Unknown'),
                                '时间': row.get('start_time', '')
                            })
                
                if fig_data:
                    fig_df = pd.DataFrame(fig_data)
                    fig = px.bar(
                        fig_df,
                        x='指标',
                        y='值',
                        color='实验',
                        title='RMSE对比',
                        barmode='group'
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("没有找到RMSE指标数据")
    
    with tab3:
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
    
    with tab4:
        st.header("🔍 实验详情")
        
        if not df.empty:
            selected_run = st.selectbox(
                "选择实验运行",
                options=df['run_name'].unique() if 'run_name' in df.columns else df.index
            )
            
            selected_row = df[df['run_name'] == selected_run].iloc[0] if 'run_name' in df.columns else df.iloc[selected_run]
            
            # 显示参数
            st.subheader("参数")
            param_data = {k.replace('param_', ''): v for k, v in selected_row.items() if k.startswith('param_')}
            st.json(param_data)
            
            # 显示指标
            st.subheader("指标")
            metric_data = {k.replace('metric_', ''): v for k, v in selected_row.items() if k.startswith('metric_')}
            st.json(metric_data)
    
    with tab5:
        st.header("📈 时间序列对比")
        st.markdown("显示位置、速度、姿态的时间序列对比图（从日志文件读取）")
        
        if not df.empty:
            # 选择实验
            selected_run_ts = st.selectbox(
                "选择要查看的实验",
                options=df['run_name'].unique() if 'run_name' in df.columns else df.index,
                key="ts_run_selector"
            )
            
            selected_row_ts = df[df['run_name'] == selected_run_ts].iloc[0] if 'run_name' in df.columns else df.iloc[selected_run_ts]
            
            # 获取日志文件名
            log_file_name = selected_row_ts.get('log_file', None)
            
            if log_file_name:
                # 加载日志文件
                with st.spinner(f"正在加载日志文件: {log_file_name}"):
                    log_df = load_log_file(log_file_name, project_root)
                
                if log_df is not None and not log_df.empty:
                    # 绘制时间序列图
                    pos_figs, vel_figs, att_figs = plot_time_series(log_df, selected_run_ts)
                    
                    # 使用标签页分别显示姿态、速度、位置
                    tab_att, tab_vel, tab_pos = st.tabs(["🎯 姿态对比", "⚡ 速度对比", "📍 位置对比"])
                    
                    with tab_att:
                        st.subheader("姿态对比（X、Y、Z方向）")
                        if att_figs:
                            for i, (fig, label) in enumerate(zip(att_figs, ['X', 'Y', 'Z'])):
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("无法生成姿态对比图")
                    
                    with tab_vel:
                        st.subheader("速度对比（东向、北向、天向）")
                        if vel_figs:
                            for i, (fig, label) in enumerate(zip(vel_figs, ['东向', '北向', '天向'])):
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("无法生成速度对比图")
                    
                    with tab_pos:
                        st.subheader("位置对比（东向、北向、天向）")
                        if pos_figs:
                            for i, (fig, label) in enumerate(zip(pos_figs, ['东向', '北向', '天向'])):
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("无法生成位置对比图")
                else:
                    st.warning(f"无法加载日志文件: {log_file_name}")
                    st.info("提示：请确保日志文件在 navigation_logs 目录中")
            else:
                st.info("该实验没有关联的日志文件")


if __name__ == "__main__":
    main()

