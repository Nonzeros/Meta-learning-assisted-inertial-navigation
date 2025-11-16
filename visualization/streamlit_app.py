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


def load_mlflow_experiments(tracking_uri: str = "./mlruns"):
    """
    加载MLflow实验数据
    
    参数:
        tracking_uri: MLflow跟踪URI
    """
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
                
                experiments.append(run_data)
    except Exception as e:
        st.error(f"加载MLflow数据时出错: {str(e)}")
    
    return pd.DataFrame(experiments)


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
    tab1, tab2, tab3, tab4 = st.tabs(["📋 实验列表", "📊 RMSE对比", "📈 参数分析", "🔍 实验详情"])
    
    with tab1:
        st.header("实验列表")
        
        # 显示数据表格
        display_columns = ['run_name', 'experiment_name', 'start_time', 'status']
        metric_columns = [col for col in df.columns if col.startswith('metric_')]
        display_columns.extend(metric_columns[:10])  # 显示前10个指标
        
        st.dataframe(
            df[display_columns].sort_values('start_time', ascending=False),
            use_container_width=True,
            height=400
        )
        
        # 下载按钮
        csv = df.to_csv(index=False)
        st.download_button(
            label="下载数据 (CSV)",
            data=csv,
            file_name=f"experiments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
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
                        title=f'{selected_param.replace("param_", "")} vs {selected_metric.replace("metric_", "")}',
                        trendline="ols"
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("实验详情")
        
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


if __name__ == "__main__":
    main()

