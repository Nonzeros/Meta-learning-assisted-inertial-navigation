"""
MLflow工具函数，用于记录实验参数和结果
"""
import os
import mlflow
import mlflow.pytorch
from datetime import datetime
import numpy as np
from typing import Dict, Any, Optional


def setup_mlflow_experiment(experiment_name: Optional[str] = None, tracking_uri: str = "./mlruns"):
    """
    设置MLflow实验
    
    参数:
        experiment_name: 实验名称，如果为None则使用日期命名
        tracking_uri: MLflow跟踪URI，默认为本地文件系统
    """
    # 将路径转换为MLflow支持的URI格式
    # 如果是绝对路径，需要添加file://前缀
    if os.path.isabs(tracking_uri):
        # Windows路径需要file:///（三个斜杠），Unix路径需要file://（两个斜杠）
        if os.name == 'nt':  # Windows
            # 将反斜杠转换为正斜杠，并添加file:///前缀
            normalized_path = tracking_uri.replace('\\', '/')
            # 处理Windows盘符（如 F:/ -> /F:/）
            if ':' in normalized_path:
                parts = normalized_path.split(':', 1)
                normalized_path = f"/{parts[0]}:{parts[1]}"
            tracking_uri = f"file://{normalized_path}"
        else:  # Unix/Linux/Mac
            tracking_uri = f"file://{tracking_uri}"
    
    # 设置跟踪URI
    mlflow.set_tracking_uri(tracking_uri)
    
    # 如果没有指定实验名称，使用日期命名
    if experiment_name is None:
        experiment_name = f"navigation_experiment_{datetime.now().strftime('%Y%m%d')}"
    
    # 创建或获取实验
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except Exception:
        # 如果实验已存在，获取其ID
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
    
    mlflow.set_experiment(experiment_name)
    return experiment_name, experiment_id


def log_experiment_params(
    model_params: Dict[str, Any],
    filter_params: Dict[str, Any],
    ukf_params: Dict[str, Any],
    dataset_params: Dict[str, Any],
    other_params: Optional[Dict[str, Any]] = None
):
    """
    记录实验参数到MLflow
    
    参数:
        model_params: 模型相关参数
        filter_params: 滤波方法相关参数
        ukf_params: UKF相关参数
        dataset_params: 数据集相关参数
        other_params: 其他参数
    """
    # 记录模型参数
    mlflow.log_params(model_params)
    
    # 记录滤波参数
    mlflow.log_params(filter_params)
    
    # 记录UKF参数（需要转换为字符串，因为可能是数组）
    for key, value in ukf_params.items():
        if isinstance(value, np.ndarray):
            # 如果是数组，记录为字符串或记录主要值
            if value.size <= 9:  # 小数组，记录所有值
                mlflow.log_param(f"ukf_{key}", str(value.tolist()))
            else:  # 大数组，记录形状和主要统计信息
                mlflow.log_param(f"ukf_{key}_shape", str(value.shape))
                mlflow.log_param(f"ukf_{key}_mean", float(np.mean(value)))
                mlflow.log_param(f"ukf_{key}_std", float(np.std(value)))
        else:
            mlflow.log_param(f"ukf_{key}", value)
    
    # 记录数据集参数
    mlflow.log_params(dataset_params)
    
    # 记录其他参数
    if other_params:
        mlflow.log_params(other_params)


def log_experiment_metrics(metrics: Dict[str, float]):
    """
    记录实验指标到MLflow
    
    参数:
        metrics: 指标字典，键为指标名称，值为指标值
    """
    mlflow.log_metrics(metrics)


def log_model_file(model_path: str, artifact_path: str = "model"):
    """
    记录模型文件到MLflow
    
    参数:
        model_path: 模型文件路径
        artifact_path: MLflow中的artifact路径
    """
    if os.path.exists(model_path):
        mlflow.log_artifact(model_path, artifact_path=artifact_path)
    else:
        print(f"警告: 模型文件不存在: {model_path}")


def start_run(run_name: Optional[str] = None):
    """
    开始一个MLflow运行
    
    参数:
        run_name: 运行名称
    """
    return mlflow.start_run(run_name=run_name)


def end_run():
    """
    结束当前MLflow运行
    """
    mlflow.end_run()

