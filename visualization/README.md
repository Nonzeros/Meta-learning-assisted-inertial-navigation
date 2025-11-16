# 可视化工具使用说明

## 文件说明

- `mlflow_utils.py`: MLflow工具函数，用于记录实验参数和结果
- `streamlit_app.py`: Streamlit可视化应用
- `plot_utils.py`: 绘图工具函数

## 使用方法

### 1. 安装依赖

```bash
pip install mlflow streamlit plotly
```

### 2. 运行实验

运行 `src/main.py` 时，实验数据会自动记录到 `mlruns/` 目录下。

### 3. 启动Streamlit可视化

```bash
streamlit run visualization/streamlit_app.py
```

然后在浏览器中打开显示的URL（通常是 http://localhost:8501）

## MLflow记录的内容

### 模型参数
- `model_dataset`: 数据集名称
- `model_dim_a`: 模型维度a
- `model_features`: 特征列表
- `model_name`: 模型名称
- `model_stopping_epoch`: 训练轮数
- `model_file`: 模型文件路径

### 滤波参数
- `filter_solve_type`: 滤波类型（1=KF, 2=PF）
- `filter_numPar`: 粒子数（PF时）
- `filter_lambda1`: lambda1参数
- `filter_R`: 观测噪声协方差R
- `filter_Q`: 过程噪声协方差Q

### UKF参数
- `ukf_Qk`: UKF过程噪声协方差
- `ukf_Rk`: UKF观测噪声协方差
- `ukf_Pxk`: UKF状态协方差初值

### 数据集参数
- `dataset_folder`: 数据集文件夹
- `adapt_end_index`: 适应阶段长度

### 性能指标
- `ukf_vel_rmse_*`: UKF速度RMSE（东向、北向、天向、总体）
- `ukf_pos_rmse_*`: UKF位置RMSE（东向、北向、天向、总体）
- `pure_ins_vel_rmse_*`: 纯惯导速度RMSE
- `pure_ins_pos_rmse_*`: 纯惯导位置RMSE
- `fa_rmse_*`: 气动力RMSE（X、Y、Z、总体）

