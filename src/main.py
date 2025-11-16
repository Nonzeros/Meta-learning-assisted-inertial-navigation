import matlab.engine
import time
import sys
import os
import numpy as np
import torch
import csv
import yaml
from datetime import datetime
from scipy.interpolate import interp1d

import utils
import mlmodel
import data_extractor

import numpy as np
import matplotlib.pyplot as plt
import traceback

# 导入MLflow工具
import sys
import mlflow
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'visualization'))
from mlflow_utils import (
    setup_mlflow_experiment, 
    log_experiment_params, 
    log_experiment_metrics,
    log_model_file,
    start_run,
    end_run
)

# 获取项目根目录（main.py 在 src/ 目录下，所以需要向上两级）
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 读取滤波器配置文件
config_path = os.path.join(project_root, 'configs', 'filter_config.yaml')
with open(config_path, 'r', encoding='utf-8') as f:
    filter_config = yaml.safe_load(f)

# 动力学信息获取
## 1.模型、数据准备
#测试集
from experiment_runner import run_single_experiment

# 要处理的CSV文件列表
csv_files_to_process = [
    'custom_figure8_baseline_35wind.csv',
    'custom_figure8_baseline_70p20sint.csv',
    'custom_figure8_baseline_70wind.csv',
    'custom_figure8_baseline_100wind.csv',
    'custom_figure8_baseline_nowind.csv'
]

adapt_end_index = 100 # 适应部分下标(不包括)

# ========== MLflow实验记录初始化 ==========
# 设置MLflow实验
mlflow_tracking_uri = os.path.join(project_root, "mlruns")
experiment_name, _ = setup_mlflow_experiment(tracking_uri=mlflow_tracking_uri)

# 2.matlab设置
eng = matlab.engine.start_matlab()
# 将matlab文件加入工作目录
matlab_utils_path = os.path.join(project_root, 'matlab', 'utils')
matlab_psins_path = os.path.join(project_root, 'matlab', 'third_part', 'psins240809')
# 将路径转换为MATLAB格式（使用正斜杠）
eng.addpath(matlab_utils_path.replace('\\', '/'))
eng.addpath(matlab_psins_path.replace('\\', '/'))
# 导入psins全局变量
glv_init_code = """
    % 声明glv为全局变量
    global glv;
    % 初始化Re、f、wie（若未定义则设为空，再赋值默认值）
    if ~exist('Re', 'var'),  Re = [];  end;
    if ~exist('f', 'var'),   f = [];  end;
    if ~exist('wie', 'var'), wie = [];  end;
    if isempty(Re),  Re = 6378137;  end;
    if isempty(f),   f = 1/298.257;  end;
    if isempty(wie), wie = 7.2921151467e-5;  end;
    % 赋值glv核心属性（地球椭球、物理参数等）
    glv.Re = Re;                    % 地球长半轴
    glv.f = f;                      % 地球扁率
    glv.Rp = (1-glv.f)*glv.Re;      % 地球短半轴
    glv.e = sqrt(2*glv.f-glv.f^2);  glv.e2 = glv.e^2; % 第一偏心率及平方
    glv.ep = sqrt(glv.Re^2-glv.Rp^2)/glv.Rp;  glv.ep2 = glv.ep^2; % 第二偏心率及平方
    glv.GM = 3.986004418e14;        % 地球引力常数
    glv.wie = wie;                  % 地球自转角速度
    glv.meru = glv.wie/1000;        % 毫地球自转角速度单位
    glv.g0 = 9.7803267715;          % 标准重力加速度
    % 计算beta相关参数
    m = Re*glv.wie^2/glv.g0;  glv.beta = 5/2*m-f-17/14*m*f;
    glv.beta1 = (5*m*f-f^2)/8;  glv.beta2 = 3.086e-6;  glv.beta3 = 8.08e-9;
    % 单位转换相关属性
    glv.mg = 1.0e-3*glv.g0;         % 毫重力加速度
    glv.ug = 1.0e-6*glv.g0;         % 微重力加速度
    glv.mGal = 1.0e-3*0.01;         % 毫伽（1cm/s²）
    glv.uGal = glv.mGal/1000;       % 微伽
    glv.ugpg = glv.ug/glv.g0;       % 微重力/重力比（ug/g）
    glv.ugpg2 = glv.ug/glv.g0^2;    % ug/g²
    glv.ugpg3 = glv.ug/glv.g0^3;    % ug/g³
    glv.ws = 1/sqrt(glv.Re/glv.g0); % 舒勒频率
    glv.ppm = 1.0e-6;               % 百万分之一
    glv.deg = pi/180;               % 角度转弧度系数
    glv.min = glv.deg/60;           % 角分转弧度系数
    glv.sec = glv.min/60;           % 角秒转弧度系数
    glv.mas = glv.sec/1000;         % 毫角秒转弧度系数
    glv.hur = 3600;                 % 小时转秒系数
    glv.dps = pi/180/1;             % 度/秒（转弧度后）
    glv.mdps = glv.dps/1000;        % 毫度/秒
    glv.rps = 360*glv.dps;          % 转/秒（转弧度后）
    glv.dph = glv.deg/glv.hur;      % 度/小时
    glv.dpss = glv.deg/sqrt(1);     % 度/√秒
    glv.dpsh = glv.deg/sqrt(glv.hur);  % 度/√小时
    glv.dphpsh = glv.dph/sqrt(glv.hur); % (度/小时)/√小时
    glv.dph2 = glv.dph/glv.hur;     % (度/小时)/小时
    glv.secpg = glv.sec/glv.g0;     % 角秒/g
    glv.secpdps2 = glv.sec/(glv.deg/1^2);    % 角秒/(度/秒²)
    glv.secprps2 = glv.sec/(1/1^2);    % 角秒/(弧度/秒²)
    glv.Hz = 1/1;                   % 赫兹（1/秒）
    glv.dphpsHz = glv.dph/glv.Hz;   % (度/小时)/√赫兹
    glv.dphpg = glv.dph/glv.g0;     % (度/小时)/g
    glv.dphpg2 = glv.dphpg/glv.g0;  % (度/小时)/g²
    glv.ugpsHz = glv.ug/sqrt(glv.Hz);  % ug/√赫兹
    glv.ugpsh = glv.ug/sqrt(glv.hur); % ug/√小时
    glv.ugph = glv.ug/glv.hur;      % ug/小时
    glv.ugphpsh = glv.ugph/sqrt(glv.hur);  % (ug/小时)/√小时
    glv.mpsh = 1/sqrt(glv.hur);     % m/√小时
    glv.mpspsh = 1/1/sqrt(glv.hur); % (m/s)/√小时
    glv.ppmpsh = glv.ppm/sqrt(glv.hur); % ppm/√小时
    glv.mil = 2*pi/6000;            % 密位（2π/6000弧度）
    glv.nm = 1853;                  % 海里（1853米）
    glv.kn = glv.nm/glv.hur;        % 节（海里/小时）
    glv.kmph = 1000/glv.hur;        % 千米/小时
    % 初始化惯性导航相关缓存变量
    glv.wm_1 = [0,0,0];  glv.vm_1 = [0,0,0];   % 前一时刻陀螺/加速度计采样值
    % 圆锥/划船补偿系数矩阵
    glv.cs = [                     
        [2,    0,    0,    0,    0    ]/3;
        [9,    27,   0,    0,    0    ]/20;
        [54,   92,   214,  0,    0    ]/105;
        [250,  525,  650,  1375, 0    ]/504;
        [2315, 4558, 7296, 7834, 15797]/4620
    ];
    glv.csmax = size(glv.cs,1)+1;  % 最大子采样数
    glv.csCompensate = 1;          % 补偿使能（1=使能，0=关闭）
    glv.v0 = [0;0;0];              % 3×1零向量
    glv.qI = [1;0;0;0];            % 单位四元数
    glv.I33 = eye(3);  glv.o33 = zeros(3);  % 3×3单位矩阵和零矩阵
    % 初始位置（NWPU新位置）
    glv.pos0 = [34.034310*glv.deg; 108.775427*glv.deg; 450];
    glv.eth = [];  glv.eth = earth(glv.pos0);  % 地球参数计算
    glv.t0 = 0;                    % 初始时间
    glv.tscale = 1;                % 时间缩放系数（1=秒，60=分，3600=小时）
    glv.isfig = 1;                 % 图形显示使能
    glv.gfix = [];  glv.dgn = [];  % 备用变量
"""

# ========== 循环处理多个文件 ==========
# 初始化运行日志
run_log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_log_dir = os.path.join(project_root, "navigation_logs")
if not os.path.exists(run_log_dir):
    os.makedirs(run_log_dir)
run_log_file = os.path.join(run_log_dir, f"experiment_run_log_{run_log_timestamp}.txt")
run_log = open(run_log_file, 'w', encoding='utf-8')

# 实验结果汇总
experiment_results = []
failed_files = []

print(f"\n{'='*60}")
print(f"开始自动化循环实验")
print(f"共 {len(csv_files_to_process)} 个文件需要处理")
print(f"{'='*60}\n")

# 循环处理每个文件
for idx, csv_filename in enumerate(csv_files_to_process, 1):
    print(f"\n{'='*60}")
    print(f"正在处理第 {idx}/{len(csv_files_to_process)} 个文件: {csv_filename}")
    print(f"{'='*60}")
    
    run_log.write(f"\n{'='*60}\n")
    run_log.write(f"文件 {idx}/{len(csv_files_to_process)}: {csv_filename}\n")
    run_log.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    run_log.write(f"{'='*60}\n")
    run_log.flush()
    
    try:
        # 为每个文件创建独立的MLflow run
        mlflow_run = start_run(run_name=f"{csv_filename.replace('.csv', '')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        # 运行单个实验
        success, error_msg, results = run_single_experiment(
            csv_filename=csv_filename,
            project_root=project_root,
            filter_config=filter_config,
            eng=eng,
            adapt_end_index=adapt_end_index,
            glv_init_code=glv_init_code
        )
        
        if success:
            print(f"✓ 文件 {csv_filename} 处理成功")
            run_log.write(f"状态: 成功\n")
            if results:
                experiment_results.append({
                    'file': csv_filename,
                    'success': True,
                    'results': results
                })
                print(f"  - 日志文件: {results.get('log_file', 'N/A')}")
                print(f"  - UKF速度RMSE: {results.get('ukf_vel_rmse', 'N/A')}")
                print(f"  - UKF位置RMSE: {results.get('ukf_pos_rmse', 'N/A')}")
        else:
            print(f"✗ 文件 {csv_filename} 处理失败: {error_msg}")
            run_log.write(f"状态: 失败\n")
            run_log.write(f"错误信息: {error_msg}\n")
            failed_files.append({
                'file': csv_filename,
                'error': error_msg
            })
            experiment_results.append({
                'file': csv_filename,
                'success': False,
                'error': error_msg
            })
        
        # 结束当前MLflow run
        end_run()
        
    except Exception as e:
        error_msg = f"处理文件 {csv_filename} 时发生异常: {str(e)}\n{traceback.format_exc()}"
        print(f"✗ 文件 {csv_filename} 处理异常: {error_msg}")
        run_log.write(f"状态: 异常\n")
        run_log.write(f"错误信息: {error_msg}\n")
        failed_files.append({
            'file': csv_filename,
            'error': error_msg
        })
        experiment_results.append({
            'file': csv_filename,
            'success': False,
            'error': error_msg
        })
        # 确保MLflow run被结束（即使出错）
        try:
            end_run()
        except:
            pass

# 关闭MATLAB引擎
try:
    eng.quit()
except:
    pass

# ========== 输出汇总报告 ==========
print(f"\n{'='*60}")
print(f"实验汇总报告")
print(f"{'='*60}")

success_count = sum(1 for r in experiment_results if r.get('success', False))
failed_count = len(failed_files)

print(f"\n处理完成统计:")
print(f"  - 总文件数: {len(csv_files_to_process)}")
print(f"  - 成功: {success_count}")
print(f"  - 失败: {failed_count}")

if success_count > 0:
    print(f"\n成功处理的文件:")
    for r in experiment_results:
        if r.get('success', False):
            print(f"  ✓ {r['file']}")
            if 'results' in r and r['results']:
                metrics = r['results'].get('metrics', {})
                print(f"    - UKF速度RMSE (总体): {metrics.get('ukf_vel_rmse_total', 'N/A'):.6f}")
                print(f"    - UKF位置RMSE (总体): {metrics.get('ukf_pos_rmse_total', 'N/A'):.6f}")

if failed_count > 0:
    print(f"\n失败的文件:")
    for f in failed_files:
        print(f"  ✗ {f['file']}")
        print(f"    错误: {f['error'][:200]}...")  # 只显示前200个字符

print(f"\n运行日志已保存到: {run_log_file}")
print(f"{'='*60}\n")

# 写入运行日志
run_log.write(f"\n{'='*60}\n")
run_log.write(f"实验汇总报告\n")
run_log.write(f"{'='*60}\n")
run_log.write(f"处理完成统计:\n")
run_log.write(f"  - 总文件数: {len(csv_files_to_process)}\n")
run_log.write(f"  - 成功: {success_count}\n")
run_log.write(f"  - 失败: {failed_count}\n")
run_log.close()

print("所有实验已完成！")
