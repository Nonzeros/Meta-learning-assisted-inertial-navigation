import matlab.engine
import time
import sys
import os
import numpy as np
import torch
import csv
from datetime import datetime
from scipy.interpolate import interp1d

import utils
import mlmodel
import experiment_utils

import numpy as np
import matplotlib.pyplot as plt

# 导入MLflow工具
import sys
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

# 动力学信息获取
## 1.模型、数据准备
#测试集
from intelligent_dynamic_module import intelligent_dynamic_module

# ========== 从配置文件加载参数 ==========
config_path = os.path.join(project_root, 'configs', 'experiment_config.yaml')
config = experiment_utils.load_config(config_path)

# 模型参数
model_config = config['model']
dataset = model_config['dataset']
dim_a = model_config['dim_a']
features = model_config['features']
stopping_epoch = model_config['stopping_epoch']

# 验证集配置
validation_config = config['validation']
dataset_folder = os.path.join(project_root, validation_config['dataset_folder'])
validation_files = validation_config['validation_files']

# 适应阶段配置
adaptation_config = config['adaptation']
adapt_duration = adaptation_config['adapt_duration']  # 适应阶段时长（秒）

# 滤波参数
filter_config = config['filter']
solver_type = filter_config['solve_type']
numPar = filter_config['numPar']
lambda1 = filter_config['lambda1']
q0 = filter_config['q0']
r0 = filter_config['r0']

# 循环配置
loop_config = config.get('loop', {})
loops = loop_config.get('loops', 2000)
max_duration = loop_config.get('max_duration', None)

# 构建options字典
options = {
    'dim_a': dim_a,
    'loss_type': 'crossentropy-loss',
    'solve_type': solver_type,
    'lambda1': lambda1,
    'numPar': numPar,
    'q0': q0,
    'r0': r0,
}

# 输出options各参数
print("实验配置:")
print(f"  验证集文件夹: {dataset_folder}")
print(f"  验证集文件: {validation_files}")
print(f"  适应阶段时长: {adapt_duration}秒")
print(f"  滤波参数: solve_type={solver_type}, q0={q0}, r0={r0}")
print(f"  Options: {options}")

# ========== 批量实验循环 ==========
for validation_file in validation_files:
    print(f"\n{'='*60}")
    print(f"处理验证集文件: {validation_file}")
    print(f"{'='*60}")
    
    # 加载验证集CSV文件
    validation_csv_path = os.path.join(dataset_folder, validation_file)
    validation_data = experiment_utils.load_validation_csv(validation_csv_path)
    
    # 计算时间步长和适应阶段索引
    ts = validation_data['t']
    if len(ts) > 1:
        dt = float(ts[1] - ts[0])
    else:
        dt = 0.02
    adapt_end_index = int(adapt_duration / dt)
    
    print(f"  时间步长: {dt}秒")
    print(f"  适应阶段索引: {adapt_end_index} (对应 {adapt_duration}秒)")
    
    # 提取数据（用于动力学模型）
    # 使用utils.load_data加载数据（用于format_data）
    RawData = utils.load_data(dataset_folder, expnames=[validation_file.replace('.csv', '')])
    if len(RawData) == 0:
        print(f"警告: 无法加载数据文件 {validation_file}，跳过")
        continue
    
    data_num = 0  # 因为只加载了一个文件
    Data = utils.format_data(RawData, features=features, output='fa')
    data = Data[data_num]
    
    # 提取真实数据
R = RawData[data_num]['R']
T_sps = RawData[data_num]['T_sp']
hover_throttles = RawData[data_num]['hover_throttle']
    real_p = RawData[data_num]['p']
real_p_row = real_p.T
    real_v = RawData[data_num]['v']
real_v_row = real_v.T
    real_fas = RawData[data_num]['fa']
desire_ps = RawData[data_num]['p_d']
desire_vs = RawData[data_num]['v_d']
    
    # 计算验证集数据长度
    validation_length = experiment_utils.calculate_data_length(
        validation_data, adapt_end_index, loops, max_duration
    )
    print(f"  验证集数据长度: {validation_length} (从索引 {adapt_end_index} 开始)")

    # ========== MLflow实验记录初始化 ==========
    # 为每个验证集文件创建独立的MLflow运行
    mlflow_tracking_uri = os.path.join(project_root, "mlruns")
    experiment_name, _ = setup_mlflow_experiment(tracking_uri=mlflow_tracking_uri)
    run_name = f"{validation_file.replace('.csv', '')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    mlflow_run = start_run(run_name=run_name)

    # 2.matlab设置（只在第一次启动）
    if 'eng' not in locals():
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
eng.eval(glv_init_code, nargout=0)  # nargout=0：无输出，仅执行初始化
    
    # 准备传给MATLAB的数据
    pavq_data, matlab_dt = experiment_utils.prepare_matlab_data(validation_data, adapt_end_index)
    
    # 初始位置（用于坐标转换）
    lat0 = 34.13801
    lon0 = -118.12528
    h0 = 2.0470
    
    # 调用新的MATLAB函数进行惯导反演
    matlab_imu, matlab_avp0 = eng.av2imu_from_python(
        matlab.double(pavq_data.tolist()),
        matlab.double([1]),  # start_index从1开始（MATLAB索引）
        matlab.double([lat0]),
        matlab.double([lon0]),
        matlab.double([h0]),
        nargout=2
    )
# 转为Python的NumPy数组
last_avp = np.array(matlab_avp0)
    
# 初始信息设置
    start_time_sec = adapt_end_index * dt
last_avp = np.append(last_avp, start_time_sec)
last_avp = last_avp.reshape((1,10)) # 1*10
    last_avp[0,3:6] = real_v[adapt_end_index,0:3]

# xyz的avp0转llh 取验证集数据作为真实的位置
last_avp_xyz = last_avp
last_avp_xyz[0,6:9] = real_p[adapt_end_index, 0:3]

last_avp_llh_matlab = eng.xyz2llh_subfun( matlab.double( last_avp_xyz.tolist() ) )
last_avp = np.array(last_avp_llh_matlab)

# 计算纯惯导求解结果
matlab_pure_avps = eng.pure_ins_solve(matlab_imu, matlab.double( last_avp.tolist() ),nargout=1 )
pure_avps = np.array(matlab_pure_avps) # num * 10
pure_avps = pure_avps.T

avp0_change = last_avp # 记录初值
imu = np.array(matlab_imu) # 这个imu，就是拿截取以后的数据来算的

# UKF初始化
matlab_kf, matlab_ins = eng.SINS_dynamic_UKF153_init(matlab.double(avp0_change.tolist()),nargout=2)

    # 获取UKF参数（Q, R, P0）
    try:
        # 从MATLAB结构体中获取参数
        ukf_Qk = np.array(eng.getfield(matlab_kf, 'Qk')) if 'Qk' in str(matlab_kf) else None
        ukf_Rk = np.array(eng.getfield(matlab_kf, 'Rk')) if 'Rk' in str(matlab_kf) else None
        ukf_Pxk = np.array(eng.getfield(matlab_kf, 'Pxk')) if 'Pxk' in str(matlab_kf) else None
    except Exception as e:
        print(f"警告: 无法获取UKF参数: {e}")
        # 如果无法获取，尝试使用MATLAB eval
        try:
            ukf_Qk = np.array(eng.eval('kf.Qk', nargout=1))
            ukf_Rk = np.array(eng.eval('kf.Rk', nargout=1))
            ukf_Pxk = np.array(eng.eval('kf.Pxk', nargout=1))
        except:
            print("无法从MATLAB获取UKF参数，将使用默认值")
            ukf_Qk = None
            ukf_Rk = None
            ukf_Pxk = None
    
# 3.动力学模型神经网络初始设置
# 适应阶段最小二乘计算 a 的初始值
modelname = f"{dataset}_dim-a-{dim_a}_{'-'.join(features)}"
final_model = mlmodel.load_model(modelname=modelname + '-epoch-' + str(stopping_epoch))  # 导入最终模型
lam = 0.1

adaptinput = data.X[0:adapt_end_index, :]
adaptlabel = data.Y[0:adapt_end_index, :]
X = torch.from_numpy(adaptinput)  # K x dim_x
Y = torch.from_numpy(adaptlabel)  # K x dim_y
Phi = final_model.phi(X)  # K x dim_a
Phi_T = Phi.transpose(0, 1)  # dim_a x K
A = torch.inverse(torch.mm(Phi_T, Phi) + lam * torch.eye(dim_a))  # dim_a x dim_a
a0 = torch.mm(torch.mm(A, Phi_T), Y)  # dim_a x dim_y
adapt_prediction = torch.mm(final_model.phi(X), a0)  # K x dim_y

dynamic_a = a0.detach().numpy()
p0 = 0.1
dynamic_P = np.full((3, dim_a), 0.1)

    # 4.循环依次计算
    # 动态计算循环次数
    actual_loops = min(validation_length, loops) if loops > 0 else validation_length
    ukf_avps = np.empty((10, actual_loops))
ukf_avps[:,0] = last_avp
first_index = adapt_end_index
last_last_avp = last_avp.copy()
    
    # 用于收集预测的气动力数据（用于计算RMSE）
    neural_fa_collection = []  # 存储所有预测的气动力
    real_fa_collection = []    # 存储对应的真实气动力
    fa_time_collection = []    # 存储对应的时间

# 初始化日志文件
log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(project_root, "navigation_logs")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

    log_file = os.path.join(log_dir, f"navigation_log_{validation_file.replace('.csv', '')}_{log_timestamp}.csv")
log_header = [
    'loop_index', 'time',
    # UKF融合前状态（上一时刻）
    'ukf_prev_att_x', 'ukf_prev_att_y', 'ukf_prev_att_z',
    'ukf_prev_vx', 'ukf_prev_vy', 'ukf_prev_vz',
    'ukf_prev_px', 'ukf_prev_py', 'ukf_prev_pz',
    # 动力学模型输入
    'dynamic_input_vx', 'dynamic_input_vy', 'dynamic_input_vz',
    'dynamic_input_q0', 'dynamic_input_q1', 'dynamic_input_q2', 'dynamic_input_q3',
    'pwm0', 'pwm1', 'pwm2', 'pwm3',
    # 气动力相关
    'neural_fa_x', 'neural_fa_y', 'neural_fa_z',  # 神经网络预测的气动力
    'calculated_fa_x', 'calculated_fa_y', 'calculated_fa_z',  # 通过运动方程计算的气动力
    'dynamic_a_0', 'dynamic_a_1', 'dynamic_a_2',  # 动力学模型参数
    # 动力学模型输出
    'dynamic_pos_x', 'dynamic_pos_y', 'dynamic_pos_z',
    'dynamic_vel_x', 'dynamic_vel_y', 'dynamic_vel_z',
    'dynamic_vdot_x', 'dynamic_vdot_y', 'dynamic_vdot_z',
    # 惯导预测（UKF融合前）
    'ins_pred_pos_x', 'ins_pred_pos_y', 'ins_pred_pos_z',
    'ins_pred_att_x', 'ins_pred_att_y', 'ins_pred_att_z',
    'ins_pred_vx', 'ins_pred_vy', 'ins_pred_vz',
    # 观测残差（惯导位置 - 动力学位置）
    'obs_residual_x', 'obs_residual_y', 'obs_residual_z',
    # UKF融合后状态
    'ukf_fused_att_x', 'ukf_fused_att_y', 'ukf_fused_att_z',
    'ukf_fused_vx', 'ukf_fused_vy', 'ukf_fused_vz',
    'ukf_fused_px', 'ukf_fused_py', 'ukf_fused_pz',
    # 真实值（用于对比）
    'real_att_x', 'real_att_y', 'real_att_z',
    'real_vx', 'real_vy', 'real_vz',
    'real_px', 'real_py', 'real_pz',
    # 期望值
    'desire_vx', 'desire_vy', 'desire_vz',
    'desire_px', 'desire_py', 'desire_pz',
    # 其他关键变量
    'T_sp', 'hover_throttle'
]

log_csv = open(log_file, 'w', newline='', encoding='utf-8')
log_writer = csv.writer(log_csv)
log_writer.writerow(log_header)
print(f"日志文件已创建: {log_file}")
    
    for loop_index in range(first_index, first_index + actual_loops):
    # 1)动力学取数据
    inputdata = data.X[loop_index-1,:]
    inputdata[0:3] = last_avp[0,3:6]
    # 添加一个姿态角转四元数的，使用 a2qua_subfun.m 函数
    matlab_qua = eng.a2qua_subfun(matlab.double(last_avp[0,0:3].tolist()),nargout=1)
    qua_array = np.array(matlab_qua).flatten()  # 转换为numpy数组并展平为一维
    inputdata[3:7] = qua_array

    # 标签，也就是气动力得用UKF给的结果来算，这里先试着用已有数据计算吧
    last_last_avp = last_last_avp.reshape((1, 10))
        middle = (last_avp - last_last_avp) / dt
    vdot_mins1 = middle[:,3:6].reshape(3,1)
    # 使用 a2mat_fun 函数计算姿态变换矩阵 Ri
    att = last_avp[0,0:3]
    matlab_Ri = eng.a2mat_subfun(matlab.double(att.tolist()), nargout=1)
    Ri = np.array(matlab_Ri)
    hover_throttle = hover_throttles[loop_index-1]
    T_sp = T_sps[loop_index-1]
    m0 = 2.6 # 原文文献提到他们自定义无人机的质量为2.6kg
    g_ = 9.8
    g = np.array([0,0,-g_])
    g = g.reshape((3,1))
    fT = np.array([0, 0, float(T_sp / hover_throttle) * 9.8 * m0])
    fT = fT.reshape((3, 1))

    outputlabel = m0 * vdot_mins1 - m0 * g - Ri @ fT

    desire_v = desire_vs[loop_index-1,:]
    desire_v = desire_v.reshape((3,1))
    desire_p = desire_ps[loop_index-1,:]
    desire_p = desire_p.reshape((3,1))

    # if last_avp.size > 9:
    last_avp = last_avp.reshape((10,1)) # 原来是行向量

    last_v = last_avp[3:6,:] # 10 * 1
    delta_v = desire_v - last_v

    # 2)动力学更新(输入：速度、姿态(上一时刻的)、控制输入，label，由微分计算得到； 输出：位置)
    # 输入的速度位置是3*1的，输出是3*1的位置
    # 输入的位置需要是xyz下的。
    last_avp = last_avp.reshape((1,10))
    # matlab_last_avp_xyz = eng.llh2xyz_subfun(matlab.double(last_avp.tolist()), matlab.double(avp0.tolist()))
    matlab_last_avp_xyz = eng.llh2xyz_subfun(matlab.double(last_avp.tolist()))

    last_avp_xyz = np.array(matlab_last_avp_xyz)
    last_avp_xyz = np.append(last_avp_xyz, last_avp[0,9])
    last_avp_xyz = last_avp_xyz.reshape((10, 1))

    last_p = last_avp_xyz[6:9,0].reshape((3,1))
    delta_p = desire_p - last_p

    last_avp = last_avp.reshape((10,1))

    vt_minus1 = last_avp_xyz[3:6,:]
    pt_minus1 = last_avp_xyz[6:9,:]

    px = np.tile(dynamic_a, (numPar, 1, 1))
    pw = np.full((numPar, 1), 1 / numPar)
    dynamic_pos,dynamic_P,dynamic_a,px,pw,neural_fa,dynamic_vel,dynamic_vdot = intelligent_dynamic_module(inputdata,outputlabel,delta_v,delta_p,dynamic_a,dynamic_P, dim_a,
                                                                 hover_throttle,T_sp,Ri,vt_minus1,pt_minus1,px,pw,options)

    # 记录动力学模型输出
    dynamic_pos_xyz = dynamic_pos.flatten()
    dynamic_vel_xyz = dynamic_vel.flatten()
    dynamic_vdot_xyz = dynamic_vdot.flatten()
    neural_fa_xyz = neural_fa.flatten()
        
        # 收集预测的气动力和对应的真实气动力（用于后续计算RMSE）
        if loop_index-1 < len(real_fas):
            neural_fa_collection.append(neural_fa_xyz)
            real_fa_collection.append(real_fas[loop_index-1, :])
            fa_time_collection.append(ts[loop_index-1] if loop_index-1 < len(ts) else loop_index * dt)

    # 3)UKF更新(需要输入和输出的  输入：比例增量、角增量，无需输入。动力学位置  输出：求解的avp)
    imu_index = loop_index - first_index
    dynamic_pos_list = dynamic_pos.tolist()
    matlab_dynamic_pos = matlab.double(dynamic_pos_list)
    imu_list = imu[imu_index,:].tolist()
    matlab_imu_i = matlab.double(imu_list)

    # 3)UKF更新
    # 调用MATLAB函数进行UKF融合，返回融合后的状态和融合前的INS预测位置
    matlab_avp,matlab_ins,matlab_kf,matlab_ins_pred_pos_llh = eng.test_SINS_dynamic_UKF_153_forpython(matlab_dynamic_pos,matlab_imu_i,matlab_kf,matlab_ins,nargout=4)

    # 获取INS预测的位置（融合前的位置）
    # ins_pred_pos_llh 只是一个3元素的位置向量 [lat, lon, h]，需要包装成10元素的avp格式
    # 先将MATLAB数组转换为numpy数组
    ins_pred_pos_llh_np = np.array(matlab_ins_pred_pos_llh).flatten()
    # 创建一个临时的avp数组，只填充位置信息，其他用零填充
    temp_avp_llh = [0, 0, 0, 0, 0, 0, ins_pred_pos_llh_np[0], ins_pred_pos_llh_np[1], ins_pred_pos_llh_np[2], 0]
    temp_avp_llh_matlab = matlab.double(temp_avp_llh)
    ins_pred_pos_xyz_matlab = eng.llh2xyz_subfun(temp_avp_llh_matlab)
    ins_pred_pos_xyz = np.array(ins_pred_pos_xyz_matlab).flatten()
    # 只取位置部分（索引6-8，对应x, y, z）
    ins_pred_pos_xyz = ins_pred_pos_xyz[6:9]
    
    # 计算观测残差（在XYZ坐标系下）
    # 观测残差实际上是：ins.pos - dynamic_pos (在MATLAB函数中计算，这里在XYZ下验证)
    obs_residual_xyz = ins_pred_pos_xyz - dynamic_pos_xyz

    # # 查看matlab_avp位置结果
    matlab_ukf_avp_xyz = eng.llh2xyz_subfun( matlab_avp )
    ukf_fused_avp_xyz = np.array(matlab_ukf_avp_xyz).flatten()
    
    # 获取当前时刻的真实值和期望值
    # 从四元数转换为姿态角（如果需要）或者使用上一时刻的姿态作为近似
    # 暂时使用上一时刻的UKF姿态作为真实姿态的近似（因为RawData中没有直接的姿态角数据）
    if 'q' in RawData[data_num] and loop_index-1 < RawData[data_num]['q'].shape[0]:
        # 如果有四元数数据，可以从四元数转换，这里简化处理
        current_real_att = [0, 0, 0]  # 需要时可以添加四元数转欧拉角的转换
    else:
        current_real_att = [0, 0, 0]
    
    current_real_v = real_v[loop_index-1, 0:3] if loop_index-1 < len(real_v) else [0, 0, 0]
    current_real_p = real_p[loop_index-1, 0:3] if loop_index-1 < len(real_p) else [0, 0, 0]
    current_desire_v = desire_vs[loop_index-1, 0:3] if loop_index-1 < len(desire_vs) else [0, 0, 0]
    current_desire_p = desire_ps[loop_index-1, 0:3] if loop_index-1 < len(desire_ps) else [0, 0, 0]

    # 获取PWM控制信号（从inputdata中，features是['v', 'q', 'pwm']，所以PWM在索引7:11）
    if inputdata.shape[0] >= 11:
        current_pwm = inputdata[7:11].tolist()
    else:
        current_pwm = [0, 0, 0, 0]

    # 记录日志
    # 确保last_avp和matlab_avp是1维数组以便统一访问
    last_avp_flat = last_avp.flatten()
    matlab_avp_flat = np.array(matlab_avp).flatten()
    
        current_time = ts[loop_index-1] if loop_index-1 < len(ts) else loop_index * dt
    log_row = [
        loop_index, current_time,
        # UKF融合前状态（上一时刻）
        last_avp_flat[0], last_avp_flat[1], last_avp_flat[2],  # 姿态
        last_avp_flat[3], last_avp_flat[4], last_avp_flat[5],  # 速度
        last_avp_xyz[6,0], last_avp_xyz[7,0], last_avp_xyz[8,0],  # 位置
        # 动力学模型输入
        inputdata[0], inputdata[1], inputdata[2],  # 速度
        inputdata[3], inputdata[4], inputdata[5], inputdata[6],  # 四元数
        current_pwm[0] if len(current_pwm) > 0 else 0,
        current_pwm[1] if len(current_pwm) > 1 else 0,
        current_pwm[2] if len(current_pwm) > 2 else 0,
        current_pwm[3] if len(current_pwm) > 3 else 0,
        # 气动力相关
        neural_fa_xyz[0], neural_fa_xyz[1], neural_fa_xyz[2],  # 神经网络预测的气动力
        outputlabel[0,0], outputlabel[1,0], outputlabel[2,0],  # 计算的气动力
        dynamic_a[0,0], dynamic_a[1,0], dynamic_a[2,0],  # 动力学参数
        # 动力学模型输出
        dynamic_pos_xyz[0], dynamic_pos_xyz[1], dynamic_pos_xyz[2],  # 位置
        dynamic_vel_xyz[0], dynamic_vel_xyz[1], dynamic_vel_xyz[2],  # 速度
        dynamic_vdot_xyz[0], dynamic_vdot_xyz[1], dynamic_vdot_xyz[2],  # 加速度
        # 惯导预测（UKF融合前）
        ins_pred_pos_xyz[0], ins_pred_pos_xyz[1], ins_pred_pos_xyz[2],  # 位置
        last_avp_flat[0], last_avp_flat[1], last_avp_flat[2],  # 姿态（使用上一时刻的）
        last_avp_flat[3], last_avp_flat[4], last_avp_flat[5],  # 速度
        # 观测残差
        obs_residual_xyz[0], obs_residual_xyz[1], obs_residual_xyz[2],
        # UKF融合后状态
        matlab_avp_flat[0], matlab_avp_flat[1], matlab_avp_flat[2],  # 姿态
        matlab_avp_flat[3], matlab_avp_flat[4], matlab_avp_flat[5],  # 速度
        ukf_fused_avp_xyz[0], ukf_fused_avp_xyz[1], ukf_fused_avp_xyz[2],  # 位置
        # 真实值
        current_real_att[0], current_real_att[1], current_real_att[2],
        current_real_v[0], current_real_v[1], current_real_v[2],
        current_real_p[0], current_real_p[1], current_real_p[2],
        # 期望值
        current_desire_v[0], current_desire_v[1], current_desire_v[2],
        current_desire_p[0], current_desire_p[1], current_desire_p[2],
        # 其他
        T_sp, hover_throttle
    ]
    log_writer.writerow(log_row)
    
    # 每100个循环输出一次进度
    if (loop_index - first_index) % 100 == 0:
        log_csv.flush()  # 确保数据写入文件
            print(f"已处理 {loop_index - first_index}/{actual_loops} 个循环，日志已保存")

    last_last_avp = last_avp.copy()
    last_avp = np.array(matlab_avp)
        ukf_avps_index = loop_index - first_index
        if ukf_avps_index < actual_loops:
            ukf_avps[:,ukf_avps_index] = last_avp

# 关闭日志文件
log_csv.close()
print(f"日志记录完成，文件已保存: {log_file}")
    print(f"共记录了 {actual_loops} 个循环的数据")

# 将ukf_avps转换到xyz
ukf_avp_size = ukf_avps.shape
pure_avp_size = pure_avps.shape
ukf_avps_xyz = np.empty((9,ukf_avp_size[1]))
pure_avps_xyz = np.empty((9,pure_avp_size[1]))
for i in range(ukf_avp_size[1]):
    ukf_avps_xyz[:,i] = np.array(eng.llh2xyz_subfun(matlab.double( ukf_avps[:,i].tolist() ) ))
for i in range(pure_avp_size[1]):
    pure_avps_xyz[:,i] = np.array(eng.llh2xyz_subfun(matlab.double( pure_avps[:,i].tolist() ) ))

    # 参考结果
    # 动态计算数据长度
    end_index = adapt_end_index + actual_loops
    y_real_data_total = np.empty((10, actual_loops))
    y_real_data_total[3:6, :] = real_v_row[:, adapt_end_index:end_index]
    y_real_data_total[6:9, :] = real_p_row[:, adapt_end_index:end_index]
    
    # ========== 误差计算和RMSE统计 ==========
    # 获取真实值的时间轴（从adapt_end_index开始）
    real_time = ts[adapt_end_index:end_index]  # 真实值的时间轴

    # UKF时间轴和数据
    ukf_time = ukf_avps[9, :]  # UKF时间数据
    ukf_vel_xyz = ukf_avps_xyz[3:6, :]  # UKF速度 (3 x N)
    ukf_pos_xyz = ukf_avps_xyz[6:9, :]  # UKF位置 (3 x N)

    # 纯惯导数据设置
    # 纯惯导的时间步长从MATLAB函数返回的imu数据中获取（通常是0.04秒，因为双子样）
    pure_ins_time_step = 0.04  # 时间步长（秒）
    pure_ins_start_time = start_time_sec  # 起始时间
    # 纯惯导数据长度
    actual_pure_avps_length = pure_avps_xyz.shape[1]
    pure_ins_data_length = actual_pure_avps_length
    
    # 纯惯导时间轴
    x2_data = np.arange(pure_ins_start_time, pure_ins_start_time + pure_ins_data_length * pure_ins_time_step, pure_ins_time_step)
    
    # 纯惯导时间轴和数据
    pure_ins_time = x2_data[:pure_ins_data_length]  # 纯惯导时间轴
    pure_ins_vel_xyz = pure_avps_xyz[3:6, 0:pure_ins_data_length]  # 纯惯导速度 (3 x N)
    pure_ins_pos_xyz = pure_avps_xyz[6:9, 0:pure_ins_data_length]  # 纯惯导位置 (3 x N)

    # 真实值数据（速度在索引3:6，位置在索引6:9）
    real_vel_xyz = y_real_data_total[3:6, :]  # 真实速度 (3 x N)
    real_pos_xyz = y_real_data_total[6:9, :]  # 真实位置 (3 x N)
    
    def calculate_rmse(calc_time, calc_data, real_time, real_data):
        """
        根据时间匹配计算值和真实值，计算RMSE
        
        参数:
            calc_time: 计算值的时间轴 (1D array)
            calc_data: 计算值数据 (3 x N 或 1 x N)
            real_time: 真实值的时间轴 (1D array)
            real_data: 真实值数据 (3 x N 或 1 x N)
        
        返回:
            rmse: RMSE值（如果是3维数据，返回3个分量的RMSE）
            matched_calc: 匹配后的计算值
            matched_real: 匹配后的真实值
            matched_time: 匹配后的时间轴
        """
        # 找到共同的时间范围
        time_min = max(calc_time.min(), real_time.min())
        time_max = min(calc_time.max(), real_time.max())
        
        # 创建匹配的时间轴（使用真实值的时间轴作为基准，在共同时间范围内）
        time_mask = (real_time >= time_min) & (real_time <= time_max)
        matched_time = real_time[time_mask]
        real_indices = np.where(time_mask)[0]
        
        # 确保calc_data是2D数组
        if calc_data.ndim == 1:
            calc_data = calc_data.reshape(1, -1)
        if real_data.ndim == 1:
            real_data = real_data.reshape(1, -1)
        
        # 对每个维度进行插值
        matched_calc = np.zeros((calc_data.shape[0], len(matched_time)))
        matched_real = np.zeros((real_data.shape[0], len(matched_time)))
        
        for i in range(calc_data.shape[0]):
            # 插值计算值到匹配时间轴
            interp_func = interp1d(calc_time, calc_data[i, :], 
                                  kind='linear', bounds_error=False, fill_value='extrapolate')
            matched_calc[i, :] = interp_func(matched_time)
            
            # 提取对应的真实值（matched_time 和 real_indices 长度应该一致）
            matched_real[i, :] = real_data[i, real_indices]
        
        # 计算误差
        error = matched_calc - matched_real
        
        # 计算RMSE（每个分量的RMSE）
        if error.shape[0] == 1:
            rmse = np.sqrt(np.mean(error**2, axis=1))[0]
        else:
            rmse = np.sqrt(np.mean(error**2, axis=1))  # 每个分量的RMSE
        
        return rmse, matched_calc, matched_real, matched_time

    # 计算UKF的RMSE
    print("\n========== UKF融合解误差分析 ==========")
    ukf_vel_rmse, ukf_vel_calc, ukf_vel_real, ukf_vel_time = calculate_rmse(
        ukf_time, ukf_vel_xyz, real_time, real_vel_xyz)
    ukf_pos_rmse, ukf_pos_calc, ukf_pos_real, ukf_pos_time = calculate_rmse(
        ukf_time, ukf_pos_xyz, real_time, real_pos_xyz)

    print("速度RMSE (m/s):")
    print(f"  东向: {ukf_vel_rmse[0]:.6f}")
    print(f"  北向: {ukf_vel_rmse[1]:.6f}")
    print(f"  天向: {ukf_vel_rmse[2]:.6f}")
    print(f"  总体: {np.sqrt(np.mean(ukf_vel_rmse**2)):.6f}")
    
    print("\n位置RMSE (m):")
    print(f"  东向: {ukf_pos_rmse[0]:.6f}")
    print(f"  北向: {ukf_pos_rmse[1]:.6f}")
    print(f"  天向: {ukf_pos_rmse[2]:.6f}")
    print(f"  总体: {np.sqrt(np.mean(ukf_pos_rmse**2)):.6f}")
    
    # 计算纯惯导的RMSE
    print("\n========== 纯惯导解误差分析 ==========")
    pure_vel_rmse, pure_vel_calc, pure_vel_real, pure_vel_time = calculate_rmse(
        pure_ins_time, pure_ins_vel_xyz, real_time, real_vel_xyz)
    pure_pos_rmse, pure_pos_calc, pure_pos_real, pure_pos_time = calculate_rmse(
        pure_ins_time, pure_ins_pos_xyz, real_time, real_pos_xyz)
    
    print("速度RMSE (m/s):")
    print(f"  东向: {pure_vel_rmse[0]:.6f}")
    print(f"  北向: {pure_vel_rmse[1]:.6f}")
    print(f"  天向: {pure_vel_rmse[2]:.6f}")
    print(f"  总体: {np.sqrt(np.mean(pure_vel_rmse**2)):.6f}")
    
    print("\n位置RMSE (m):")
    print(f"  东向: {pure_pos_rmse[0]:.6f}")
    print(f"  北向: {pure_pos_rmse[1]:.6f}")
    print(f"  天向: {pure_pos_rmse[2]:.6f}")
    print(f"  总体: {np.sqrt(np.mean(pure_pos_rmse**2)):.6f}")
    
    # 计算气动力RMSE
    print("\n========== 气动力误差分析 ==========")
    if len(neural_fa_collection) > 0 and len(real_fa_collection) > 0:
        neural_fa_array = np.array(neural_fa_collection)  # N x 3
        real_fa_array = np.array(real_fa_collection)      # N x 3
        
        # 计算误差
        fa_error = neural_fa_array - real_fa_array
        
        # 计算RMSE（每个分量）
        fa_rmse = np.sqrt(np.mean(fa_error**2, axis=0))  # 3个分量的RMSE
        fa_rmse_total = np.sqrt(np.mean(fa_error**2))     # 总体RMSE
        
        print("气动力RMSE (N):")
        print(f"  X方向: {fa_rmse[0]:.6f}")
        print(f"  Y方向: {fa_rmse[1]:.6f}")
        print(f"  Z方向: {fa_rmse[2]:.6f}")
        print(f"  总体: {fa_rmse_total:.6f}")
    else:
        print("警告: 没有收集到足够的气动力数据")
        fa_rmse = np.array([0.0, 0.0, 0.0])
        fa_rmse_total = 0.0
    
    # ========== 记录实验参数和指标到MLflow ==========
    # 准备模型参数
    model_params = {
        'model_dataset': dataset,
        'model_dim_a': dim_a,
        'model_features': '-'.join(features),
        'model_name': modelname,
        'model_stopping_epoch': stopping_epoch,
        'model_file': f"{modelname}-epoch-{stopping_epoch}.pth"
    }

    # 准备滤波参数（使用实际配置的值）
    filter_params = {
        'filter_solve_type': solver_type,  # 1=KF, 2=PF
        'filter_numPar': numPar,
        'filter_lambda1': lambda1,
        'filter_R': r0,  # 从配置中读取的r0值
        'filter_Q': q0,  # 从配置中读取的q0值
    }

    # 准备UKF参数
    ukf_params = {}
    if ukf_Qk is not None:
        ukf_params['Qk'] = ukf_Qk
    if ukf_Rk is not None:
        ukf_params['Rk'] = ukf_Rk
    if ukf_Pxk is not None:
        ukf_params['Pxk'] = ukf_Pxk
    
    # 准备数据集参数
    dataset_params = {
        'dataset_folder': validation_config['dataset_folder'],
        'validation_file': validation_file,
        'adapt_end_index': adapt_end_index,
        'adapt_duration': adapt_duration,
        'validation_length': actual_loops,
    }
    
    # 准备指标
    metrics = {
    # UKF速度RMSE
    'ukf_vel_rmse_east': float(ukf_vel_rmse[0]),
    'ukf_vel_rmse_north': float(ukf_vel_rmse[1]),
    'ukf_vel_rmse_up': float(ukf_vel_rmse[2]),
    'ukf_vel_rmse_total': float(np.sqrt(np.mean(ukf_vel_rmse**2))),
    # UKF位置RMSE
    'ukf_pos_rmse_east': float(ukf_pos_rmse[0]),
    'ukf_pos_rmse_north': float(ukf_pos_rmse[1]),
    'ukf_pos_rmse_up': float(ukf_pos_rmse[2]),
    'ukf_pos_rmse_total': float(np.sqrt(np.mean(ukf_pos_rmse**2))),
    # 纯惯导速度RMSE
    'pure_ins_vel_rmse_east': float(pure_vel_rmse[0]),
    'pure_ins_vel_rmse_north': float(pure_vel_rmse[1]),
    'pure_ins_vel_rmse_up': float(pure_vel_rmse[2]),
    'pure_ins_vel_rmse_total': float(np.sqrt(np.mean(pure_vel_rmse**2))),
    # 纯惯导位置RMSE
    'pure_ins_pos_rmse_east': float(pure_pos_rmse[0]),
    'pure_ins_pos_rmse_north': float(pure_pos_rmse[1]),
    'pure_ins_pos_rmse_up': float(pure_pos_rmse[2]),
    'pure_ins_pos_rmse_total': float(np.sqrt(np.mean(pure_pos_rmse**2))),
    # 气动力RMSE
    'fa_rmse_x': float(fa_rmse[0]),
    'fa_rmse_y': float(fa_rmse[1]),
    'fa_rmse_z': float(fa_rmse[2]),
        'fa_rmse_total': float(fa_rmse_total),
    }
    
    # 记录参数和指标
    log_experiment_params(
        model_params=model_params,
        filter_params=filter_params,
        ukf_params=ukf_params,
        dataset_params=dataset_params
    )
    log_experiment_metrics(metrics)
    
    # 记录模型文件
    model_file_path = os.path.join(project_root, 'models', f"{modelname}-epoch-{stopping_epoch}.pth")
    log_model_file(model_file_path)
    
    print("\n========== MLflow记录完成 ==========")
    
    # 创建3行3列的子图网格，figsize控制画布大小（宽15，高12）
    # Y轴标签：姿态、速度、位置（速度单位使用LaTeX格式显示上标）
    Ylabels = [
        "姿态x[°]", "姿态y[°]", "姿态z[°]",
        "东向速度 /(m·$s^{-1}$)", "北向速度 /(m·$s^{-1}$)", "天向速度 /(m·$s^{-1}$)",
        "东向位置 /m", "北向位置 /m", "天向位置 /m"
    ]
    # 创建3x3的子图布局
    plt.figure(figsize=(15, 12))  # 整体画布大小

# 使用各自的时间轴（不进行对齐）
    x_data = ukf_avps[9, :]  # UKF时间数据

for i in range(9):
        # 创建子图（3行3列，第i+1个子图）
        plt.subplot(3, 3, i + 1)
    
    # 姿态 速度 位置绘图 - 使用各自的时间轴和数据
        y_data = ukf_avps_xyz[i, :]
        # 纯惯导数据：截取到指定长度
    y2_data = pure_avps_xyz[i, 0:pure_ins_data_length]
    
    # 确保时间轴和数据长度一致
    actual_pure_length = min(len(x2_data), len(y2_data))
    x2_data_plot = x2_data[:actual_pure_length]
    y2_data_plot = y2_data[:actual_pure_length]
    
    # UKF结果 - 实线，蓝色
    plt.plot(x_data, y_data,
             linestyle='-',  # 实线
             color='#2E86AB',  # 蓝色
             linewidth=2.0,  # 线宽
             alpha=0.9,  # 透明度
             label="UKF融合解")  # 标签
    
    # Pure INS结果 - 虚线，红色
    plt.plot(x2_data_plot, y2_data_plot,
             linestyle='--',  # 虚线
             color='#F24236',  # 红色
             linewidth=2.0,  # 线宽
             alpha=0.9,  # 透明度
             label="纯惯导解")  # 标签
    
    # 真实值 - 点划线，绿色
    if i > 2 and i < 9:
            y_real_data = y_real_data_total[i, :]
        plt.plot(x_data, y_real_data,
                 linestyle='-.',  # 点划线
                 color='#06A77D',  # 绿色
                 linewidth=2.0,  # 线宽
                 alpha=0.9,  # 透明度
                 label="参考值")  # 标签

    # 子图标题和标签
    plt.xlabel('时间 /s', fontsize=10)
    # Y轴标签（速度单位中的上标使用LaTeX数学模式）
    plt.ylabel(Ylabels[i], fontsize=10)

    # 添加图例和网格
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3)

# 调整子图间距，避免重叠
plt.tight_layout()

# 显示图形
plt.show()

# （可选）保存图片到本地（分辨率300dpi，无白边）
    # plt.savefig(f"{validation_file.replace('.csv', '')}_result.png", dpi=300, bbox_inches='tight')
    plt.close('all')  # 关闭所有图形，避免内存积累
    
    # 结束当前MLflow运行
    end_run()
    print(f"实验记录已保存到MLflow: {validation_file}")

# 关闭MATLAB引擎（在所有实验完成后）
if 'eng' in locals():
eng.quit()
print("\n所有实验完成！")
