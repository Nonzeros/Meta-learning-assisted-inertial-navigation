"""
实验运行器：处理单个CSV文件的实验逻辑
"""
import os
import sys
import numpy as np
import torch
import csv
import matlab.engine
import traceback
from datetime import datetime
from scipy.interpolate import interp1d

import utils
import mlmodel
from intelligent_dynamic_module import intelligent_dynamic_module
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'visualization'))
from mlflow_utils import (
    log_experiment_params, 
    log_experiment_metrics,
    log_model_file
)
import mlflow


def run_single_experiment(csv_filename, project_root, filter_config, eng, 
                         adapt_end_index=100, glv_init_code=None):
    """
    处理单个CSV文件的实验
    
    参数:
        csv_filename: CSV文件名（不含路径）
        project_root: 项目根目录
        filter_config: 滤波器配置
        eng: MATLAB引擎实例
        adapt_end_index: 适应阶段结束索引
        glv_init_code: MATLAB glv初始化代码
    
    返回:
        success: 是否成功
        error_msg: 错误信息（如果失败）
        results: 结果字典（包含RMSE等指标）
    """
    try:
        # 1. 数据提取和准备
        from data_extractor import extract_csv_to_excel, get_excel_filename_from_csv
        
        # 生成Excel文件名
        excel_filename = get_excel_filename_from_csv(csv_filename)
        csv_file_path = os.path.join(project_root, 'data', 'experiment2', csv_filename)
        excel_file_path = os.path.join(project_root, 'ProcessedData', excel_filename)
        
        # 提取CSV数据到Excel
        success, error_msg = extract_csv_to_excel(csv_file_path, excel_file_path)
        if not success:
            return False, f"数据提取失败: {error_msg}", None
        
        # 2. 加载数据
        dataset_folder = os.path.join(project_root, 'data', 'experiment2')
        RawData = utils.load_data(dataset_folder, expnames=[csv_filename.replace('.csv', '')])
        if not RawData:
            return False, f"无法加载数据文件: {csv_filename}", None
        
        data_num = 0  # 单个文件时，索引为0
        features = ['v', 'q', 'pwm']
        label = 'fa'
        Data = utils.format_data(RawData, features=features, output=label)
        
        # 获取数据
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
        ts = RawData[data_num]['t']
        real_q = RawData[data_num]['q']  # 真实姿态四元数
        data = Data[data_num]
        
        # 3. 参数设置
        solver_type = filter_config['filter']['type']
        numPar = filter_config['filter']['numPar']
        model_name = filter_config['model']['name']
        dim_a = filter_config['model']['dim_a']
        
        options = {
            'dim_a': dim_a,
            'loss_type': 'crossentropy-loss',
            'solve_type': solver_type,
            'lambda1': 0.1,
            'numPar': numPar,
            'filter_config': filter_config
        }
        
        # 4. MATLAB设置
        if glv_init_code:
            eng.eval(glv_init_code, nargout=0)
        
        processed_data_path = os.path.join(project_root, 'ProcessedData')
        eng.cd(processed_data_path.replace('\\', '/'), nargout=0)
        
        # 调用MATLAB函数，传入Excel文件名
        excel_filename_matlab = excel_filename  # 只有文件名，因为已经cd到ProcessedData目录
        matlab_imu, matlab_avp0 = eng.av2imu_main3(adapt_end_index+1, excel_filename_matlab, nargout=2)
        
        # 5. 初始化
        last_avp = np.array(matlab_avp0)
        start_time_sec = adapt_end_index * 0.02
        last_avp = np.append(last_avp, start_time_sec)
        last_avp = last_avp.reshape((1,10))
        last_avp[0,3:6] = real_v[adapt_end_index,0:3]
        
        last_avp_xyz = last_avp.copy()
        last_avp_xyz[0,6:9] = real_p[adapt_end_index, 0:3]
        last_avp_llh_matlab = eng.xyz2llh_subfun(matlab.double(last_avp_xyz.tolist()))
        last_avp = np.array(last_avp_llh_matlab)
        
        # 纯惯导求解
        matlab_pure_avps = eng.pure_ins_solve(matlab_imu, matlab.double(last_avp.tolist()), nargout=1)
        pure_avps = np.array(matlab_pure_avps).T
        
        # 提前转换纯惯导数据到XYZ坐标系（用于日志记录）
        pure_avp_size = pure_avps.shape
        pure_avps_xyz_log = np.empty((9, pure_avp_size[1]))
        for i in range(pure_avp_size[1]):
            pure_avps_xyz_log[:,i] = np.array(eng.llh2xyz_subfun(matlab.double(pure_avps[:,i].tolist())))
        
        avp0_change = last_avp
        imu = np.array(matlab_imu)
        
        # UKF初始化
        matlab_kf, matlab_ins = eng.SINS_dynamic_UKF153_init(matlab.double(avp0_change.tolist()), nargout=2)
        
        # 获取UKF参数
        try:
            ukf_Qk = np.array(eng.getfield(matlab_kf, 'Qk')) if 'Qk' in str(matlab_kf) else None
            ukf_Rk = np.array(eng.getfield(matlab_kf, 'Rk')) if 'Rk' in str(matlab_kf) else None
            ukf_Pxk = np.array(eng.getfield(matlab_kf, 'Pxk')) if 'Pxk' in str(matlab_kf) else None
        except:
            ukf_Qk = None
            ukf_Rk = None
            ukf_Pxk = None
        
        # 6. 动力学模型初始化
        final_model = mlmodel.load_model(modelname=model_name)
        lam = 0.1
        
        adaptinput = data.X[0:adapt_end_index, :]
        adaptlabel = data.Y[0:adapt_end_index, :]
        X = torch.from_numpy(adaptinput)
        Y = torch.from_numpy(adaptlabel)
        Phi = final_model.phi(X)
        Phi_T = Phi.transpose(0, 1)
        A = torch.inverse(torch.mm(Phi_T, Phi) + lam * torch.eye(dim_a))
        a0 = torch.mm(torch.mm(A, Phi_T), Y)
        
        dynamic_a = a0.detach().numpy()
        dynamic_P = np.full((3, dim_a), 0.1)
        
        # 7. 循环计算
        # 根据实际数据长度自适应确定循环范围
        # imu数组的长度决定了实际可以处理的循环次数
        imu_length = imu.shape[0]
        validation_data_length = len(ts) - adapt_end_index
        # 循环次数取imu长度和validation_data_length的较小值，确保不会越界
        loops = min(imu_length, validation_data_length)
        first_index = adapt_end_index
        
        ukf_avps = np.empty((10, loops + 1))
        ukf_avps[:,0] = last_avp
        last_last_avp = last_avp.copy()
        
        neural_fa_collection = []
        real_fa_collection = []
        fa_time_collection = []
        
        # 初始化日志文件
        log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(project_root, "navigation_logs")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # 在日志文件名中添加CSV文件名标识
        csv_base_name = csv_filename.replace('.csv', '')
        log_file = os.path.join(log_dir, f"navigation_log_{log_timestamp}_{csv_base_name}.csv")
        
        log_header = [
            'loop_index', 'time',
            'ukf_prev_att_x', 'ukf_prev_att_y', 'ukf_prev_att_z',
            'ukf_prev_vx', 'ukf_prev_vy', 'ukf_prev_vz',
            'ukf_prev_px', 'ukf_prev_py', 'ukf_prev_pz',
            'dynamic_input_vx', 'dynamic_input_vy', 'dynamic_input_vz',
            'dynamic_input_q0', 'dynamic_input_q1', 'dynamic_input_q2', 'dynamic_input_q3',
            'pwm0', 'pwm1', 'pwm2', 'pwm3',
            'neural_fa_x', 'neural_fa_y', 'neural_fa_z',
            'calculated_fa_x', 'calculated_fa_y', 'calculated_fa_z',
            'dynamic_a_0', 'dynamic_a_1', 'dynamic_a_2',
            'dynamic_pos_x', 'dynamic_pos_y', 'dynamic_pos_z',
            'dynamic_vel_x', 'dynamic_vel_y', 'dynamic_vel_z',
            'dynamic_vdot_x', 'dynamic_vdot_y', 'dynamic_vdot_z',
            'ins_pred_pos_x', 'ins_pred_pos_y', 'ins_pred_pos_z',
            'ins_pred_att_x', 'ins_pred_att_y', 'ins_pred_att_z',
            'ins_pred_vx', 'ins_pred_vy', 'ins_pred_vz',
            'obs_residual_x', 'obs_residual_y', 'obs_residual_z',
            'ukf_fused_att_x', 'ukf_fused_att_y', 'ukf_fused_att_z',
            'ukf_fused_vx', 'ukf_fused_vy', 'ukf_fused_vz',
            'ukf_fused_px', 'ukf_fused_py', 'ukf_fused_pz',
            'pure_ins_att_x', 'pure_ins_att_y', 'pure_ins_att_z',
            'pure_ins_vx', 'pure_ins_vy', 'pure_ins_vz',
            'pure_ins_px', 'pure_ins_py', 'pure_ins_pz',
            'real_att_x', 'real_att_y', 'real_att_z',
            'real_vx', 'real_vy', 'real_vz',
            'real_px', 'real_py', 'real_pz',
            'desire_vx', 'desire_vy', 'desire_vz',
            'desire_px', 'desire_py', 'desire_pz',
            'T_sp', 'hover_throttle'
        ]
        
        log_csv = open(log_file, 'w', newline='', encoding='utf-8')
        log_writer = csv.writer(log_csv)
        log_writer.writerow(log_header)
        
        # 主循环（与原代码相同）
        # 循环范围：从adapt_end_index开始，执行loops次（根据imu数组长度自适应确定）
        for loop_index in range(first_index, first_index + loops):
            # 动力学取数据
            # 添加边界检查，确保不会越界
            if loop_index - 1 >= len(data.X):
                break  # 如果超出数据范围，提前退出循环
            inputdata = data.X[loop_index-1,:].copy()
            inputdata[0:3] = last_avp[0,3:6]
            matlab_qua = eng.a2qua_subfun(matlab.double(last_avp[0,0:3].tolist()), nargout=1)
            qua_array = np.array(matlab_qua).flatten()
            inputdata[3:7] = qua_array
            
            # 计算标签
            last_last_avp_reshaped = last_last_avp.reshape((1, 10))
            middle = (last_avp - last_last_avp_reshaped) / 0.02
            vdot_mins1 = middle[:,3:6].reshape(3,1)
            att = last_avp[0,0:3]
            matlab_Ri = eng.a2mat_subfun(matlab.double(att.tolist()), nargout=1)
            Ri = np.array(matlab_Ri)
            hover_throttle = hover_throttles[loop_index-1]
            T_sp = T_sps[loop_index-1]
            m0 = 2.6
            g_ = 9.8
            g = np.array([0,0,-g_]).reshape((3,1))
            fT = np.array([0, 0, float(T_sp / hover_throttle) * 9.8 * m0]).reshape((3, 1))
            outputlabel = m0 * vdot_mins1 - m0 * g - Ri @ fT
            
            desire_v = desire_vs[loop_index-1,:].reshape((3,1))
            desire_p = desire_ps[loop_index-1,:].reshape((3,1))
            
            last_avp = last_avp.reshape((10,1))
            last_v = last_avp[3:6,:]
            delta_v = desire_v - last_v
            
            # 动力学更新
            last_avp = last_avp.reshape((1,10))
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
            dynamic_pos, dynamic_P, dynamic_a, px, pw, neural_fa, dynamic_vel, dynamic_vdot = \
                intelligent_dynamic_module(inputdata, outputlabel, delta_v, delta_p, dynamic_a, dynamic_P, dim_a,
                                         hover_throttle, T_sp, Ri, vt_minus1, pt_minus1, px, pw, options)
            
            dynamic_pos_xyz = dynamic_pos.flatten()
            dynamic_vel_xyz = dynamic_vel.flatten()
            dynamic_vdot_xyz = dynamic_vdot.flatten()
            neural_fa_xyz = neural_fa.flatten()
            
            if loop_index-1 < len(real_fas):
                neural_fa_collection.append(neural_fa_xyz)
                real_fa_collection.append(real_fas[loop_index-1, :])
                fa_time_collection.append(ts[loop_index-1] if loop_index-1 < len(ts) else loop_index * 0.02)
            
            # UKF更新
            imu_index = loop_index - first_index
            dynamic_pos_list = dynamic_pos.tolist()
            matlab_dynamic_pos = matlab.double(dynamic_pos_list)
            matlab_imu_i = matlab.double(imu[imu_index,:].tolist())
            
            matlab_avp, matlab_ins, matlab_kf, matlab_ins_pred_pos_llh = \
                eng.test_SINS_dynamic_UKF_153_forpython(matlab_dynamic_pos, matlab_imu_i, matlab_kf, matlab_ins, nargout=4)
            
            # 处理INS预测位置
            ins_pred_pos_llh_np = np.array(matlab_ins_pred_pos_llh).flatten()
            temp_avp_llh = [0, 0, 0, 0, 0, 0, ins_pred_pos_llh_np[0], ins_pred_pos_llh_np[1], ins_pred_pos_llh_np[2], 0]
            temp_avp_llh_matlab = matlab.double(temp_avp_llh)
            ins_pred_pos_xyz_matlab = eng.llh2xyz_subfun(temp_avp_llh_matlab)
            ins_pred_pos_xyz = np.array(ins_pred_pos_xyz_matlab).flatten()[6:9]
            
            obs_residual_xyz = ins_pred_pos_xyz - dynamic_pos_xyz
            
            # UKF融合结果
            matlab_ukf_avp_xyz = eng.llh2xyz_subfun(matlab_avp)
            ukf_fused_avp_xyz = np.array(matlab_ukf_avp_xyz).flatten()
            ukf_fused_att_xyz = ukf_fused_avp_xyz[0:3]
            ukf_fused_vel_xyz = ukf_fused_avp_xyz[3:6]
            ukf_fused_pos_xyz = ukf_fused_avp_xyz[6:9]
            
            # INS预测完整状态
            last_avp_for_xyz = last_avp.copy()
            if last_avp_for_xyz.ndim > 1 and last_avp_for_xyz.shape[0] > 1:
                last_avp_for_xyz = last_avp_for_xyz.reshape((1, -1))
            elif last_avp_for_xyz.ndim == 1:
                last_avp_for_xyz = last_avp_for_xyz.reshape((1, -1))
            
            last_avp_xyz_full = eng.llh2xyz_subfun(matlab.double(last_avp_for_xyz.tolist()))
            last_avp_xyz_full = np.array(last_avp_xyz_full).flatten()
            ins_pred_att_xyz = last_avp_xyz_full[0:3]
            ins_pred_vel_xyz = last_avp_xyz_full[3:6]
            
            # 真实值
            # 从四元数转换为欧拉角
            if loop_index-1 < len(real_q):
                real_q_current = real_q[loop_index-1, :]  # 获取当前时刻的四元数
                matlab_real_q = matlab.double(real_q_current.tolist())
                real_att_rad = np.array(eng.q2att(matlab_real_q)).flatten()  # 转换为欧拉角（弧度）
                current_real_att = real_att_rad * 180.0 / np.pi  # 转换为度
            else:
                current_real_att = [0, 0, 0]
            
            current_real_v = real_v[loop_index-1, 0:3] if loop_index-1 < len(real_v) else [0, 0, 0]
            current_real_p = real_p[loop_index-1, 0:3] if loop_index-1 < len(real_p) else [0, 0, 0]
            current_desire_v = desire_vs[loop_index-1, 0:3] if loop_index-1 < len(desire_vs) else [0, 0, 0]
            current_desire_p = desire_ps[loop_index-1, 0:3] if loop_index-1 < len(desire_ps) else [0, 0, 0]
            
            # 纯惯导数据（根据时间索引匹配）
            # 纯惯导从adapt_end_index开始，所以索引需要调整
            # 注意：纯惯导可能使用双子样算法，输出频率可能是输入频率的一半
            # 因此需要根据实际数据长度进行索引匹配
            pure_ins_index = loop_index - first_index
            pure_avps_length = pure_avps_xyz_log.shape[1]
            
            # 如果纯惯导数据长度小于循环长度，可能需要插值或使用最近的索引
            if pure_avps_length < validation_data_length:
                # 使用比例索引：将循环索引映射到纯惯导数据索引
                # 假设纯惯导输出频率是循环频率的一半
                scaled_index = int(pure_ins_index * pure_avps_length / validation_data_length)
                scaled_index = min(scaled_index, pure_avps_length - 1)
                if scaled_index >= 0 and scaled_index < pure_avps_length:
                    pure_ins_att_xyz = pure_avps_xyz_log[0:3, scaled_index] * 180.0 / np.pi  # 转换为度
                    pure_ins_vel_xyz = pure_avps_xyz_log[3:6, scaled_index]
                    pure_ins_pos_xyz = pure_avps_xyz_log[6:9, scaled_index]
                else:
                    pure_ins_att_xyz = [0, 0, 0]
                    pure_ins_vel_xyz = [0, 0, 0]
                    pure_ins_pos_xyz = [0, 0, 0]
            else:
                # 如果纯惯导数据长度大于等于循环长度，直接使用索引
                if pure_ins_index < pure_avps_length:
                    pure_ins_att_xyz = pure_avps_xyz_log[0:3, pure_ins_index] * 180.0 / np.pi  # 转换为度
                    pure_ins_vel_xyz = pure_avps_xyz_log[3:6, pure_ins_index]
                    pure_ins_pos_xyz = pure_avps_xyz_log[6:9, pure_ins_index]
                else:
                    # 如果索引超出范围，使用最后一个数据点
                    last_index = pure_avps_length - 1
                    pure_ins_att_xyz = pure_avps_xyz_log[0:3, last_index] * 180.0 / np.pi  # 转换为度
                    pure_ins_vel_xyz = pure_avps_xyz_log[3:6, last_index]
                    pure_ins_pos_xyz = pure_avps_xyz_log[6:9, last_index]
            
            current_pwm = inputdata[7:11].tolist() if inputdata.shape[0] >= 11 else [0, 0, 0, 0]
            
            # 记录日志
            last_avp_flat = last_avp.flatten()
            current_time = ts[loop_index-1] if loop_index-1 < len(ts) else loop_index * 0.02
            log_row = [
                loop_index, current_time,
                last_avp_flat[0], last_avp_flat[1], last_avp_flat[2],
                last_avp_flat[3], last_avp_flat[4], last_avp_flat[5],
                last_avp_xyz[6,0], last_avp_xyz[7,0], last_avp_xyz[8,0],
                inputdata[0], inputdata[1], inputdata[2],
                inputdata[3], inputdata[4], inputdata[5], inputdata[6],
                current_pwm[0] if len(current_pwm) > 0 else 0,
                current_pwm[1] if len(current_pwm) > 1 else 0,
                current_pwm[2] if len(current_pwm) > 2 else 0,
                current_pwm[3] if len(current_pwm) > 3 else 0,
                neural_fa_xyz[0], neural_fa_xyz[1], neural_fa_xyz[2],
                outputlabel[0,0], outputlabel[1,0], outputlabel[2,0],
                dynamic_a[0,0], dynamic_a[1,0], dynamic_a[2,0],
                dynamic_pos_xyz[0], dynamic_pos_xyz[1], dynamic_pos_xyz[2],
                dynamic_vel_xyz[0], dynamic_vel_xyz[1], dynamic_vel_xyz[2],
                dynamic_vdot_xyz[0], dynamic_vdot_xyz[1], dynamic_vdot_xyz[2],
                ins_pred_pos_xyz[0], ins_pred_pos_xyz[1], ins_pred_pos_xyz[2],
                ins_pred_att_xyz[0], ins_pred_att_xyz[1], ins_pred_att_xyz[2],
                ins_pred_vel_xyz[0], ins_pred_vel_xyz[1], ins_pred_vel_xyz[2],
                obs_residual_xyz[0], obs_residual_xyz[1], obs_residual_xyz[2],
                ukf_fused_att_xyz[0], ukf_fused_att_xyz[1], ukf_fused_att_xyz[2],
                ukf_fused_vel_xyz[0], ukf_fused_vel_xyz[1], ukf_fused_vel_xyz[2],
                ukf_fused_pos_xyz[0], ukf_fused_pos_xyz[1], ukf_fused_pos_xyz[2],
                pure_ins_att_xyz[0], pure_ins_att_xyz[1], pure_ins_att_xyz[2],
                pure_ins_vel_xyz[0], pure_ins_vel_xyz[1], pure_ins_vel_xyz[2],
                pure_ins_pos_xyz[0], pure_ins_pos_xyz[1], pure_ins_pos_xyz[2],
                current_real_att[0], current_real_att[1], current_real_att[2],
                current_real_v[0], current_real_v[1], current_real_v[2],
                current_real_p[0], current_real_p[1], current_real_p[2],
                current_desire_v[0], current_desire_v[1], current_desire_v[2],
                current_desire_p[0], current_desire_p[1], current_desire_p[2],
                T_sp, hover_throttle
            ]
            log_writer.writerow(log_row)
            
            if (loop_index - first_index) % 100 == 0:
                log_csv.flush()
            
            last_last_avp = last_avp.copy()
            last_avp = np.array(matlab_avp)
            ukf_avps_index = loop_index - first_index + 1
            if ukf_avps_index < ukf_avps.shape[1]:
                ukf_avps[:,ukf_avps_index] = last_avp
        
        log_csv.close()
        
        # 8. 计算RMSE（与原代码相同）
        # 转换到xyz
        ukf_avp_size = ukf_avps.shape
        pure_avp_size = pure_avps.shape
        ukf_avps_xyz = np.empty((9,ukf_avp_size[1]))
        pure_avps_xyz = np.empty((9,pure_avp_size[1]))
        for i in range(ukf_avp_size[1]):
            ukf_avps_xyz[:,i] = np.array(eng.llh2xyz_subfun(matlab.double(ukf_avps[:,i].tolist())))
        for i in range(pure_avp_size[1]):
            pure_avps_xyz[:,i] = np.array(eng.llh2xyz_subfun(matlab.double(pure_avps[:,i].tolist())))
        
        validation_length = len(ts) - adapt_end_index
        y_real_data_total = np.empty((10, validation_length))
        y_real_data_total[3:6, 0:validation_length] = real_v_row[:, adapt_end_index:adapt_end_index+validation_length]
        y_real_data_total[6:9, 0:validation_length] = real_p_row[:, adapt_end_index:adapt_end_index+validation_length]
        
        real_time_start_index = adapt_end_index
        real_time_end_index = adapt_end_index + validation_length
        real_time = ts[real_time_start_index:real_time_end_index]
        
        exclude_last = min(5, ukf_avps_xyz.shape[1] - 1)
        ukf_time = ukf_avps[9, 0:-exclude_last if exclude_last > 0 else None]
        ukf_vel_xyz = ukf_avps_xyz[3:6, 0:-exclude_last if exclude_last > 0 else None]
        ukf_pos_xyz = ukf_avps_xyz[6:9, 0:-exclude_last if exclude_last > 0 else None]
        
        pure_ins_time_step = 0.04
        pure_ins_start_time = 2.0
        pure_ins_end_time = 40.0
        pure_ins_time_range = pure_ins_end_time - pure_ins_start_time
        pure_ins_target_points = int(pure_ins_time_range / pure_ins_time_step)
        actual_pure_avps_length = pure_avps_xyz.shape[1]
        pure_ins_data_length = min(pure_ins_target_points, actual_pure_avps_length)
        x2_data = np.arange(pure_ins_start_time, pure_ins_start_time + pure_ins_data_length * pure_ins_time_step, pure_ins_time_step)
        pure_ins_time = x2_data[:pure_ins_data_length]
        pure_ins_vel_xyz = pure_avps_xyz[3:6, 0:pure_ins_data_length]
        pure_ins_pos_xyz = pure_avps_xyz[6:9, 0:pure_ins_data_length]
        
        real_vel_xyz = y_real_data_total[3:6, 0:validation_length]
        real_pos_xyz = y_real_data_total[6:9, 0:validation_length]
        
        def calculate_rmse(calc_time, calc_data, real_time, real_data):
            time_min = max(calc_time.min(), real_time.min())
            time_max = min(calc_time.max(), real_time.max())
            time_mask = (real_time >= time_min) & (real_time <= time_max)
            matched_time = real_time[time_mask]
            real_indices = np.where(time_mask)[0]
            
            if calc_data.ndim == 1:
                calc_data = calc_data.reshape(1, -1)
            if real_data.ndim == 1:
                real_data = real_data.reshape(1, -1)
            
            matched_calc = np.zeros((calc_data.shape[0], len(matched_time)))
            matched_real = np.zeros((real_data.shape[0], len(matched_time)))
            
            for i in range(calc_data.shape[0]):
                interp_func = interp1d(calc_time, calc_data[i, :], 
                                      kind='linear', bounds_error=False, fill_value='extrapolate')
                matched_calc[i, :] = interp_func(matched_time)
                matched_real[i, :] = real_data[i, real_indices]
            
            error = matched_calc - matched_real
            if error.shape[0] == 1:
                rmse = np.sqrt(np.mean(error**2, axis=1))[0]
            else:
                rmse = np.sqrt(np.mean(error**2, axis=1))
            
            return rmse, matched_calc, matched_real, matched_time
        
        ukf_vel_rmse, _, _, _ = calculate_rmse(ukf_time, ukf_vel_xyz, real_time, real_vel_xyz)
        ukf_pos_rmse, _, _, _ = calculate_rmse(ukf_time, ukf_pos_xyz, real_time, real_pos_xyz)
        pure_vel_rmse, _, _, _ = calculate_rmse(pure_ins_time, pure_ins_vel_xyz, real_time, real_vel_xyz)
        pure_pos_rmse, _, _, _ = calculate_rmse(pure_ins_time, pure_ins_pos_xyz, real_time, real_pos_xyz)
        
        if len(neural_fa_collection) > 0 and len(real_fa_collection) > 0:
            neural_fa_array = np.array(neural_fa_collection)
            real_fa_array = np.array(real_fa_collection)
            fa_error = neural_fa_array - real_fa_array
            fa_rmse = np.sqrt(np.mean(fa_error**2, axis=0))
            fa_rmse_total = np.sqrt(np.mean(fa_error**2))
        else:
            fa_rmse = np.array([0.0, 0.0, 0.0])
            fa_rmse_total = 0.0
        
        # 9. 记录到MLflow
        model_params = {
            'model_dataset': 'neural-fly',
            'model_dim_a': dim_a,
            'model_features': '-'.join(features),
            'model_name': model_name,
            'model_file': f"{model_name}.pth"
        }
        
        filter_params = {
            'filter_solve_type': solver_type,
            'filter_numPar': numPar,
            'filter_lambda1': options['lambda1'],
            'filter_R': filter_config.get('particle_filter', {}).get('R', 0.1),
            'filter_Q': filter_config.get('kalman_filter', {}).get('Q', 0.1),
        }
        
        ukf_params = {}
        if ukf_Qk is not None:
            ukf_params['Qk'] = ukf_Qk
        if ukf_Rk is not None:
            ukf_params['Rk'] = ukf_Rk
        if ukf_Pxk is not None:
            ukf_params['Pxk'] = ukf_Pxk
        
        dataset_params = {
            'dataset_folder': dataset_folder,
            'adapt_end_index': adapt_end_index,
            'csv_filename': csv_filename
        }
        
        metrics = {
            'ukf_vel_rmse_east': float(ukf_vel_rmse[0]),
            'ukf_vel_rmse_north': float(ukf_vel_rmse[1]),
            'ukf_vel_rmse_up': float(ukf_vel_rmse[2]),
            'ukf_vel_rmse_total': float(np.sqrt(np.mean(ukf_vel_rmse**2))),
            'ukf_pos_rmse_east': float(ukf_pos_rmse[0]),
            'ukf_pos_rmse_north': float(ukf_pos_rmse[1]),
            'ukf_pos_rmse_up': float(ukf_pos_rmse[2]),
            'ukf_pos_rmse_total': float(np.sqrt(np.mean(ukf_pos_rmse**2))),
            'pure_ins_vel_rmse_east': float(pure_vel_rmse[0]),
            'pure_ins_vel_rmse_north': float(pure_vel_rmse[1]),
            'pure_ins_vel_rmse_up': float(pure_vel_rmse[2]),
            'pure_ins_vel_rmse_total': float(np.sqrt(np.mean(pure_vel_rmse**2))),
            'pure_ins_pos_rmse_east': float(pure_pos_rmse[0]),
            'pure_ins_pos_rmse_north': float(pure_pos_rmse[1]),
            'pure_ins_pos_rmse_up': float(pure_pos_rmse[2]),
            'pure_ins_pos_rmse_total': float(np.sqrt(np.mean(pure_pos_rmse**2))),
            'fa_rmse_x': float(fa_rmse[0]),
            'fa_rmse_y': float(fa_rmse[1]),
            'fa_rmse_z': float(fa_rmse[2]),
            'fa_rmse_total': float(fa_rmse_total),
        }
        
        log_experiment_params(model_params, filter_params, ukf_params, dataset_params)
        log_experiment_metrics(metrics)
        
        model_file_path = os.path.join(project_root, 'models', f"{model_name}.pth")
        log_model_file(model_file_path)
        
        mlflow.log_param('log_file', os.path.basename(log_file))
        mlflow.log_artifact(log_file, artifact_path="logs")
        
        results = {
            'log_file': log_file,
            'metrics': metrics,
            'ukf_vel_rmse': ukf_vel_rmse,
            'ukf_pos_rmse': ukf_pos_rmse,
            'pure_vel_rmse': pure_vel_rmse,
            'pure_pos_rmse': pure_pos_rmse,
            'fa_rmse': fa_rmse,
            'fa_rmse_total': fa_rmse_total
        }
        
        return True, None, results
        
    except Exception as e:
        error_msg = f"处理文件 {csv_filename} 时发生错误: {str(e)}\n{traceback.format_exc()}"
        return False, error_msg, None

