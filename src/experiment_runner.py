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
from scipy.stats import chi2

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


def adaptive_r_update_zheng_raukf(eng, matlab_kf, matlab_avp, ins_pred_pos_xyz, dynamic_pos_xyz,
                                   initial_R_diag, chi_sq_threshold=None, tune0=0.1, a=5):
    """
    基于Zheng et al. (2018)文献的RA-UKF自适应R矩阵调整函数
    
    文献: "A Robust Adaptive Unscented Kalman Filter for Nonlinear Estimation 
          with Uncertain Noise Covariance" (DOI:10.3390/s18030808)
    
    该方法使用residual-based方法计算测量噪声协方差：
    - 首先计算psi统计量进行故障检测（基于innovation）
    - 如果psi超过阈值，计算自适应权重lambda (tune)
    - 使用residual（观测值 - 更新后的预测值）和S矩阵更新R
    - R = (1-lambda) * R + lambda * (residual * residual.T + S)
    
    参数:
        eng: MATLAB引擎
        matlab_kf: MATLAB UKF滤波器对象（更新后）
        matlab_avp: MATLAB UKF更新后的AVP状态（用于计算更新后的预测位置）
        ins_pred_pos_xyz: INS预测位置（更新前，用于计算innovation，XYZ坐标，3x1数组）
        dynamic_pos_xyz: 动力学模型预测的位置（观测值，3x1数组）
        initial_R_diag: 初始R矩阵的对角线元素（用于限制调整范围）
        chi_sq_threshold: 卡方分布阈值（默认根据3维95%分位数计算，约7.815）
        tune0: 自适应权重的基础值（默认0.1）
        a: 自适应权重的调整参数（默认5）
    
    返回:
        updated_R: 更新后的R矩阵（numpy数组）
        adapted: 是否进行了自适应调整（布尔值）
    """
    try:
        # 1. 获取更新后的P矩阵和R矩阵（UKF更新后）
        try:
            P_updated = np.array(eng.getfield(matlab_kf, 'Pxk'))
            R_old = np.array(eng.getfield(matlab_kf, 'Rk'))
        except Exception as e:
            print(f"获取P或R矩阵失败: {e}")
            return np.diag(initial_R_diag), False
        
        # 2. 从更新后的状态中提取位置（用于计算residual）
        # matlab_avp是LLH格式，需要转换为XYZ
        matlab_ukf_avp_xyz = eng.llh2xyz_subfun(matlab_avp)
        ukf_fused_avp_xyz = np.array(matlab_ukf_avp_xyz).flatten()
        updated_pred_pos_xyz = ukf_fused_avp_xyz[6:9]  # 提取位置部分
        
        # 3. 计算residual（观测值 - 更新后的预测值）
        # 这是文献中的关键：residual = x - h(x_updated)
        residual = dynamic_pos_xyz - updated_pred_pos_xyz
        residual = residual.reshape((3, 1))  # 列向量
        
        # 4. 计算innovation（用于故障检测）
        # innovation = 观测值 - 更新前的预测值
        innovation = dynamic_pos_xyz - ins_pred_pos_xyz
        innovation = innovation.reshape((3, 1))  # 列向量
        
        # 5. 提取位置相关的P矩阵子块
        H = np.eye(3)  # 位置观测矩阵（3x3单位矩阵）
        n_state = P_updated.shape[0]
        if n_state >= 15:
            # 15状态UKF：位置在索引12-14
            pos_indices = slice(12, 15)
        elif n_state >= 9:
            # 9状态UKF：位置在索引6-8
            pos_indices = slice(6, 9)
        else:
            # 默认假设位置在最后3个元素
            pos_indices = slice(-3, None)
        
        P_pos_updated = P_updated[pos_indices, :][:, pos_indices]  # 位置相关的协方差子块（更新后）
        
        # 6. 计算S矩阵（从更新后的状态计算的预测协方差，不包含R）
        # 在文献中，S是从更新后的sigma点计算的，这里我们使用更新后的P矩阵近似
        # S = H * P_updated_pos * H^T（不包含R）
        S = H @ P_pos_updated @ H.T
        
        # 7. 计算新息协方差矩阵 S_pred = H*P*H^T + R（用于计算psi）
        # 注意：为了准确计算psi，应该使用更新前的P和R
        # 但由于我们已经更新了，我们使用更新后的P和R作为近似
        # 或者，可以从MATLAB获取更新前的P（如果保存了）
        # 为了简化，这里使用更新后的P和R
        S_pred = S + R_old  # 使用更新后的S和R
        
        # 8. 计算psi统计量（用于故障检测）
        try:
            S_pred_inv = np.linalg.inv(S_pred)
            psi = (innovation.T @ S_pred_inv @ innovation)[0, 0]
        except Exception as e:
            print(f"计算psi失败: {e}")
            return R_old, False
        
        # 9. 设置卡方分布阈值（如果未提供）
        if chi_sq_threshold is None:
            chi_sq_threshold = chi2.ppf(0.95, df=3)  # 3维，95%分位数，约7.815
        
        # 10. 故障检测：如果psi不超过阈值，不进行自适应调整
        if psi <= chi_sq_threshold:
            return R_old, False
        
        # 11. 计算自适应权重lambda (tune)
        # lambda = max(tune0, (psi - a * chi_sq_threshold) / psi)
        tune = max(tune0, (psi - a * chi_sq_threshold) / psi)
        
        # 12. 计算自适应R矩阵
        # R_adaptive = residual * residual.T + S
        residual_outer = residual @ residual.T
        R_adaptive = residual_outer + S
        
        # 13. 确保R_adaptive是正定的
        # 提取对角线元素并确保为正
        R_adaptive_diag = np.diag(R_adaptive)
        R_adaptive_diag = np.maximum(R_adaptive_diag, initial_R_diag * 0.001)  # 设置下限
        
        # 限制R_adaptive_diag的范围
        R_adaptive_diag = np.clip(R_adaptive_diag, 
                                  initial_R_diag * 0.001, 
                                  initial_R_diag * 1000.0)
        
        # 14. 加权融合：R_new = (1-lambda) * R_old + lambda * R_adaptive
        R_old_diag = np.diag(R_old)
        R_new_diag = (1 - tune) * R_old_diag + tune * R_adaptive_diag
        
        # 15. 构建新的R矩阵（对角矩阵）
        R_new = np.diag(R_new_diag)
        
        # 16. 将更新后的R设置回MATLAB的kf对象
        eng.setfield(matlab_kf, 'Rk', matlab.double(R_new.tolist()), nargout=0)
        
        return R_new, True
        
    except Exception as e:
        # 如果出现任何错误，返回原始R
        print(f"Zheng RA-UKF自适应R调整出错: {e}")
        try:
            return np.array(eng.getfield(matlab_kf, 'Rk')), False
        except:
            return np.diag(initial_R_diag), False


def adaptive_r_update_raukf(eng, matlab_kf, ins_pred_pos_xyz, dynamic_pos_xyz, 
                             initial_R_diag, alpha=0.9, chi2_percentile=0.95):
    """
    RA-UKF自适应R矩阵调整函数
    
    基于新息统计（NIS）和能量项进行条件触发的R矩阵自适应调整
    
    参数:
        eng: MATLAB引擎
        matlab_kf: MATLAB UKF滤波器对象
        ins_pred_pos_xyz: INS预测位置（XYZ坐标，3x1数组）
        dynamic_pos_xyz: 动力学模型预测的位置（观测值，3x1数组）
        initial_R_diag: 初始R矩阵的对角线元素（用于限制调整范围）
        alpha: 加权融合的权重（默认0.9）
        chi2_percentile: 卡方分布的分位数（默认0.95，即95%）
    
    返回:
        updated_R: 更新后的R矩阵（numpy数组）
    """
    try:
        # 1. 计算新息（innovation）
        innovation = ins_pred_pos_xyz - dynamic_pos_xyz
        innovation = innovation.reshape((3, 1))  # 列向量
        
        # 2. 获取当前的P矩阵和R矩阵
        try:
            P = np.array(eng.getfield(matlab_kf, 'Pxk'))
            R_old = np.array(eng.getfield(matlab_kf, 'Rk'))
        except:
            # 如果获取失败，返回原始R
            return np.array(eng.getfield(matlab_kf, 'Rk')) if 'R_old' not in locals() else np.diag(initial_R_diag)
        
        # 3. 计算新息协方差矩阵 S = H*P*H^T + R
        # H是3x3单位矩阵（位置观测）
        H = np.eye(3)
        
        # 提取位置相关的P矩阵子块（假设状态向量中位置在最后3个元素）
        # 需要根据实际UKF状态向量结构调整索引
        # 对于15状态UKF，位置通常在索引12-14（0-based）
        n_state = P.shape[0]
        if n_state >= 15:
            # 15状态UKF：位置在索引12-14
            pos_indices = slice(12, 15)
        elif n_state >= 9:
            # 9状态UKF：位置在索引6-8
            pos_indices = slice(6, 9)
        else:
            # 默认假设位置在最后3个元素
            pos_indices = slice(-3, None)
        
        P_pos = P[pos_indices, :][:, pos_indices]  # 位置相关的协方差子块
        
        # 计算新息协方差（使用更新前的R，因为这是理论值）
        S = H @ P_pos @ H.T + R_old
        
        # 4. 计算归一化新息平方（NIS）
        try:
            S_inv = np.linalg.inv(S)
            NIS = (innovation.T @ S_inv @ innovation)[0, 0]
        except:
            # 如果S不可逆，返回原始R
            return R_old
        
        # 5. 计算卡方分布阈值（3维，95%分位数）
        chi2_threshold = chi2.ppf(chi2_percentile, df=3)  # 约7.815
        
        # 6. 基于新息能量计算自适应R
        # R_adaptive = innovation * innovation^T - H*P*H^T
        innovation_outer = innovation @ innovation.T
        R_adaptive = innovation_outer - H @ P_pos @ H.T
        
        # 确保R_adaptive是正定的（只保留对角线元素，并确保为正）
        R_adaptive_diag = np.diag(R_adaptive)
        R_adaptive_diag = np.maximum(R_adaptive_diag, initial_R_diag * 0.1)  # 设置下限
        
        # 7. 根据NIS统计检验结果调整R
        if NIS > chi2_threshold:
            # NIS超过阈值：观测噪声被低估，需要增大R
            # 增大比例基于NIS与阈值的比值
            scale_factor = 1.0 + 0.1 * (NIS / chi2_threshold - 1.0)  # 自适应增大
            R_adaptive_diag = R_adaptive_diag * scale_factor
        elif NIS < chi2_threshold * 0.5:
            # NIS远低于阈值：观测噪声可能被高估，可以适当减小R
            scale_factor = 0.95  # 小幅减小
            R_adaptive_diag = R_adaptive_diag * scale_factor
        
        # 限制R_adaptive_diag的范围（不能小于初始值的10%，不能大于初始值的10倍）
        R_adaptive_diag = np.clip(R_adaptive_diag, 
                                  initial_R_diag * 0.001, 
                                  initial_R_diag * 1000.0)
        
        # 8. 加权融合：R_new = alpha * R_old + (1-alpha) * R_adaptive
        R_old_diag = np.diag(R_old)
        R_new_diag = alpha * R_old_diag + (1 - alpha) * R_adaptive_diag
        
        # 9. 构建新的R矩阵（对角矩阵）
        R_new = np.diag(R_new_diag)
        
        # 10. 将更新后的R设置回MATLAB的kf对象
        eng.setfield(matlab_kf, 'Rk', matlab.double(R_new.tolist()), nargout=0)
        
        return R_new
        
    except Exception as e:
        # 如果出现任何错误，返回原始R或初始R
        print(f"自适应R调整出错: {e}")
        try:
            return np.array(eng.getfield(matlab_kf, 'Rk'))
        except:
            return np.diag(initial_R_diag)


def run_single_experiment(csv_filename, project_root, filter_config, eng, 
                         adapt_end_index=100, glv_init_code=None, task_batch_folder=None):
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
        
        # 从filter_config读取lambda1，如果没有则使用默认值0.1
        lambda1 = filter_config.get('filter', {}).get('lambda1', 0.1)
        
        options = {
            'dim_a': dim_a,
            'loss_type': 'crossentropy-loss',
            'solve_type': solver_type,
            'lambda1': lambda1,
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
        
        # 提取纯惯导的时间序列（第10行，索引9是时间）
        pure_avps_time = pure_avps[9, :] if pure_avps.shape[0] > 9 else None
        # 计算实际时间间隔（取前两个时间点的差值，或平均值）
        if pure_avps_time is not None and len(pure_avps_time) > 1:
            pure_ins_time_step = pure_avps_time[1] - pure_avps_time[0]
        else:
            pure_ins_time_step = 0.04  # 默认值（双子样）
        
        # 提前转换纯惯导数据到XYZ坐标系（用于日志记录）
        pure_avp_size = pure_avps.shape
        pure_avps_xyz_log = np.empty((9, pure_avp_size[1]))
        for i in range(pure_avp_size[1]):
            pure_avps_xyz_log[:,i] = np.array(eng.llh2xyz_subfun(matlab.double(pure_avps[:,i].tolist())))
        
        avp0_change = last_avp
        imu = np.array(matlab_imu)
        
        # 从filter_config读取MATLAB UKF参数
        matlab_ukf_config = filter_config.get('matlab_ukf', {})
        imu_err = matlab_ukf_config.get('imu_err', {})
        avp_err = matlab_ukf_config.get('avp_err', {})
        pos_err = matlab_ukf_config.get('pos_err', [0.001, 0.001, 0.001])
        # Rk矩阵通过pos_err设置，不再使用Rk_diag覆盖
        
        # 准备MATLAB UKF初始化参数
        imu_err_params = [
            imu_err.get('eb', 10),
            imu_err.get('db', 1000),
            imu_err.get('web', 0.0001),
            imu_err.get('wdb', 0.0001)
        ]
        phi_err = avp_err.get('phi', [0.1, 0.1, 0.1])
        dvn_err = avp_err.get('dvn', 0.1)
        dpos_err = avp_err.get('dpos', [1, 1, 3])
        
        # 确保所有参数都是列表格式（MATLAB Engine需要）
        # 对于标量值（如dvn_err），需要转换为包含一个元素的列表
        if not isinstance(dvn_err, (list, tuple, np.ndarray)):
            dvn_err = [dvn_err]
        # 对于已经是列表的参数，确保是列表类型
        if isinstance(phi_err, np.ndarray):
            phi_err = phi_err.tolist()
        elif not isinstance(phi_err, (list, tuple)):
            phi_err = [phi_err] if not hasattr(phi_err, '__iter__') else list(phi_err)
        
        if isinstance(dpos_err, np.ndarray):
            dpos_err = dpos_err.tolist()
        elif not isinstance(dpos_err, (list, tuple)):
            dpos_err = [dpos_err] if not hasattr(dpos_err, '__iter__') else list(dpos_err)
        
        if isinstance(pos_err, np.ndarray):
            pos_err = pos_err.tolist()
        elif not isinstance(pos_err, (list, tuple)):
            pos_err = [pos_err] if not hasattr(pos_err, '__iter__') else list(pos_err)
        
        # UKF初始化
        # Rk矩阵通过pos_err设置：rk = poserrset(pos_err)，然后kfinit自动设置 kf.Rk = diag(rk)^2
        matlab_kf, matlab_ins = eng.SINS_dynamic_UKF153_init(
            matlab.double(avp0_change.tolist()),
            matlab.double(imu_err_params),
            matlab.double(phi_err),
            matlab.double(dvn_err),
            matlab.double(dpos_err),
            matlab.double(pos_err),
            nargout=2
        )
        
        # 获取UKF参数
        try:
            ukf_Qk = np.array(eng.getfield(matlab_kf, 'Qk')) if 'Qk' in str(matlab_kf) else None
            ukf_Rk = np.array(eng.getfield(matlab_kf, 'Rk')) if 'Rk' in str(matlab_kf) else None
            ukf_Pxk = np.array(eng.getfield(matlab_kf, 'Pxk')) if 'Pxk' in str(matlab_kf) else None
        except:
            ukf_Qk = None
            ukf_Rk = None
            ukf_Pxk = None
        
        # 保存初始R矩阵的对角线元素（用于自适应调整时的范围限制）
        initial_R_diag = np.diag(ukf_Rk) if ukf_Rk is not None else np.array(pos_err) ** 2
        
        # 打印UKF的Rk和Qk值（用于调试和验证）
        print(f"\n  ========== UKF参数值 ==========")
        if ukf_Rk is not None:
            print(f"  Rk矩阵 (观测噪声协方差):")
            print(f"    完整矩阵:\n{ukf_Rk}")
            print(f"    对角元素: {np.diag(ukf_Rk)}")
            print(f"    矩阵大小: {ukf_Rk.shape}")
            # 显示输入参数信息
            print(f"    来源: pos_err = {pos_err} (通过poserrset设置rk，kfinit自动设置Rk = diag(rk)^2)")
        else:
            print(f"  Rk矩阵: 未获取到")
        
        if ukf_Qk is not None:
            print(f"  Qk矩阵 (过程噪声协方差):")
            print(f"    矩阵大小: {ukf_Qk.shape}")
            # 显示对角元素（Qk通常是对角矩阵）
            diag_Qk = np.diag(ukf_Qk)
            print(f"    对角元素 (前10个): {diag_Qk[:min(10, len(diag_Qk))]}")
            if ukf_Qk.size <= 25:
                print(f"    完整矩阵:\n{ukf_Qk}")
            else:
                print(f"    完整矩阵 (前5x5子矩阵):\n{ukf_Qk[:5, :5]}")
            # 显示输入参数信息
            print(f"    来源: IMU误差参数 = [eb={imu_err_params[0]}, db={imu_err_params[1]}, web={imu_err_params[2]}, wdb={imu_err_params[3]}]")
            print(f"          (kfinit使用imuerr.web和imuerr.wdb设置Qt，然后Qk = Qt*nts)")
        else:
            print(f"  Qk矩阵: 未获取到")
        print(f"  =================================\n")
        
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
        neural_f_total_collection = []
        real_fa_total_collection = []
        
        # 初始化日志文件
        log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 如果指定了大任务文件夹，使用它；否则使用默认的navigation_logs目录
        if task_batch_folder:
            log_dir = task_batch_folder
        else:
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
            'real_fa_x', 'real_fa_y', 'real_fa_z',
            'neural_f_total_x', 'neural_f_total_y', 'neural_f_total_z',
            'real_fa_total_x', 'real_fa_total_y', 'real_fa_total_z',
            'dynamic_a_0', 'dynamic_a_1', 'dynamic_a_2',
            'dynamic_pos_x', 'dynamic_pos_y', 'dynamic_pos_z',
            'dynamic_vel_x', 'dynamic_vel_y', 'dynamic_vel_z',
            'dynamic_vdot_x', 'dynamic_vdot_y', 'dynamic_vdot_z',
            'ins_pred_pos_x', 'ins_pred_pos_y', 'ins_pred_pos_z',
            'ins_pred_att_x', 'ins_pred_att_y', 'ins_pred_att_z',
            'ins_pred_vx', 'ins_pred_vy', 'ins_pred_vz',
            'obs_residual_x', 'obs_residual_y', 'obs_residual_z',
            'innovation_x', 'innovation_y', 'innovation_z',
            'Kk_pos_00', 'Kk_pos_01', 'Kk_pos_02',
            'Kk_pos_10', 'Kk_pos_11', 'Kk_pos_12',
            'Kk_pos_20', 'Kk_pos_21', 'Kk_pos_22',
            'correction_x', 'correction_y', 'correction_z',  # K * innovation，动力学模型观测带来的修正
            'R_adaptive_00', 'R_adaptive_11', 'R_adaptive_22',
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
        # 初始化R_adaptive_diag为初始值（用于第一次循环的日志记录）
        R_adaptive_diag = initial_R_diag.copy()
        
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
            
            # 计算 total 力：neural_f + R@fT + m*g 和 real_fa + R@fT + m*g
            m0 = 2.6
            g_ = 9.8
            m_g = np.array([0, 0, -m0 * g_])  # 重力（惯性坐标系）
            R_fT = (Ri @ fT).flatten()  # 推力转换到惯性坐标系
            neural_f_total_xyz = neural_fa_xyz + R_fT + m_g
            
            if loop_index-1 < len(real_fas):
                real_fa_xyz = real_fas[loop_index-1, :]
                real_fa_total_xyz = real_fa_xyz + R_fT + m_g
                
                neural_fa_collection.append(neural_fa_xyz)
                real_fa_collection.append(real_fa_xyz)
                neural_f_total_collection.append(neural_f_total_xyz)
                real_fa_total_collection.append(real_fa_total_xyz)
                fa_time_collection.append(ts[loop_index-1] if loop_index-1 < len(ts) else loop_index * 0.02)
            else:
                real_fa_xyz = np.array([0.0, 0.0, 0.0])
                real_fa_total_xyz = neural_f_total_xyz  # 如果没有真实值，使用neural_f_total作为占位符
            
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
            
            # 计算新息（innovation），用于日志记录
            innovation_xyz = obs_residual_xyz.copy()  # 新息就是观测残差
            
            # 获取卡尔曼增益Kk（从MATLAB的kf对象中）
            try:
                Kk = np.array(eng.getfield(matlab_kf, 'Kk'))
                # 提取位置相关的Kk子块（对于15状态UKF，位置在索引12-14）
                # Kk是n_state x 3的矩阵，最后3行对应位置状态
                n_state = Kk.shape[0]
                if n_state >= 15:
                    # 15状态UKF：位置在索引12-14
                    pos_indices = slice(12, 15)
                elif n_state >= 9:
                    # 9状态UKF：位置在索引6-8
                    pos_indices = slice(6, 9)
                else:
                    # 默认假设位置在最后3个元素
                    pos_indices = slice(-3, None)
                
                Kk_pos = Kk[pos_indices, :]  # 位置相关的Kk子块（3x3）
            except Exception as e:
                # 如果获取失败，使用零矩阵
                print(f"获取卡尔曼增益失败: {e}")
                Kk_pos = np.zeros((3, 3))
            
            # 计算修正值：K * innovation（动力学模型观测带来的修正）
            # innovation_xyz是行向量，需要转换为列向量进行计算
            innovation_col = innovation_xyz.reshape((3, 1))  # 转换为列向量
            correction_col = Kk_pos @ innovation_col  # K * innovation（3x1列向量）
            correction_xyz = correction_col.flatten()  # 转换回行向量，用于日志记录
            
            # Zheng RA-UKF自适应R矩阵调整（在UKF更新之后，用于下一时刻）
            # 基于Zheng et al. (2018)文献的方法，使用residual-based方法调整R矩阵
            try:
                updated_R, adapted = adaptive_r_update_zheng_raukf(
                    eng, matlab_kf, matlab_avp, ins_pred_pos_xyz, dynamic_pos_xyz,
                    initial_R_diag, chi_sq_threshold=None, tune0=0.1, a=5
                )
                # 保存更新后的R矩阵对角线元素，用于日志记录
                R_adaptive_diag = np.diag(updated_R)
            except Exception as e:
                # 如果自适应调整失败，获取当前的R矩阵
                print(f"Zheng RA-UKF自适应R调整失败，使用原始R: {e}")
                try:
                    current_R = np.array(eng.getfield(matlab_kf, 'Rk'))
                    R_adaptive_diag = np.diag(current_R)
                except:
                    # 如果获取失败，使用初始R
                    R_adaptive_diag = initial_R_diag.copy()
            
            # UKF融合结果
            matlab_ukf_avp_xyz = eng.llh2xyz_subfun(matlab_avp)
            ukf_fused_avp_xyz = np.array(matlab_ukf_avp_xyz).flatten()
            ukf_fused_att_xyz = ukf_fused_avp_xyz[0:3] * 180.0 / np.pi
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
            
            # 纯惯导数据（根据索引关系匹配）
            # INS时间间隔是0.04s，UKF时间间隔是0.02s，所以每2个UKF循环对应1个INS数据点
            # INS索引 = (loop_index - first_index) // 2
            pure_ins_index = (loop_index - first_index) // 2
            pure_avps_length = pure_avps_xyz_log.shape[1]
            
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
                real_fa_xyz[0], real_fa_xyz[1], real_fa_xyz[2],
                neural_f_total_xyz[0], neural_f_total_xyz[1], neural_f_total_xyz[2],
                real_fa_total_xyz[0], real_fa_total_xyz[1], real_fa_total_xyz[2],
                dynamic_a[0,0], dynamic_a[1,0], dynamic_a[2,0],
                dynamic_pos_xyz[0], dynamic_pos_xyz[1], dynamic_pos_xyz[2],
                dynamic_vel_xyz[0], dynamic_vel_xyz[1], dynamic_vel_xyz[2],
                dynamic_vdot_xyz[0], dynamic_vdot_xyz[1], dynamic_vdot_xyz[2],
                ins_pred_pos_xyz[0], ins_pred_pos_xyz[1], ins_pred_pos_xyz[2],
                ins_pred_att_xyz[0], ins_pred_att_xyz[1], ins_pred_att_xyz[2],
                ins_pred_vel_xyz[0], ins_pred_vel_xyz[1], ins_pred_vel_xyz[2],
                obs_residual_xyz[0], obs_residual_xyz[1], obs_residual_xyz[2],
                innovation_xyz[0], innovation_xyz[1], innovation_xyz[2],
                Kk_pos[0, 0], Kk_pos[0, 1], Kk_pos[0, 2],
                Kk_pos[1, 0], Kk_pos[1, 1], Kk_pos[1, 2],
                Kk_pos[2, 0], Kk_pos[2, 1], Kk_pos[2, 2],
                correction_xyz[0], correction_xyz[1], correction_xyz[2],
                R_adaptive_diag[0], R_adaptive_diag[1], R_adaptive_diag[2],
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
        
        # 使用pure_avps的实际时间序列（第10行，索引9是时间）
        # pure_avps_time已经在前面提取过了
        if pure_avps_time is not None and len(pure_avps_time) > 0:
            pure_ins_time = pure_avps_time
            actual_pure_avps_length = len(pure_avps_time)
        else:
            # 如果没有时间信息，使用默认计算（向后兼容）
            pure_ins_time_step = 0.04  # 默认双子样时间间隔
            pure_ins_start_time = (adapt_end_index + 1) * 0.02
            actual_pure_avps_length = pure_avps_xyz.shape[1]
            pure_ins_time_range = (actual_pure_avps_length - 1) * pure_ins_time_step
            pure_ins_end_time = pure_ins_start_time + pure_ins_time_range
            pure_ins_time = np.arange(pure_ins_start_time, pure_ins_end_time + pure_ins_time_step, pure_ins_time_step)
            pure_ins_time = pure_ins_time[:actual_pure_avps_length]
        
        # 使用所有pure_avps_xyz数据计算速度和位置的RMSE
        pure_ins_vel_xyz = pure_avps_xyz[3:6, :]
        pure_ins_pos_xyz = pure_avps_xyz[6:9, :]
        
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
        
        # 计算 neural_f_total vs real_fa_total 的RMSE
        if len(neural_f_total_collection) > 0 and len(real_fa_total_collection) > 0:
            neural_f_total_array = np.array(neural_f_total_collection)
            real_fa_total_array = np.array(real_fa_total_collection)
            fa_total_error = neural_f_total_array - real_fa_total_array
            fa_total_rmse = np.sqrt(np.mean(fa_total_error**2, axis=0))
            fa_total_rmse_total = np.sqrt(np.mean(fa_total_error**2))
        else:
            fa_total_rmse = np.array([0.0, 0.0, 0.0])
            fa_total_rmse_total = 0.0
        
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
        
        # 记录UKF参数：使用原始输入参数值，而不是转换后的矩阵
        # 这样在可视化平台上可以更方便地比较参数
        ukf_params = {}
        
        # 记录位置误差参数（用于设置Rk矩阵）
        # pos_err格式：[rx, ry, rz]，单位：米
        # 参考：rk = poserrset([1;1;3])，kfinit会自动设置 kf.Rk = diag(rk)^2
        ukf_params['pos_err_rx'] = float(pos_err[0]) if len(pos_err) > 0 else 0.0
        ukf_params['pos_err_ry'] = float(pos_err[1]) if len(pos_err) > 1 else 0.0
        ukf_params['pos_err_rz'] = float(pos_err[2]) if len(pos_err) > 2 else 0.0
        ukf_params['pos_err'] = str(pos_err)  # 完整列表，便于查看
        
        # 记录IMU误差参数（用于设置Qk矩阵）
        # imu_err_params格式：[eb, db, web, wdb]
        # eb: 陀螺常值偏置 (deg/h)
        # db: 加速度计常值偏置 (ug)
        # web: 角随机游走 (deg/sqrt(h))
        # wdb: 速度随机游走 (ug/sqrt(Hz))
        ukf_params['imu_err_eb'] = float(imu_err_params[0]) if len(imu_err_params) > 0 else 0.0
        ukf_params['imu_err_db'] = float(imu_err_params[1]) if len(imu_err_params) > 1 else 0.0
        ukf_params['imu_err_web'] = float(imu_err_params[2]) if len(imu_err_params) > 2 else 0.0
        ukf_params['imu_err_wdb'] = float(imu_err_params[3]) if len(imu_err_params) > 3 else 0.0
        
        # 记录AVP误差参数（用于设置Pxk矩阵）
        ukf_params['avp_err_phi'] = str(phi_err)  # 平台失准角 [phi_x, phi_y, phi_z] (arcmin)
        ukf_params['avp_err_dvn'] = float(dvn_err[0]) if isinstance(dvn_err, (list, tuple, np.ndarray)) else float(dvn_err)  # 速度误差 (m/s)
        ukf_params['avp_err_dpos'] = str(dpos_err)  # 位置误差 [dlat, dlon, dhgt] (m)
        
        # 保留转换后的矩阵值作为参考（可选，用于调试）
        # 注意：这些是转换后的矩阵，主要用于调试，可视化时应该使用上面的原始参数
        if ukf_Qk is not None:
            # Qk矩阵较大，只记录对角元素和统计信息
            if ukf_Qk.size <= 9:
                ukf_params['Qk_matrix'] = str(ukf_Qk.tolist())
            else:
                ukf_params['Qk_matrix_shape'] = str(ukf_Qk.shape)
                ukf_params['Qk_matrix_diag'] = str(np.diag(ukf_Qk).tolist())
        
        if ukf_Rk is not None:
            # Rk矩阵较小，记录完整矩阵
            if ukf_Rk.size <= 9:
                ukf_params['Rk_matrix'] = str(ukf_Rk.tolist())
                ukf_params['Rk_matrix_diag'] = str(np.diag(ukf_Rk).tolist())
        
        if ukf_Pxk is not None:
            # Pxk矩阵很大，只记录对角元素的前几个
            if ukf_Pxk.size <= 25:
                ukf_params['Pxk_matrix_diag'] = str(np.diag(ukf_Pxk).tolist())
            else:
                diag_pxk = np.diag(ukf_Pxk)
                ukf_params['Pxk_matrix_diag_first10'] = str(diag_pxk[:10].tolist())
                ukf_params['Pxk_matrix_shape'] = str(ukf_Pxk.shape)
        
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
            'neural_fa_total_rmse_x': float(fa_total_rmse[0]),
            'neural_fa_total_rmse_y': float(fa_total_rmse[1]),
            'neural_fa_total_rmse_z': float(fa_total_rmse[2]),
            'neural_fa_total_rmse_total': float(fa_total_rmse_total),
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

