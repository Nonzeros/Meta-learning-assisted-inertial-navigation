"""
参数扫描工具：生成滤波器参数组合
"""
import yaml
import itertools
import copy
from typing import List, Dict, Any


def load_param_scan_config(config_path: str) -> Dict[str, Any]:
    """
    加载参数扫描配置文件
    
    参数:
        config_path: 配置文件路径
    
    返回:
        参数字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def generate_param_combinations(scan_config: Dict[str, Any], base_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    生成所有参数组合
    
    参数:
        scan_config: 参数扫描配置
        base_config: 基础配置（filter_config.yaml）
    
    返回:
        参数组合列表
    """
    combinations = []
    scan_mode = scan_config.get('scan_mode', 'grid')
    
    if scan_mode == 'auto':
        # 自动扫描模式：自动测试KF和PF，以及所有相关参数
        auto_scan = scan_config.get('auto_scan', {})
        scan_filter_type = auto_scan.get('scan_filter_type', True)
        
        if scan_filter_type:
            # 扫描卡尔曼滤波
            kf_params = auto_scan.get('kalman_filter_params', {})
            kf_lambda1_values = kf_params.get('lambda1_values', [0.1])
            kf_Q_values = kf_params.get('Q_values', [0.1])
            kf_R_values = kf_params.get('R_values', [0.1])
            
            for lambda1, Q, R in itertools.product(kf_lambda1_values, kf_Q_values, kf_R_values):
                new_config = copy.deepcopy(base_config)
                new_config['filter']['type'] = 1  # 卡尔曼滤波
                new_config['filter']['lambda1'] = lambda1
                new_config['kalman_filter']['Q'] = Q
                new_config['kalman_filter']['R'] = R
                combinations.append(new_config)
            
            # 扫描粒子滤波
            pf_params = auto_scan.get('particle_filter_params', {})
            pf_lambda1_values = pf_params.get('lambda1_values', [0.1])
            pf_numPar_values = pf_params.get('numPar_values', [30])
            pf_R_values = pf_params.get('R_values', [0.1])
            pf_q0_values = pf_params.get('q0_values', [0.1])
            
            for lambda1, numPar, R, q0 in itertools.product(
                pf_lambda1_values, pf_numPar_values, pf_R_values, pf_q0_values
            ):
                new_config = copy.deepcopy(base_config)
                new_config['filter']['type'] = 2  # 粒子滤波
                new_config['filter']['lambda1'] = lambda1
                new_config['filter']['numPar'] = int(numPar)
                new_config['particle_filter']['R'] = R
                new_config['particle_filter']['q0'] = q0
                combinations.append(new_config)
        else:
            # 不扫描滤波器类型，只使用基础配置中的类型
            filter_type = base_config.get('filter', {}).get('type', 2)
            if filter_type == 1:
                # 卡尔曼滤波
                kf_params = auto_scan.get('kalman_filter_params', {})
                kf_lambda1_values = kf_params.get('lambda1_values', [0.1])
                kf_Q_values = kf_params.get('Q_values', [0.1])
                kf_R_values = kf_params.get('R_values', [0.1])
                
                for lambda1, Q, R in itertools.product(kf_lambda1_values, kf_Q_values, kf_R_values):
                    new_config = copy.deepcopy(base_config)
                    new_config['filter']['lambda1'] = lambda1
                    new_config['kalman_filter']['Q'] = Q
                    new_config['kalman_filter']['R'] = R
                    combinations.append(new_config)
            else:
                # 粒子滤波
                pf_params = auto_scan.get('particle_filter_params', {})
                pf_lambda1_values = pf_params.get('lambda1_values', [0.1])
                pf_numPar_values = pf_params.get('numPar_values', [30])
                pf_R_values = pf_params.get('R_values', [0.1])
                pf_q0_values = pf_params.get('q0_values', [0.1])
                
                for lambda1, numPar, R, q0 in itertools.product(
                    pf_lambda1_values, pf_numPar_values, pf_R_values, pf_q0_values
                ):
                    new_config = copy.deepcopy(base_config)
                    new_config['filter']['lambda1'] = lambda1
                    new_config['filter']['numPar'] = int(numPar)
                    new_config['particle_filter']['R'] = R
                    new_config['particle_filter']['q0'] = q0
                    combinations.append(new_config)
    
    elif scan_mode == 'single':
        # 单参数扫描模式：一次只改变一个参数
        single_scan = scan_config.get('single_param_scan', {})
        param_name = single_scan.get('param_name', 'lambda1')
        param_values = single_scan.get('param_values', [0.1])
        
        for param_value in param_values:
            new_config = copy.deepcopy(base_config)
            
            # 根据参数名称更新对应的参数
            if param_name == 'lambda1':
                new_config['filter']['lambda1'] = param_value
            elif param_name == 'numPar':
                new_config['filter']['numPar'] = int(param_value)  # 确保是整数
            elif param_name == 'particle_filter.R':
                new_config['particle_filter']['R'] = param_value
            elif param_name == 'particle_filter.q0':
                new_config['particle_filter']['q0'] = param_value
            elif param_name == 'kalman_filter.Q':
                new_config['kalman_filter']['Q'] = param_value
            elif param_name == 'kalman_filter.R':
                new_config['kalman_filter']['R'] = param_value
            elif param_name == 'matlab_ukf.Rk':
                # 如果param_value是标量，转换为三个相同元素的列表
                if isinstance(param_value, (int, float)):
                    new_config['matlab_ukf']['Rk'] = [param_value, param_value, param_value]
                elif isinstance(param_value, list) and len(param_value) == 3:
                    new_config['matlab_ukf']['Rk'] = param_value
                else:
                    raise ValueError(f"matlab_ukf.Rk 参数值必须是标量或长度为3的列表，当前值: {param_value}")
            elif param_name == 'matlab_ukf.web':
                new_config['matlab_ukf']['imu_err']['web'] = param_value
            elif param_name == 'matlab_ukf.wdb':
                new_config['matlab_ukf']['imu_err']['wdb'] = param_value
            elif param_name == 'matlab_ukf.phi':
                new_config['matlab_ukf']['avp_err']['phi'] = param_value
            elif param_name == 'matlab_ukf.dvn':
                new_config['matlab_ukf']['avp_err']['dvn'] = param_value
            elif param_name == 'matlab_ukf.dpos':
                new_config['matlab_ukf']['avp_err']['dpos'] = param_value
            else:
                raise ValueError(f"不支持的参数名称: {param_name}")
            
            combinations.append(new_config)
    
    elif scan_mode == 'list':
        # 列表模式：直接使用指定的参数组合
        param_combinations = scan_config.get('param_combinations', [])
        for param_combo in param_combinations:
            # 深度复制嵌套字典
            new_config = copy.deepcopy(base_config)
            
            # 更新粒子滤波参数
            if 'particle_filter' in param_combo:
                new_config['particle_filter'].update(param_combo['particle_filter'])
            
            # 更新卡尔曼滤波参数
            if 'kalman_filter' in param_combo:
                new_config['kalman_filter'].update(param_combo['kalman_filter'])
            
            # 更新MATLAB UKF参数
            if 'matlab_ukf' in param_combo:
                for key, value in param_combo['matlab_ukf'].items():
                    if key in ['Rk', 'phi', 'dpos']:
                        new_config['matlab_ukf'][key] = value
                    elif key in ['web', 'wdb', 'dvn']:
                        if key == 'web':
                            new_config['matlab_ukf']['imu_err']['web'] = value
                        elif key == 'wdb':
                            new_config['matlab_ukf']['imu_err']['wdb'] = value
                        elif key == 'dvn':
                            new_config['matlab_ukf']['avp_err']['dvn'] = value
            
            combinations.append(new_config)
    
    else:
        # 网格搜索模式：生成所有参数组合
        filter_type = base_config.get('filter', {}).get('type', 2)
        
        if filter_type == 2:
            # 粒子滤波
            pf_scan = scan_config.get('particle_filter_scan', {})
            R_values = pf_scan.get('R_values', [0.1])
            q0_values = pf_scan.get('q0_values', [0.1])
            
            for R, q0 in itertools.product(R_values, q0_values):
                new_config = copy.deepcopy(base_config)
                new_config['particle_filter']['R'] = R
                new_config['particle_filter']['q0'] = q0
                
                # 生成MATLAB UKF参数组合
                ukf_combos = _generate_ukf_combinations(scan_config, new_config)
                combinations.extend(ukf_combos)
        
        elif filter_type == 1:
            # 卡尔曼滤波
            kf_scan = scan_config.get('kalman_filter_scan', {})
            Q_values = kf_scan.get('Q_values', [0.1])
            R_values = kf_scan.get('R_values', [0.1])
            
            for Q, R in itertools.product(Q_values, R_values):
                new_config = copy.deepcopy(base_config)
                new_config['kalman_filter']['Q'] = Q
                new_config['kalman_filter']['R'] = R
                
                # 生成MATLAB UKF参数组合
                ukf_combos = _generate_ukf_combinations(scan_config, new_config)
                combinations.extend(ukf_combos)
    
    return combinations


def _generate_ukf_combinations(scan_config: Dict[str, Any], base_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    生成MATLAB UKF参数组合
    
    参数:
        scan_config: 参数扫描配置
        base_config: 基础配置（已包含粒子滤波或卡尔曼滤波参数）
    
    返回:
        参数组合列表
    """
    import copy
    combinations = []
    
    ukf_scan = scan_config.get('matlab_ukf_scan', {})
    
    # 获取参数列表
    Rk_values = ukf_scan.get('Rk_values', [[0.000000010, 0.00000065, 0.0000011]])
    web_values = ukf_scan.get('web_values', [0.0001])
    wdb_values = ukf_scan.get('wdb_values', [0.0001])
    phi_values = ukf_scan.get('phi_values', [[0.1, 0.1, 0.1]])
    dvn_values = ukf_scan.get('dvn_values', [0.1])
    dpos_values = ukf_scan.get('dpos_values', [[1, 1, 3]])
    
    # 生成所有组合
    for Rk, web, wdb, phi, dvn, dpos in itertools.product(
        Rk_values, web_values, wdb_values, phi_values, dvn_values, dpos_values
    ):
        new_config = copy.deepcopy(base_config)
        new_config['matlab_ukf']['Rk'] = Rk
        new_config['matlab_ukf']['imu_err']['web'] = web
        new_config['matlab_ukf']['imu_err']['wdb'] = wdb
        new_config['matlab_ukf']['avp_err']['phi'] = phi
        new_config['matlab_ukf']['avp_err']['dvn'] = dvn
        new_config['matlab_ukf']['avp_err']['dpos'] = dpos
        
        combinations.append(new_config)
    
    return combinations


def get_param_string(config: Dict[str, Any]) -> str:
    """
    生成参数组合的字符串表示（用于MLflow run名称）
    
    参数:
        config: 配置字典
    
    返回:
        参数字符串
    """
    parts = []
    
    # 检查是否有lambda1或numPar参数（单参数扫描时可能只改变这些）
    filter_config = config.get('filter', {})
    if 'lambda1' in filter_config:
        parts.append(f"lambda1{filter_config['lambda1']}")
    if 'numPar' in filter_config:
        parts.append(f"numPar{filter_config['numPar']}")
    
    filter_type = filter_config.get('type', 2)
    
    if filter_type == 2:
        # 粒子滤波
        pf = config.get('particle_filter', {})
        parts.append(f"PF_R{pf.get('R', 0.1)}_q0{pf.get('q0', 0.1)}")
    elif filter_type == 1:
        # 卡尔曼滤波
        kf = config.get('kalman_filter', {})
        parts.append(f"KF_Q{kf.get('Q', 0.1)}_R{kf.get('R', 0.1)}")
    
    # MATLAB UKF参数
    ukf = config.get('matlab_ukf', {})
    Rk = ukf.get('Rk', [0.000000010, 0.00000065, 0.0000011])
    imu_err = ukf.get('imu_err', {})
    avp_err = ukf.get('avp_err', {})
    
    # Rk表示：如果三个元素相同，只显示一个值；否则显示所有值
    if len(Rk) == 3 and Rk[0] == Rk[1] == Rk[2]:
        # 三个元素相同，只显示一个值
        Rk_val = Rk[0]
        if Rk_val >= 1:
            Rk_str = f"Rk{Rk_val:.0f}"
        elif Rk_val >= 0.1:
            Rk_str = f"Rk{Rk_val:.1f}"
        elif Rk_val >= 0.01:
            Rk_str = f"Rk{Rk_val:.2f}"
        elif Rk_val >= 0.001:
            Rk_str = f"Rk{Rk_val:.3f}"
        else:
            # 很小的值，使用科学计数法
            Rk_str = f"Rk{Rk_val:.2e}"
    else:
        # 三个元素不同，显示所有值
        Rk_str = f"Rk[{Rk[0]},{Rk[1]},{Rk[2]}]"
    parts.append(f"UKF_{Rk_str}_web{imu_err.get('web', 0.0001)}_wdb{imu_err.get('wdb', 0.0001)}")
    parts.append(f"phi{avp_err.get('phi', [0.1, 0.1, 0.1])[0]}_dvn{avp_err.get('dvn', 0.1)}")
    
    return "_".join(parts)

