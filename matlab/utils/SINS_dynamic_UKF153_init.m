% UKF初始化
% glvs
% 参数说明：
%   avp0 - 初始姿态、速度、位置
%   imu_err_params - [eb, db, web, wdb] IMU误差参数
%   phi_err - [phi_x, phi_y, phi_z] 平台失准角 (arcmin)
%   dvn_err - 速度误差 (m/s)
%   dpos_err - [dlat, dlon, dhgt] 位置误差 (m)
%   pos_err - [rx, ry, rz] 位置误差 (m) - 用于poserrset
%   Rk_diag - [Rk_xx, Rk_yy, Rk_zz] R矩阵对角元素
function [kf,ins] = SINS_dynamic_UKF153_init(avp0, imu_err_params, phi_err, dvn_err, dpos_err, pos_err, Rk_diag)
psinstypedef('test_SINS_GPS_UKF_153_def');
ts = 0.02; % 时间步长

ins = insinit(avp0, ts);

% 如果参数未提供，使用默认值
if nargin < 2
    rk = poserrset([0.001;0.001;0.001]);
    imuerr = imuerrset(10, 1000, 0.0001, 0.0001);
    davp0 = avperrset([1;1;1]*0.1, 0.1, [1;1;3]);
    Rk_diag = [0.000000010, 0.00000065, 0.0000011];
else
    % 使用传入的参数
    rk = poserrset(pos_err(:));
    imuerr = imuerrset(imu_err_params(1), imu_err_params(2), imu_err_params(3), imu_err_params(4));
    davp0 = avperrset(phi_err(:), dvn_err, dpos_err(:));
end

kf = kfinit(ins, davp0, imuerr, rk);

% 设置R矩阵
if nargin >= 7
    kf.Rk = diag(Rk_diag(:));
else
    % 默认值
    kf.Rk = [0.000000010,0,0; 0,0.00000065, 0; 0,0,0.0000011];
end
end