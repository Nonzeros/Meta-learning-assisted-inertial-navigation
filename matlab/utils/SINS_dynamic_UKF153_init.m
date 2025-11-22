% UKF初始化
% glvs
% 参数说明：
%   avp0 - 初始姿态、速度、位置
%   imu_err_params - [eb, db, web, wdb] IMU误差参数
%   phi_err - [phi_x, phi_y, phi_z] 平台失准角 (arcmin)
%   dvn_err - 速度误差 (m/s)
%   dpos_err - [dlat, dlon, dhgt] 位置误差 (m)
%   pos_err - [rx, ry, rz] 位置误差 (m) - 用于poserrset，参考test_SINS_GPS_UKF_153.m
%   Rk矩阵通过pos_err设置：rk = poserrset(pos_err)，然后kfinit自动设置 kf.Rk = diag(rk)^2
function [kf,ins] = SINS_dynamic_UKF153_init(avp0, imu_err_params, phi_err, dvn_err, dpos_err, pos_err)
psinstypedef('test_SINS_GPS_UKF_153_def');
ts = 0.02; % 时间步长

ins = insinit(avp0, ts);

% 如果参数未提供，使用默认值（参考test_SINS_GPS_UKF_153.m）
if nargin < 2
    % 默认位置误差：参考test_SINS_GPS_UKF_153.m使用[1;1;3]
    rk = poserrset([1;1;3]);
    imuerr = imuerrset(10, 1000, 0.0001, 0.0001);
    davp0 = avperrset([1;1;1]*0.1, 0.1, [1;1;3]);
else
    % 使用传入的参数
    % pos_err用于poserrset，设置位置观测误差（单位：米）
    % 参考test_SINS_GPS_UKF_153.m: rk = poserrset([1;1;3]);
    if exist('pos_err', 'var') && ~isempty(pos_err)
        rk = poserrset(pos_err(:));
    else
        % 如果pos_err未提供，使用默认值
        rk = poserrset([1;1;3]);
    end
    imuerr = imuerrset(imu_err_params(1), imu_err_params(2), imu_err_params(3), imu_err_params(4));
    davp0 = avperrset(phi_err(:), dvn_err, dpos_err(:));
end

% kfinit会自动使用rk设置Rk矩阵：kf.Rk = diag(rk)^2
% 参考test_SINS_GPS_UKF_153.m: kf = kfinit(ins, davp0, imuerr, rk);
kf = kfinit(ins, davp0, imuerr, rk);
end