% UKF初始化
% glvs
% 参数说明：
%   avp0 - 初始姿态、速度、位置
%   imu_err_params - [eb, db, web, wdb] IMU误差参数
%   phi_err - [phi_x, phi_y, phi_z] 平台失准角 (arcmin)
%   dvn_err - 速度误差 (m/s)
%   dpos_err - [dlat, dlon, dhgt] 位置误差 (m)
%   vel_err - [vx, vy, vz] 速度观测误差 (m/s) - 用于速度观测，直接设置rk = vel_err
%   Rk矩阵通过vel_err设置：rk = vel_err（速度观测），然后kfinit自动设置 kf.Rk = diag(rk)^2
%   注意：观测矩阵Hk在test_SINS_GPS_UKF_153_def.m中已定义为速度观测：[zeros(3,3), eye(3), zeros(3,9)]
function [kf,ins] = SINS_dynamic_UKF153_init(avp0, imu_err_params, phi_err, dvn_err, dpos_err, vel_err)
psinstypedef('test_SINS_GPS_UKF_153_def');
ts = 0.02; % 时间步长

ins = insinit(avp0, ts);

% 如果参数未提供，使用默认值
if nargin < 2
    % 默认速度观测误差：0.1 m/s
    rk = [0.1; 0.1; 0.1];
    imuerr = imuerrset(10, 1000, 0.0001, 0.0001);
    davp0 = avperrset([1;1;1]*0.1, 0.1, [1;1;3]);
else
    % 使用传入的参数
    % vel_err用于速度观测，直接设置rk = vel_err（单位：m/s）
    % 速度观测时，不需要使用poserrset或vperrset，直接使用速度误差值
    if exist('vel_err', 'var') && ~isempty(vel_err)
        rk = vel_err(:);  % 直接使用速度误差值，单位：m/s
    else
        % 如果vel_err未提供，使用默认值
        rk = [0.1; 0.1; 0.1];  % 默认速度观测误差：0.1 m/s
    end
    imuerr = imuerrset(imu_err_params(1), imu_err_params(2), imu_err_params(3), imu_err_params(4));
    davp0 = avperrset(phi_err(:), dvn_err, dpos_err(:));
end

% kfinit会自动使用rk设置Rk矩阵：kf.Rk = diag(rk)^2
% 对于速度观测，rk的单位是m/s，所以Rk的单位是(m/s)^2
% 参考test_SINS_GPS_UKF_153_def.m: kf.Rk = diag(rk)^2
kf = kfinit(ins, davp0, imuerr, rk);
end