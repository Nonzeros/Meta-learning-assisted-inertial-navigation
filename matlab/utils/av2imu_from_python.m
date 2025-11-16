% 从Python传入的数据进行惯导反演
% 输入参数：
%   pavq_data: N x 11 矩阵，列顺序为 [p(3列), q(4列), v(3列), t(1列)]
%   start_index: 起始索引（从1开始）
%   lat0, lon0, h0: 初始位置（经纬度，高度）
function [imu,avp0] = av2imu_from_python(pavq_data, start_index, lat0, lon0, h0)
glvs;

% pavq_data格式: [p_x, p_y, p_z, q0, q1, q2, q3, v_x, v_y, v_z, t]
real_pav_q = pavq_data;
real_pav_att = zeros(length(real_pav_q),10);

% 提取位置、四元数、速度、时间
real_pav_att(:,1:3) = real_pav_q(:,1:3);  % 位置 (ENU坐标系)
real_pav_att(:,4:7) = real_pav_q(:,4:7);  % 四元数
real_pav_att(:,8:10) = real_pav_q(:,8:10); % 速度
real_pav_att(:,10) = real_pav_q(:,11);     % 时间

% 如果没有提供初始位置，使用默认值
if nargin < 3 || isempty(lat0)
    lat0 = 34.13801;
    lon0 = -118.12528;
    h0 = 2.0470;
end

wgs84 = wgs84Ellipsoid;

% 四元数转姿态角
for i = 1:length(real_pav_q)
    att = q2att(real_pav_q(i,4:7));
    real_pav_att(i,4:6) = att;
end

real_pav_att_m = real_pav_att;

% 位置转为经纬高
for i = 1:length(real_pav_q)
    % 位置需要换成 纬度、经度、高度来表示(前两个是弧度)
    [lat,lon,h] = enu2geodetic(real_pav_att(i,1),real_pav_att(i,2),real_pav_att(i,3),lat0,lon0,h0,wgs84); % 出来的经纬度是角度
    p = [lat*pi/180 ,lon*pi/180]; % 经纬度弧度表示
    real_pav_att(i,1:2) = p;
    real_pav_att(i,6) = real_pav_att(i,6) + pi; % 将偏航角值域变为0~2pi
end

%% 惯导反演算法
real_ap = zeros(length(real_pav_q),7);
real_ap(:,1:3) = real_pav_att(:,4:6);  % 姿态
real_ap(:,4:6) = real_pav_att(:,1:3);  % 位置（经纬高）
real_ap(:,7) = real_pav_att(:,10);     % 时间

% 从start_index开始截取
real_ap = real_ap(start_index:end,:);

% 计算时间步长（从数据中获取）
if length(real_ap) > 1
    dt = real_ap(2,7) - real_ap(1,7);
else
    dt = 0.02;  % 默认时间步长
end

[imu, avp0] = ap2imu(real_ap, dt);

% 惯导输出添加误差
imuerr = imuerrset(10, 1000, 0.0001, 0.0001);
imu = imuadderr(imu, imuerr);

end

