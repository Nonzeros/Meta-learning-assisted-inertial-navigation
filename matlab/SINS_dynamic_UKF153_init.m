% UKF初始化
% glvs
function [kf,ins] = SINS_dynamic_UKF153_init(avp0)
psinstypedef('test_SINS_GPS_UKF_153_def');
ts = 0.02; % 时间步长

ins = insinit(avp0, ts);
rk = poserrset([0.001;0.001;0.001]);
imuerr = imuerrset(10, 1000, 0.0001, 0.0001);
% imuerr = imuerrset(100000, 100000, 100000, 100000);
% imu = imuadderr(trj.imu, imuerr);  % imuplot(imu);
davp0 = avperrset([1;1;1]*0.1, 0.1, [1;1;3]);
kf = kfinit(ins, davp0, imuerr, rk);
% kf.Rk = [1000000,0,0;0, 1000000, 0; 0,0,1000000];
% kf.Rk = [2,0,0; 0,2, 0; 0,0,2];
% kf.Rk = [0.00005,0,0; 0,0.00005, 0; 0,0,0.00005];
% kf.Rk = [0.000007,0,0; 0,0.000007, 0; 0,0,0.000007];
kf.Rk = [0.000000010,0,0; 0,0.00000065, 0; 0,0,0.0000011];
% kf.Rk = [0.0000012,0,0; 0,0.00000012, 0; 0,0,0.000000012];
% kf.Rk = [0.0000005,0,0; 0,0.0000005, 0; 0,0,0.0000005];
end

% kf.Rk = [0.0001,0,0; 0,0.00001, 0; 0,0,0.00001];