function [avp_kf,ins,kf,ins_pred_pos] = test_SINS_dynamic_UKF_153_forpython(dynamic_pos,imu_output,kf,ins)
% % 读取excel数据
% data = readtable('apts_expi_pf30_2025-6-03.xlsx');
% 读取到动力学观测数据(pvt)
% pvt_m = table2array(data);
pvt_m = dynamic_pos;
% 东北天转经纬高
lat0 = 34.13801;
lon0 = -118.12528;
h0 = 2.0470;
wgs84 = wgs84Ellipsoid;
pvt = pvt_m;

[lat,lon,h] = enu2geodetic(pvt(1),pvt(2),pvt(3),lat0,lon0,h0,wgs84); % 出来的经纬度是角度
p = [lat*pi/180 ,lon*pi/180]; % 经纬度弧度表示
pvt(1:2) = p;

% UKF filter
% UKF计算
% wvm = imu_output(1:2,1:6);  t = imu_output(2,7);
wvm = imu_output(1,1:6);  t = imu_output(1,7);
ins = insupdate(ins, wvm);
kf.px = ins;
kf = ukf(kf);

% 保存融合前的INS预测位置（用于日志记录）
ins_pred_pos = ins.pos;

kf = ukf(kf, ins.pos-pvt, 'B');  % UKF filter
[kf, ins] = kffeedback(kf, ins, 1, 'avp');

avp_kf = [ins.avp', t];
% 返回融合前的INS预测位置（第4个输出，已在第27行保存）
% 注意：Python调用时需要指定nargout=4
end
% avperr = avpcmpplot(avp, avp);
% kfplot(xkpk, avperr, imuerr);
% %% 输出结果坐标转换
% avp_copy = avp;
% avp(1,:) = [avp0',0];
% avp(2:length(avp_copy)+1,:) = avp_copy;
% % aps_att2 = zeros(length(avp),7);
% % 经纬高转局部xyz
% for i = 1:length(avp)
%     pos = avp(i,7:10);
%     pos0 = [lat0/180*pi,lon0/180*pi,h0]';
%     pos = pos2dxyz(pos,pos0);
%     avp(i,7:10) = pos;
% %     [xEast2, yNorth2, zUp2] = geodetic2enu(aps_att(i,4)*180/pi,aps_att(i,5)*180/pi,aps_att(i,6),lat0,lon0,h0,wgs84, 'degrees');
% %     aps_att(i,4:5) = [xEast2,yNorth2];
% end
