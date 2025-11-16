% % close all,clear
% global glv
function [imu,avp0] = av2imu_main3(start_index, excel_filename)
glvs;

% 读取处理过后，只有位置、速度、姿态的数据
% 如果未提供文件名，使用默认文件名（向后兼容）
if nargin < 2
    excel_filename = 'apts_expi_70psin2t.xlsx';
end
data = readtable(excel_filename);
% data = data(101:end,:);
real_pav_q = table2array(data);
real_pav_att = zeros(length(real_pav_q),10);
aps_att2 = real_pav_att;

real_pav_att(:,1:3) = real_pav_q(:,1:3);
real_pav_att(:,7:9) = real_pav_q(:,8:10);
real_pav_att(:,10) = real_pav_q(:,11);

lat0 = 34.13801;
lon0 = -118.12528;
h0 = 2.0470;
wgs84 = wgs84Ellipsoid;

for i = 1:length(real_pav_q)
% for i = 1:30
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
real_ap(:,1:3) = real_pav_att(:,4:6);
real_ap(:,4:6) = real_pav_att(:,1:3);
real_ap(:,7) = real_pav_att(:,10);
real_ap = real_ap(start_index:end,:);
[imu, avp0] = ap2imu(real_ap, 0.02);
% 惯导输出添加误差
imuerr = imuerrset(10, 1000, 0.0001, 0.0001);
% imuerr = imuerrset(0, 0, 0.15, 20,1440,3.6); % 武汉大学文章对应的MEMS惯导静态参数
% imuerr = imuerrset(10, 10000, 0.01, 0.01); % 某文献参数
% imuerr = imuerrset(0.3, 100, 0.3, 50);
imu = imuadderr(imu, imuerr);  % imuplot(imu);
% pure_avp = inspure(imu, avp0, 'f',0);
% imu = imu(1:length(real_pav_q)-2,:);
end

% % 输出的是弧度制的角增量！输出的是比力增量
% % imu = imu(50:end,:);
% imuplot(imu)
% % imu2(:,1:6) = imu(:,1:6) / 0.02;
% % imu2(:,7) = imu(:,7);
% % 
% % % 导出为excel文件
% % writematrix(imu2, 'train0_imu.xlsx')
% 
% % 惯导求解
% % avp00(1:3) = aps_att(1,1:3);
% % avp00(4:6) = [0.4533947856679178, -0.0880962570574872, -0.8517046962005139];
% % avp00(7:9) = aps_att(1,4:6);
% 
% % avp = inspure(imu, avp00', 1);
% %% 纯惯导解算
% % avp0(9) = real_pav_att_m(1,3);
% avp0(1:3) = real_pav_att(1,4:6);
% avp0(4:6) = real_pav_att(1,7:9);
% avp0(7:9) = real_pav_att(1,1:3);
% 

% % avp = inspure(imu, avp00, 'f', 1);
% % avperr = avpcmpplot(trj.avp, avp);
% 
% % for i = 1:length(avp)
% %     [xEast, yNorth, zUp] = geodetic2enu(avp(i,7)*180/pi,avp(i,8)*180/pi,avp(i,9),lat0,lon0,h0,wgs84, 'degrees');
% %     avp(i,7:8) = [xEast,yNorth];
% % end
% 
% % ins_avp_m = avp;
% % avp_middle = avp;
% ins_avp_m = zeros(length(avp)+1,10);
% ins_avp_m(1,1:9) = avp0;ins_avp_m(1,10) = 0;
% ins_avp_m(2:end,:) = avp;

% %% 惯导解算结果转为xyz
% for i = 1:length(ins_avp_m)
%     % 位置需要换成 纬度、经度、高度来表示(前两个是弧度)
% %     [xEast, yNorth, zUp] = geodetic2enu(avp(i,7)*180/pi,avp(i,8)*180/pi,avp(i,9),lat0,lon0,h0,wgs84, 'degrees');
% %     avp(i,7:8) = [xEast,yNorth];
%     pos = ins_avp_m(i,7:10);
%     pos0 = [lat0/180*pi,lon0/180*pi,h0]';
%     pos = pos2dxyz(pos,pos0);
%     ins_avp_m(i,7:10) = pos;
% %     [xEast2, yNorth2, zUp2] = geodetic2enu(aps_att(i,4)*180/pi,aps_att(i,5)*180/pi,aps_att(i,6),lat0,lon0,h0,wgs84, 'degrees');
% %     aps_att(i,4:5) = [xEast2,yNorth2];
% end
% 
% %% 作图比较
% % 位置
% figure
% subplot(2,1,1)
% plot(ins_avp_m(:,10),ins_avp_m(:,7))
% hold on
% plot(real_pav_att_m(:,10),real_pav_att_m(:,1));
% % plot(real_pav_q(:,1),real_pav_q(:,2));
% xlabel('时间t')
% ylabel('East[m]')
% legend('惯导结算结果','真实轨迹')
% 
% subplot(2,1,2)
% plot(ins_avp_m(:,10),ins_avp_m(:,8))
% hold on
% plot(real_pav_att_m(:,10),real_pav_att_m(:,2));
% % plot(real_pav_q(:,1),real_pav_q(:,2));
% xlabel('时间t')
% ylabel('North[m]')
% legend('惯导结算结果','真实轨迹')
% 
% % subplot(3,1,3)
% % plot(ins_avp_m(:,10),ins_avp_m(:,9))
% % hold on
% % plot(real_pav_att_m(:,10),real_pav_att_m(:,3));
% % % plot(real_pav_q(:,1),real_pav_q(:,2));
% % xlabel('时间t')
% % ylabel('Height[m]')
% % legend('惯导结算结果','真实轨迹')
% 
% % 速度
% figure
% subplot(2,1,1)
% plot(ins_avp_m(:,10),ins_avp_m(:,4))
% hold on
% plot(real_pav_att_m(:,10),real_pav_att_m(:,7));
% % plot(real_pav_q(:,1),real_pav_q(:,2));
% xlabel('时间t')
% ylabel('vx[m/s]')
% legend('惯导结算结果','真实轨迹')
% 
% subplot(2,1,2)
% plot(ins_avp_m(:,10),ins_avp_m(:,5))
% hold on
% plot(real_pav_att_m(:,10),real_pav_att_m(:,8));
% % plot(real_pav_q(:,1),real_pav_q(:,2));
% xlabel('时间t')
% ylabel('vy[m/s]')
% legend('惯导结算结果','真实轨迹')
% 
% subplot(3,1,3)
% plot(ins_avp_m(:,10),ins_avp_m(:,6))
% hold on
% plot(real_pav_att_m(:,10),real_pav_att_m(:,9));
% % plot(real_pav_q(:,1),real_pav_q(:,2));
% xlabel('时间t')
% ylabel('vz[m/s]')
% legend('惯导结算结果','真实轨迹')
% 
% % 真实位置（经纬度）
% figure
% subplot(2,1,1)
% plot(avp(:,10),avp(:,7))
% hold on
% plot(aps_att(:,7),aps_att(:,4));
% % plot(real_pav_q(:,1),real_pav_q(:,2));
% xlabel('时间t')
% ylabel('纬度')
% legend('惯导结算结果','真实轨迹')
% 
% subplot(2,1,2)
% plot(avp(:,10),avp(:,8))
% hold on
% plot(aps_att(:,7),aps_att(:,5));
% legend('惯导结算结果','真实轨迹')
% % plot(real_pav_q(:,1),real_pav_q(:,2));
% xlabel('时间t')
% ylabel('经度')
