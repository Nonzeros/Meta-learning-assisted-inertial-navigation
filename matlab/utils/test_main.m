av2imu_main3;
SINS_dynamic_UKF153_init;
dynamic_pos = [10;10;1];
avps = zeros(2000,10);
for i = 1:2000
imu_output = imu(i,:);
[avp_kf,ins,kf] = test_SINS_dynamic_UKF_153_forpython(dynamic_pos,imu_output,kf,ins);
avps(i,:) = avp_kf;
end
figure
plot(avps(:,10),avps(:,4));
hold on 
% plot(pure_avp(:,10),pure_avp(:,4))
