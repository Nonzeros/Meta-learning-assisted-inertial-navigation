function pure_avp = pure_ins_solve(imu,avp0)
% 纯惯导求解
pure_avp = inspure(imu, avp0, 'f',0);
end

