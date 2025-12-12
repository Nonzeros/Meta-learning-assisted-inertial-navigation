function [avp_kf,ins,kf,ins_pred_pos] = test_SINS_dynamic_UKF_153_forpython(dynamic_vel,imu_output,kf,ins)

% 确保 dynamic_vel 是列向量 (3×1)
if size(dynamic_vel, 1) == 1 && size(dynamic_vel, 2) == 3
    dynamic_vel = dynamic_vel(:);
elseif size(dynamic_vel, 1) ~= 3 || size(dynamic_vel, 2) ~= 1
    dynamic_vel = dynamic_vel(:);
    if size(dynamic_vel, 1) ~= 3
        error('dynamic_vel 的大小不正确，期望是 3 个元素，实际是 %d 个元素', size(dynamic_vel, 1));
    end
end

% UKF filter
wvm = imu_output(1,1:6);
t = imu_output(1,7);
ins = insupdate(ins, wvm);
kf.px = ins;
kf = ukf(kf);

% 保存融合前的INS预测位置（用于日志记录）
ins_pred_pos = ins.pos;

% UKF更新：使用速度观测
kf = ukf(kf, ins.vn - dynamic_vel, 'M');

% 确保kf.xk是列向量
if ~isfield(kf, 'n') || isempty(kf.n)
    if isfield(kf, 'xk') && ~isempty(kf.xk)
        kf.n = size(kf.xk, 1);
    else
        kf.n = 15;
    end
end

if isfield(kf, 'xk') && ~isempty(kf.xk)
    if size(kf.xk, 2) > 1
        kf.xk = kf.xk(:, 1);
    end
    if size(kf.xk, 1) ~= kf.n
        kf.xk = zeros(kf.n, 1);
    end
else
    kf.xk = zeros(kf.n, 1);
end

% 确保kf.xfb存在且是列向量
if ~isfield(kf, 'xfb') || isempty(kf.xfb)
    kf.xfb = zeros(kf.n, 1);
else
    if size(kf.xfb, 2) > 1
        kf.xfb = kf.xfb(:);
    end
    if size(kf.xfb, 1) ~= kf.n
        kf.xfb = zeros(kf.n, 1);
    end
end

% 卡尔曼滤波反馈
[kf, ins] = kffeedback(kf, ins, 1, 'avp');

avp_kf = [ins.avp', t];
end
