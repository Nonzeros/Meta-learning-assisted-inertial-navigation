% 数据准备
file_name = "F:\navigation_codes\dynamic_metalearning\ProcessedData\apts_expi_70psin2t.xlsx";
[imu, avp0] = av2imu_main3(1,file_name);
[kf, ins] = SINS_dynamic_UKF153_init(avp0);

% ========== 显示UKF初始化参数 ==========
fprintf('\n========== UKF初始化参数 ==========\n');
fprintf('时间步长 (ts): %.6f s\n', kf.nts);
fprintf('状态维度 (n): %d\n', kf.n);
fprintf('观测维度 (m): %d\n', kf.m);

% 显示Qk矩阵（过程噪声协方差）
if isfield(kf, 'Qk')
    fprintf('\nQk矩阵 (过程噪声协方差):\n');
    disp(kf.Qk);
    fprintf('Qk矩阵大小: %dx%d\n', size(kf.Qk, 1), size(kf.Qk, 2));
    fprintf('Qk矩阵对角元素:\n');
    disp(diag(kf.Qk));
else
    fprintf('\nQk矩阵: 未初始化\n');
end

% 显示Qt矩阵（如果存在）
if isfield(kf, 'Qt')
    fprintf('\nQt矩阵 (过程噪声):\n');
    disp(kf.Qt);
end

% 显示Rk矩阵（观测噪声协方差）
if isfield(kf, 'Rk')
    fprintf('\nRk矩阵 (观测噪声协方差):\n');
    disp(kf.Rk);
    fprintf('Rk矩阵大小: %dx%d\n', size(kf.Rk, 1), size(kf.Rk, 2));
    fprintf('Rk矩阵对角元素:\n');
    disp(diag(kf.Rk));
else
    fprintf('\nRk矩阵: 未初始化\n');
end

% 显示Pxk矩阵（状态协方差）
if isfield(kf, 'Pxk')
    fprintf('\nPxk矩阵 (状态协方差):\n');
    fprintf('Pxk矩阵大小: %dx%d\n', size(kf.Pxk, 1), size(kf.Pxk, 2));
    fprintf('Pxk矩阵对角元素:\n');
    disp(diag(kf.Pxk));
    fprintf('Pxk矩阵前5x5子矩阵:\n');
    disp(kf.Pxk(1:min(5, size(kf.Pxk,1)), 1:min(5, size(kf.Pxk,2))));
else
    fprintf('\nPxk矩阵: 未初始化\n');
end

% 显示Hk矩阵（观测矩阵）
if isfield(kf, 'Hk')
    fprintf('\nHk矩阵 (观测矩阵):\n');
    fprintf('Hk矩阵大小: %dx%d\n', size(kf.Hk, 1), size(kf.Hk, 2));
    fprintf('Hk矩阵:\n');
    disp(kf.Hk);
end

% 显示初始状态
if isfield(kf, 'xk')
    fprintf('\n初始状态向量 xk (前10个元素):\n');
    disp(kf.xk(1:min(10, length(kf.xk))));
    fprintf('状态向量维度: %d\n', length(kf.xk));
end

% 显示INS初始状态
fprintf('\n========== INS初始状态 ==========\n');
fprintf('初始姿态 (att): [%.6f, %.6f, %.6f] rad\n', ins.att(1), ins.att(2), ins.att(3));
fprintf('初始速度 (vn): [%.6f, %.6f, %.6f] m/s\n', ins.vn(1), ins.vn(2), ins.vn(3));
fprintf('初始位置 (pos): [%.6f, %.6f, %.6f]\n', ins.pos(1), ins.pos(2), ins.pos(3));

fprintf('\n========== 开始UKF滤波计算 ==========\n');
fprintf('总迭代次数: %d\n', min(2000, size(imu, 1)));

% 初始化结果数组
dynamic_pos = [10;10;1];
max_iter = min(2000, size(imu, 1));
avps = zeros(max_iter, 10);

% 显示选项
show_progress = true;  % 是否显示进度
show_intermediate = false;  % 是否显示中间结果（每100次迭代）
show_first_last = true;  % 是否显示第一次和最后一次迭代的详细信息

% UKF滤波循环
for i = 1:max_iter
    imu_output = imu(i,:);
    [avp_kf, ins, kf, ins_pred_pos] = test_SINS_dynamic_UKF_153_forpython(dynamic_pos, imu_output, kf, ins);
    avps(i,:) = avp_kf;
    
    % 显示进度
    if show_progress && mod(i, 100) == 0
        fprintf('进度: %d/%d (%.1f%%)\n', i, max_iter, i/max_iter*100);
    end
    
    % 显示第一次迭代的详细信息
    if show_first_last && i == 1
        fprintf('\n========== 第一次迭代后的状态 ==========\n');
        fprintf('时间: %.6f s\n', avp_kf(10));
        fprintf('姿态: [%.6f, %.6f, %.6f] rad\n', avp_kf(1), avp_kf(2), avp_kf(3));
        fprintf('速度: [%.6f, %.6f, %.6f] m/s\n', avp_kf(4), avp_kf(5), avp_kf(6));
        fprintf('位置: [%.6f, %.6f, %.6f]\n', avp_kf(7), avp_kf(8), avp_kf(9));
        
        if isfield(kf, 'Qk')
            fprintf('\n第一次迭代后的Qk矩阵对角元素:\n');
            disp(diag(kf.Qk));
        end
        
        if isfield(kf, 'Pxk')
            fprintf('第一次迭代后的Pxk矩阵对角元素 (前10个):\n');
            disp(diag(kf.Pxk(1:min(10, size(kf.Pxk,1)), 1:min(10, size(kf.Pxk,2)))));
        end
    end
    
    % 显示中间结果
    if show_intermediate && mod(i, 100) == 0 && i > 1
        fprintf('\n迭代 %d: 时间=%.6f, 位置=[%.6f, %.6f, %.6f]\n', ...
            i, avp_kf(10), avp_kf(7), avp_kf(8), avp_kf(9));
    end
end

% 显示最后一次迭代的详细信息
if show_first_last && max_iter > 1
    fprintf('\n========== 最后一次迭代后的状态 ==========\n');
    fprintf('时间: %.6f s\n', avps(max_iter, 10));
    fprintf('姿态: [%.6f, %.6f, %.6f] rad\n', avps(max_iter, 1), avps(max_iter, 2), avps(max_iter, 3));
    fprintf('速度: [%.6f, %.6f, %.6f] m/s\n', avps(max_iter, 4), avps(max_iter, 5), avps(max_iter, 6));
    fprintf('位置: [%.6f, %.6f, %.6f]\n', avps(max_iter, 7), avps(max_iter, 8), avps(max_iter, 9));
    
    if isfield(kf, 'Qk')
        fprintf('\n最后一次迭代后的Qk矩阵对角元素:\n');
        disp(diag(kf.Qk));
    end
    
    if isfield(kf, 'Pxk')
        fprintf('最后一次迭代后的Pxk矩阵对角元素 (前10个):\n');
        disp(diag(kf.Pxk(1:min(10, size(kf.Pxk,1)), 1:min(10, size(kf.Pxk,2)))));
    end
end

fprintf('\n========== UKF滤波计算完成 ==========\n');

% 绘图
figure
plot(avps(:,10), avps(:,4));
hold on 
xlabel('时间 (s)');
ylabel('速度 East (m/s)');
title('UKF滤波结果 - 东向速度');
grid on;
% plot(pure_avp(:,10),pure_avp(:,4))
