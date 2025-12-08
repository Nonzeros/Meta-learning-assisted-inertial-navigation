function [avp_kf,ins,kf,ins_pred_pos] = test_SINS_dynamic_UKF_153_forpython(dynamic_pos,imu_output,kf,ins)
% % 读取excel数据
% data = readtable('apts_expi_pf30_2025-6-03.xlsx');
% 读取到动力学观测数据(pvt)
% pvt_m = table2array(data);

% ========== MATLAB 调试配置 ==========
% 在 MATLAB 命令行中运行以下命令以启用调试：
%   dbstop if error      % 自动在错误处断点
%   dbstop if warning   % 自动在警告处断点
% 或者在 MATLAB 编辑器中直接设置断点
% =====================================

% ========== 调试标志：是否输出详细的UKF更新信息 ==========
% 通过全局变量 IS_BASELINE_MODEL 来判断是否是baseline模型
ENABLE_DETAILED_DEBUG = false;  % 设置为true以启用详细调试输出
DEBUG_BASELINE_ONLY = true;     % 如果为true，只对baseline模型输出调试信息

% ========== 关键修复：确保 dynamic_pos 是列向量 (3×1) ==========
% 如果 dynamic_pos 是 1×3 的行向量，转置成 3×1 的列向量
% 如果是 3×1 的列向量，保持不变
dynamic_pos_original_size = size(dynamic_pos);
if size(dynamic_pos, 1) == 1 && size(dynamic_pos, 2) == 3
    % 是 1×3 的行向量，需要转置
    dynamic_pos = dynamic_pos(:);  % 转置成列向量
    if ENABLE_DETAILED_DEBUG
        fprintf('[调试] 检测到 dynamic_pos 是 1×3 行向量 [%d, %d]，已转置为 3×1 列向量 [%d, %d]\n', ...
            dynamic_pos_original_size(1), dynamic_pos_original_size(2), ...
            size(dynamic_pos, 1), size(dynamic_pos, 2));
    end
elseif size(dynamic_pos, 1) == 3 && size(dynamic_pos, 2) == 1
    % 已经是 3×1 的列向量，不需要处理
    if ENABLE_DETAILED_DEBUG
        fprintf('[调试] dynamic_pos 已经是 3×1 列向量，无需处理\n');
    end
else
    % 其他情况，尝试转换为列向量
    dynamic_pos = dynamic_pos(:);
    if size(dynamic_pos, 1) ~= 3
        error('dynamic_pos 的大小不正确，期望是 3 个元素，实际是 %d 个元素', size(dynamic_pos, 1));
    end
    if ENABLE_DETAILED_DEBUG
        fprintf('[调试] dynamic_pos 已从 [%d, %d] 转换为列向量 [%d, %d]\n', ...
            dynamic_pos_original_size(1), dynamic_pos_original_size(2), ...
            size(dynamic_pos, 1), size(dynamic_pos, 2));
    end
end

% 检查是否是baseline模型（通过全局变量）
is_baseline = false;
if DEBUG_BASELINE_ONLY
    try
        global IS_BASELINE_MODEL;
        if exist('IS_BASELINE_MODEL', 'var') && IS_BASELINE_MODEL
            is_baseline = true;
        end
    catch
        % 如果全局变量不存在，通过dynamic_pos的值范围判断（备用方法）
        if max(abs(dynamic_pos)) < 1000
            is_baseline = true;
        end
    end
end

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

% ========== UKF更新前的状态记录（用于调试） ==========
if ENABLE_DETAILED_DEBUG
    if is_baseline || ~DEBUG_BASELINE_ONLY
        fprintf('\n========== [零动力学模型] UKF更新步骤详情 ==========\n');
        fprintf('步骤1: UKF更新前的状态\n');
        fprintf('  - 观测值 (dynamic_pos, ENU米): [%.6f, %.6f, %.6f]\n', pvt_m(1), pvt_m(2), pvt_m(3));
        fprintf('  - INS预测位置 (ins.pos, LLH弧度): [%.10f, %.10f, %.6f]\n', ins.pos(1), ins.pos(2), ins.pos(3));
        
        % 计算新息（innovation）
        innovation = ins.pos - pvt;
        fprintf('  - 新息 (innovation = ins.pos - pvt): [%.10f, %.10f, %.6f]\n', innovation(1), innovation(2), innovation(3));
        
        % 显示UKF状态
        if isfield(kf, 'xk') && ~isempty(kf.xk)
            xk_show = kf.xk;
            if size(xk_show, 2) > 1
                xk_show = xk_show(:, 1);
            end
            fprintf('  - kf.xk (状态估计, 前9个): [');
            for i = 1:min(9, length(xk_show))
                fprintf('%.6e ', xk_show(i));
            end
            fprintf(']\n');
        end
        
        % 显示协方差矩阵的对角元素
        if isfield(kf, 'Pxk') && ~isempty(kf.Pxk)
            Pxk_diag = diag(kf.Pxk);
            fprintf('  - kf.Pxk 对角元素 (前9个): [');
            for i = 1:min(9, length(Pxk_diag))
                fprintf('%.6e ', Pxk_diag(i));
            end
            fprintf(']\n');
        end
        
        % 显示观测噪声协方差
        if isfield(kf, 'Rk') && ~isempty(kf.Rk)
            Rk_diag = diag(kf.Rk);
            fprintf('  - kf.Rk 对角元素: [%.6e, %.6e, %.6e]\n', Rk_diag(1), Rk_diag(2), Rk_diag(3));
        end
    end
end

kf = ukf(kf, ins.pos-pvt, 'M');  % UKF filter

% ========== UKF更新后的状态记录（用于调试） ==========
if ENABLE_DETAILED_DEBUG
    if is_baseline || ~DEBUG_BASELINE_ONLY
        fprintf('步骤2: UKF更新后的状态\n');
        
        % 显示更新后的状态
        if isfield(kf, 'xk') && ~isempty(kf.xk)
            xk_show = kf.xk;
            if size(xk_show, 2) > 1
                xk_show = xk_show(:, 1);
            end
            fprintf('  - kf.xk (更新后, 前9个): [');
            for i = 1:min(9, length(xk_show))
                fprintf('%.6e ', xk_show(i));
            end
            fprintf(']\n');
            
            % 检查是否有异常大的值
            if any(abs(xk_show) > 1e6)
                fprintf('  ⚠️  警告：检测到异常大的状态值！\n');
                fprintf('  - 最大绝对值: %.6e\n', max(abs(xk_show)));
            end
        end
        
        % 显示卡尔曼增益（位置相关的部分）
        if isfield(kf, 'Kk') && ~isempty(kf.Kk)
            Kk = kf.Kk;
            n_state = size(Kk, 1);
            if n_state >= 15
                % 15状态UKF：位置在索引12-14
                Kk_pos = Kk(12:14, :);
            elseif n_state >= 9
                % 9状态UKF：位置在索引6-8
                Kk_pos = Kk(6:8, :);
            else
                Kk_pos = Kk(end-2:end, :);
            end
            fprintf('  - kf.Kk (位置相关, 3x3):\n');
            for i = 1:3
                fprintf('    [%.6e, %.6e, %.6e]\n', Kk_pos(i, 1), Kk_pos(i, 2), Kk_pos(i, 3));
            end
        end
        
        % 显示更新后的协方差矩阵对角元素
        if isfield(kf, 'Pxk') && ~isempty(kf.Pxk)
            Pxk_diag = diag(kf.Pxk);
            fprintf('  - kf.Pxk 对角元素 (更新后, 前9个): [');
            for i = 1:min(9, length(Pxk_diag))
                fprintf('%.6e ', Pxk_diag(i));
            end
            fprintf(']\n');
        end
        
        % 显示INS的当前状态（速度、位置）
        if isfield(ins, 'vn') && ~isempty(ins.vn)
            fprintf('  - ins.vn (速度, 米/秒): [%.6f, %.6f, %.6f]\n', ins.vn(1), ins.vn(2), ins.vn(3));
            if any(abs(ins.vn) > 1e4)
                fprintf('  ⚠️  警告：速度异常大！\n');
            end
        end
        if isfield(ins, 'pos') && ~isempty(ins.pos)
            fprintf('  - ins.pos (位置, LLH弧度): [%.10f, %.10f, %.6f]\n', ins.pos(1), ins.pos(2), ins.pos(3));
        end
    end
end

% ========== 关键修复：确保kf.xk是列向量，防止kffeedback维度错误 ==========
% ukf函数可能将kf.xk设置为矩阵，但kffeedback需要列向量
% 首先确保kf.n存在
if ~isfield(kf, 'n') || isempty(kf.n)
    if isfield(kf, 'xk') && ~isempty(kf.xk)
        kf.n = size(kf.xk, 1);
    else
        kf.n = 15;  % 默认15状态UKF
    end
end

% 修复kf.xk的形状
if isfield(kf, 'xk') && ~isempty(kf.xk)
    if size(kf.xk, 2) > 1
        % 如果xk是矩阵，取第一列（通常第一列是最新的状态估计）
        kf.xk = kf.xk(:, 1);
    end
    % 确保xk是列向量且大小匹配
    if size(kf.xk, 1) ~= kf.n
        % 如果大小不匹配，重新初始化
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

% ========== 调试断点位置 ==========
% 在 MATLAB 编辑器中，可以在下面这行设置断点，以便调试 kffeedback 调用
% 调试时可以使用以下命令检查变量：
%   size(kf.xk)        % 检查 xk 的维度
%   size(kf.xfb)       % 检查 xfb 的维度
%   size(kf.coef_fb)   % 检查 coef_fb 的维度
%   kf.n               % 检查状态数
%   dbstep             % 单步执行
%   dbcont             % 继续执行
% ====================================
% 添加调试输出（帮助定位问题，输出会显示在 Python 控制台）
if ENABLE_DETAILED_DEBUG
    if is_baseline || ~DEBUG_BASELINE_ONLY
        fprintf('步骤3: 准备调用 kffeedback\n');
        fprintf('  - kf.n = %d\n', kf.n);
        if isfield(kf, 'xk')
            fprintf('  - kf.xk 大小: [%d, %d]\n', size(kf.xk, 1), size(kf.xk, 2));
        end
        if isfield(kf, 'xfb')
            fprintf('  - kf.xfb 大小: [%d, %d]\n', size(kf.xfb, 1), size(kf.xfb, 2));
        end
        if isfield(kf, 'coef_fb')
            fprintf('  - kf.coef_fb 大小: [%d, %d]\n', size(kf.coef_fb, 1), size(kf.coef_fb, 2));
        end
    end
end

% ========== 使用调试版本的 kffeedback ==========
% 如果启用详细调试，使用 kffeedback_debug 来查看内部运行过程
% 否则使用原始的 kffeedback
USE_DEBUG_KFFEEDBACK = ENABLE_DETAILED_DEBUG && (is_baseline || ~DEBUG_BASELINE_ONLY);

if USE_DEBUG_KFFEEDBACK
    % 使用调试版本，会输出每一步的详细信息
    [kf, ins] = kffeedback_debug(kf, ins, 1, 'avp', true);
else
    % 使用原始版本
[kf, ins] = kffeedback(kf, ins, 1, 'avp');
end

% ========== kffeedback后的状态记录（用于调试） ==========
if ENABLE_DETAILED_DEBUG
    if is_baseline || ~DEBUG_BASELINE_ONLY
        fprintf('步骤4: kffeedback后的状态\n');
        
        % 显示反馈后的状态
        if isfield(kf, 'xfb') && ~isempty(kf.xfb)
            xfb_show = kf.xfb;
            if size(xfb_show, 2) > 1
                xfb_show = xfb_show(:);
            end
            fprintf('  - kf.xfb (总反馈量, 前9个): [');
            for i = 1:min(9, length(xfb_show))
                fprintf('%.6e ', xfb_show(i));
            end
            fprintf(']\n');
        end
        
        % 显示INS的最终状态
        if isfield(ins, 'vn') && ~isempty(ins.vn)
            fprintf('  - ins.vn (反馈后速度, 米/秒): [%.6f, %.6f, %.6f]\n', ins.vn(1), ins.vn(2), ins.vn(3));
            if any(abs(ins.vn) > 1e4)
                fprintf('  ⚠️  警告：反馈后速度异常大！可能导致数值爆炸！\n');
            end
        end
        if isfield(ins, 'pos') && ~isempty(ins.pos)
            fprintf('  - ins.pos (反馈后位置, LLH弧度): [%.10f, %.10f, %.6f]\n', ins.pos(1), ins.pos(2), ins.pos(3));
        end
        if isfield(ins, 'att') && ~isempty(ins.att)
            fprintf('  - ins.att (姿态, 弧度): [%.6f, %.6f, %.6f]\n', ins.att(1), ins.att(2), ins.att(3));
        end
        
        fprintf('==========================================\n\n');
    end
end

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
