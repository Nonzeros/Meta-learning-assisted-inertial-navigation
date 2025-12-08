function [kf, ins, xfb] = kffeedback_debug(kf, ins, T_fb, fbstr, enable_debug)
% kffeedback 的调试版本，逐步执行并输出详细信息
%
% 参数:
%   kf, ins, T_fb, fbstr - 与 kffeedback 相同
%   enable_debug - 是否启用详细调试输出（默认 true）
%
% 输出:
%   kf, ins, xfb - 与 kffeedback 相同

if nargin < 5, enable_debug = true; end
if nargin < 4, fbstr = kf.fbstr; end
if nargin < 3, T_fb = 1; end

if enable_debug
    fprintf('\n========== [kffeedback 内部调试] 开始执行 ==========\n');
    fprintf('输入参数:\n');
    fprintf('  - T_fb = %.6f\n', T_fb);
    fprintf('  - fbstr = ''%s''\n', fbstr);
    fprintf('  - kf.n = %d\n', kf.n);
    if isfield(kf, 'xk')
        fprintf('  - kf.xk 大小: [%d, %d]\n', size(kf.xk, 1), size(kf.xk, 2));
        fprintf('  - kf.xk (前9个): [');
        xk_show = kf.xk;
        if size(xk_show, 2) > 1
            xk_show = xk_show(:, 1);
        end
        for i = 1:min(9, length(xk_show))
            fprintf('%.6e ', xk_show(i));
        end
        fprintf(']\n');
    end
    if isfield(kf, 'coef_fb')
        fprintf('  - kf.coef_fb 大小: [%d, %d]\n', size(kf.coef_fb, 1), size(kf.coef_fb, 2));
    end
    if isfield(kf, 'xtau')
        fprintf('  - kf.xtau 大小: [%d, %d]\n', size(kf.xtau, 1), size(kf.xtau, 2));
    end
    if isfield(ins, 'vn')
        fprintf('  - ins.vn (速度, 米/秒): [%.6f, %.6f, %.6f]\n', ins.vn(1), ins.vn(2), ins.vn(3));
    end
    if isfield(ins, 'pos')
        fprintf('  - ins.pos (位置, LLH弧度): [%.10f, %.10f, %.6f]\n', ins.pos(1), ins.pos(2), ins.pos(3));
    end
end

% 步骤1: 处理反馈时间间隔和系数
if enable_debug
    fprintf('\n步骤1: 处理反馈时间间隔和系数\n');
end

if kf.T_fb ~= T_fb
    if enable_debug
        fprintf('  - kf.T_fb (%f) != T_fb (%f)，需要更新 coef_fb\n', kf.T_fb, T_fb);
    end
    kf.T_fb = T_fb;
    idx = kf.T_fb > kf.xtau;  % scale<vector
    kf.coef_fb(idx) = 1;
    kf.coef_fb(~idx) = kf.T_fb ./ kf.xtau(~idx);   %2022-6-25
    kf.coef_fb(kf.xtau < 0.001 & kf.T_fb ~= inf) = 1;
    
    if enable_debug
        fprintf('  - 更新后的 kf.coef_fb (前9个): [');
        for i = 1:min(9, length(kf.coef_fb))
            fprintf('%.6e ', kf.coef_fb(i));
        end
        fprintf(']\n');
    end
end

% 步骤2: 计算反馈量
if enable_debug
    fprintf('\n步骤2: 计算反馈量 xfb_tmp = kf.coef_fb .* kf.xk\n');
end

xfb_tmp = kf.coef_fb .* kf.xk;
xfb = xfb_tmp * 0;

if enable_debug
    fprintf('  - xfb_tmp 大小: [%d, %d]\n', size(xfb_tmp, 1), size(xfb_tmp, 2));
    fprintf('  - xfb_tmp (前9个): [');
    xfb_tmp_show = xfb_tmp;
    if size(xfb_tmp_show, 2) > 1
        xfb_tmp_show = xfb_tmp_show(:, 1);
    end
    for i = 1:min(9, length(xfb_tmp_show))
        fprintf('%.6e ', xfb_tmp_show(i));
    end
    fprintf(']\n');
    
    % 检查是否有异常大的值
    if any(abs(xfb_tmp_show) > 1e6)
        fprintf('  ⚠️  警告：xfb_tmp 中有异常大的值！\n');
        fprintf('  - 最大绝对值: %.6e\n', max(abs(xfb_tmp_show)));
    end
end

% 步骤3: 根据反馈字符串逐个处理
if enable_debug
    fprintf('\n步骤3: 根据反馈字符串 ''%s'' 逐个处理\n', fbstr);
end

for k = 1:length(fbstr)
    if enable_debug
        fprintf('  处理字符 %d/%d: ''%c''\n', k, length(fbstr), fbstr(k));
    end
    
    switch fbstr(k)
        case 'a',
            idx = 1:3;
            if enable_debug
                fprintf('    - 反馈姿态 (idx = 1:3)\n');
                fprintf('    - xfb_tmp(idx): [%.6e, %.6e, %.6e]\n', xfb_tmp(idx(1)), xfb_tmp(idx(2)), xfb_tmp(idx(3)));
            end
            ins.qnb = qdelphi(ins.qnb, xfb_tmp(idx));
        case 'v',
            idx = 4:6;
            if enable_debug
                fprintf('    - 反馈速度 (idx = 4:6)\n');
                fprintf('    - 反馈前 ins.vn: [%.6f, %.6f, %.6f]\n', ins.vn(1), ins.vn(2), ins.vn(3));
                fprintf('    - xfb_tmp(idx): [%.6e, %.6e, %.6e]\n', xfb_tmp(idx(1)), xfb_tmp(idx(2)), xfb_tmp(idx(3)));
            end
            ins.vn = ins.vn - xfb_tmp(idx);
            if enable_debug
                fprintf('    - 反馈后 ins.vn: [%.6f, %.6f, %.6f]\n', ins.vn(1), ins.vn(2), ins.vn(3));
                if any(abs(ins.vn) > 1e4)
                    fprintf('    ⚠️  警告：反馈后速度异常大！可能导致数值爆炸！\n');
                end
            end
        case 'p',
            idx = 7:9;
            if enable_debug
                fprintf('    - 反馈位置 (idx = 7:9)\n');
                fprintf('    - 反馈前 ins.pos: [%.10f, %.10f, %.6f]\n', ins.pos(1), ins.pos(2), ins.pos(3));
                fprintf('    - xfb_tmp(idx): [%.6e, %.6e, %.6e]\n', xfb_tmp(idx(1)), xfb_tmp(idx(2)), xfb_tmp(idx(3)));
            end
            ins.pos = ins.pos - xfb_tmp(idx);
            if enable_debug
                fprintf('    - 反馈后 ins.pos: [%.10f, %.10f, %.6f]\n', ins.pos(1), ins.pos(2), ins.pos(3));
            end
        case 'e',
            idx = 10:12;
            if enable_debug
                fprintf('    - 反馈陀螺零偏 (idx = 10:12)\n');
            end
            ins.eb = ins.eb + xfb_tmp(idx);
        case 'd',
            idx = 13:15;
            if enable_debug
                fprintf('    - 反馈加速度计零偏 (idx = 13:15)\n');
            end
            ins.db = ins.db + xfb_tmp(idx);
        case 'A',
            idx = 1:2;
            if enable_debug
                fprintf('    - 反馈部分姿态 (idx = 1:2)\n');
            end
            ins.qnb = qdelphi(ins.qnb, [xfb_tmp(idx); 0]);
        case 'V',
            idx = 6;
            if enable_debug
                fprintf('    - 反馈垂直速度 (idx = 6)\n');
                fprintf('    - 反馈前 ins.vn(3): %.6f\n', ins.vn(3));
                fprintf('    - xfb_tmp(idx): %.6e\n', xfb_tmp(idx));
            end
            ins.vn(3) = ins.vn(3) - xfb_tmp(idx);
            if enable_debug
                fprintf('    - 反馈后 ins.vn(3): %.6f\n', ins.vn(3));
            end
        case 'P',
            idx = 9;
            if enable_debug
                fprintf('    - 反馈高度 (idx = 9)\n');
            end
            ins.pos(3) = ins.pos(3) - xfb_tmp(idx);
        case 'E',
            idx = 10:11;
            if enable_debug
                fprintf('    - 反馈部分陀螺零偏 (idx = 10:11)\n');
            end
            ins.eb(1:2) = ins.eb(1:2) + xfb_tmp(idx);
        case 'D',
            idx = 15;
            if enable_debug
                fprintf('    - 反馈垂直加速度计零偏 (idx = 15)\n');
            end
            ins.db(3) = ins.db(3) + xfb_tmp(idx);
        case 'L',
            idx = 16:18;
            if enable_debug
                fprintf('    - 反馈杆臂 (idx = 16:18)\n');
            end
            ins.lever = ins.lever + xfb_tmp(idx);
        case 'T',
            idx = 19;
            if enable_debug
                fprintf('    - 反馈时间延迟 (idx = 19)\n');
            end
            ins.tDelay = ins.tDelay + xfb_tmp(idx);
        case 'G',
            idx = 20:28;
            if enable_debug
                fprintf('    - 反馈陀螺标度因子 (idx = 20:28)\n');
            end
            dKg = xfb_tmp(idx);
            dKg = [dKg(1:3), dKg(4:6), dKg(7:9)];
            ins.Kg = (eye(3) - dKg) * ins.Kg;
        case 'C',
            idx = 29:34;
            if enable_debug
                fprintf('    - 反馈加速度计标度因子 (idx = 29:34)\n');
            end
            dKa = xfb_tmp(idx);
            dKa = [dKa(1:3), [0; dKa(4:5)], [0; 0; dKa(6)]];
            ins.Ka = (eye(3) - dKa) * ins.Ka;
        case 'H',
            idx = [6, 9, 15];
            if enable_debug
                fprintf('    - 反馈高度相关 (idx = [6, 9, 15])\n');
            end
            ins.vn(3) = ins.vn(3) - xfb_tmp(6);
            ins.pos(3) = ins.pos(3) - xfb_tmp(9);
            ins.db(3) = ins.db(3) + xfb_tmp(15);
        otherwise,
            error('feedback string mismatch in kf_feedback');
    end
    
    % 更新状态和反馈记录
    if enable_debug
        fprintf('    - 更新 kf.xk(idx) = kf.xk(idx) - xfb_tmp(idx)\n');
        fprintf('    - 更新 kf.xfb(idx) = kf.xfb(idx) + xfb_tmp(idx)\n');
    end
    
    kf.xk(idx) = kf.xk(idx) - xfb_tmp(idx);
    kf.xfb(idx) = kf.xfb(idx) + xfb_tmp(idx);
    xfb(idx) = xfb_tmp(idx);
    
    if enable_debug
        % 检查更新后的状态
        if any(strcmp(fbstr(k), {'v', 'V', 'H'}))
            fprintf('    - 当前 ins.vn: [%.6f, %.6f, %.6f]\n', ins.vn(1), ins.vn(2), ins.vn(3));
            if any(abs(ins.vn) > 1e4)
                fprintf('    ⚠️  警告：速度异常大！\n');
            end
        end
    end
end

% 步骤4: 同步姿态
if enable_debug
    fprintf('\n步骤4: 同步姿态 [ins.qnb, ins.att, ins.Cnb] = attsyn(ins.qnb)\n');
end

[ins.qnb, ins.att, ins.Cnb] = attsyn(ins.qnb);
ins.avp = [ins.att; ins.vn; ins.pos];  % 2015-2-22

if enable_debug
    fprintf('\n步骤5: 最终状态\n');
    fprintf('  - ins.att (姿态, 弧度): [%.6f, %.6f, %.6f]\n', ins.att(1), ins.att(2), ins.att(3));
    fprintf('  - ins.vn (速度, 米/秒): [%.6f, %.6f, %.6f]\n', ins.vn(1), ins.vn(2), ins.vn(3));
    fprintf('  - ins.pos (位置, LLH弧度): [%.10f, %.10f, %.6f]\n', ins.pos(1), ins.pos(2), ins.pos(3));
    
    if isfield(kf, 'xfb')
        fprintf('  - kf.xfb (总反馈量, 前9个): [');
        xfb_show = kf.xfb;
        if size(xfb_show, 2) > 1
            xfb_show = xfb_show(:);
        end
        for i = 1:min(9, length(xfb_show))
            fprintf('%.6e ', xfb_show(i));
        end
        fprintf(']\n');
    end
    
    fprintf('========== [kffeedback 内部调试] 执行完成 ==========\n\n');
end

end


