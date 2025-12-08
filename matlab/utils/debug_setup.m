% ========== MATLAB 调试设置脚本 ==========
% 使用方法：
%   1. 在 MATLAB 命令行中运行：debug_setup
%   2. 或者在 Python 调用 MATLAB 之前，在 Python 中执行：
%      eng.eval("debug_setup", nargout=0)
%
% 这个脚本会启用以下调试功能：
%   - 自动在错误处断点
%   - 在指定文件的指定行设置断点
%   - 显示详细的错误堆栈信息
% =========================================

fprintf('========== 启用 MATLAB 调试模式 ==========\n');

% 启用自动错误断点
dbstop if error
fprintf('✓ 已启用：自动在错误处断点 (dbstop if error)\n');

% 启用自动警告断点（可选，如果警告太多可以注释掉）
% dbstop if warning
% fprintf('✓ 已启用：自动在警告处断点 (dbstop if warning)\n');

% 在关键位置设置断点（通过代码设置，不依赖 MATLAB IDE）
% 获取当前文件路径
current_file = mfilename('fullpath');
[filepath, ~, ~] = fileparts(current_file);
test_file = fullfile(filepath, 'test_SINS_dynamic_UKF_153_forpython.m');

% 检查文件是否存在
if exist(test_file, 'file')
    % 在 kffeedback 调用前设置断点（第69行）
    try
        dbstop('in', 'test_SINS_dynamic_UKF_153_forpython', 'at', '69');
        fprintf('✓ 已在 test_SINS_dynamic_UKF_153_forpython.m 第69行设置断点\n');
    catch
        fprintf('⚠ 无法在 test_SINS_dynamic_UKF_153_forpython.m 设置断点（可能文件未加载）\n');
    end
    
    % 在 ukf 调用后设置断点（第29行）
    try
        dbstop('in', 'test_SINS_dynamic_UKF_153_forpython', 'at', '29');
        fprintf('✓ 已在 test_SINS_dynamic_UKF_153_forpython.m 第29行设置断点\n');
    catch
        fprintf('⚠ 无法在 test_SINS_dynamic_UKF_153_forpython.m 第29行设置断点\n');
    end
else
    fprintf('⚠ 未找到 test_SINS_dynamic_UKF_153_forpython.m 文件\n');
end

% 显示详细的错误信息
warning('on', 'all');
fprintf('✓ 已启用：显示所有警告\n');

% 显示当前所有断点状态
fprintf('\n当前断点状态：\n');
dbstatus
fprintf('========== 调试模式已启用 ==========\n');
fprintf('重要提示：\n');
fprintf('  - Python 启动的 MATLAB 引擎是独立进程，不是 MATLAB IDE\n');
fprintf('  - 当执行到断点时，Python 会卡住等待\n');
fprintf('  - 你需要通过 Python 的 MATLAB 引擎接口来调试\n');
fprintf('  - 或者查看 Python 控制台的错误输出来定位问题\n');
fprintf('  - 常用调试方法：\n');
fprintf('    1. 查看 Python 控制台的错误信息\n');
fprintf('    2. 在 MATLAB 文件中添加 fprintf 输出调试信息\n');
fprintf('    3. 使用 dbstop if error 自动在错误处停住\n');
fprintf('==========================================\n');

