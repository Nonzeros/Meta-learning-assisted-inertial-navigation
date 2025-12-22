% 导航结果绘图脚本
% 功能：从CSV文件读取数据并绘制多张对比图
% 作者：Auto-generated
% 日期：2025-12-15

clear; close all; clc;

% 设置字体（中文用宋体，英文和数字用Times New Roman）
set(0,'DefaultAxesFontName','Times New Roman');
set(0,'DefaultTextFontName','Times New Roman');
% 注意：MATLAB中设置中文字体需要在绘图时单独指定

% 文件路径配置（相对于脚本所在目录）
script_dir = fileparts(mfilename('fullpath'));
data_dir = fullfile(script_dir, '..', 'plot_data');
output_dir = fullfile(script_dir, 'plot_output');

% 确保输出目录存在
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% 文件列表和对应的风速标题
file_config = {
    'navigation_log_20251219_181003_custom_figure8_baseline_nowind.csv', '风速 0 m/s';
    'navigation_log_20251219_175958_custom_figure8_baseline_100wind.csv', '风速 12.1 m/s';
    'navigation_log_20251219_174958_custom_figure8_baseline_70wind.csv', '风速 8.5 m/s';
    'navigation_log_20251219_173957_custom_figure8_baseline_70p20sint.csv', '风速 8.5+2.4sin(t) m/s';
    'navigation_log_20251219_173003_custom_figure8_baseline_35wind.csv', '风速 4.2 m/s';
};

% 遍历每个文件
for file_idx = 1:size(file_config, 1)
    filename = file_config{file_idx, 1};
    wind_title = file_config{file_idx, 2};
    
    fprintf('处理文件: %s\n', filename);
    
    % 读取CSV文件
    filepath = fullfile(data_dir, filename);
    if ~exist(filepath, 'file')
        warning('文件不存在: %s', filepath);
        continue;
    end
    
    % 读取数据
    data = readtable(filepath);
    
    % 提取时间，并从0时刻开始
    time = data.time;
    time = time - time(1);  % 时间归零，从0开始
    
    % ==================== 第一张图：闭环融合的东向速度比较 ====================
    figure('Position', [100, 100, 900, 500], 'Name', '闭环融合东向速度比较');
    
    hold on;
    plot(time, data.real_vx, '-', 'LineWidth', 2.2, 'DisplayName', '参考值');
    plot(time, data.ukf_fused_vx, '--', 'LineWidth', 2.0, 'DisplayName', '元学习模型辅助惯导');
    plot(time, data.baseline_vel_x, '-.', 'LineWidth', 2.0, 'DisplayName', '零气动力模型辅助惯导');
    plot(time, data.linear_drag_vel_x, ':', 'LineWidth', 2.0, 'DisplayName', '线性阻力气动力辅助惯导');
    % 注意：pure_ins_vx使用更细的虚线以区分
    p5 = plot(time, data.pure_ins_vx, '--', 'LineWidth', 1.4, 'DisplayName', '纯惯导');
    p5.Color = [0.5, 0.5, 0.5];  % 使用灰色以区分
    
    xlabel('时间/s', 'FontName', 'SimSun', 'FontSize', 18);
    ylabel('东向速度(m/s)', 'FontName', 'SimSun', 'FontSize', 18);
    title(sprintf('东向速度比较 - %s', wind_title), 'FontName', 'SimSun', 'FontSize', 20);
    legend('Location', 'best', 'FontName', 'SimSun', 'FontSize', 16);
    grid on;
    box on;
    set(gca, 'FontSize', 16);  % 坐标轴刻度字体
    hold off;
    
    % 保存图片（高分辨率，PNG + EMF 方便插入Word）
    base_name = sprintf('fig1_velocity_closed_loop_%s', extractBefore(filename, '.csv'));
    set(gcf, 'PaperPositionMode', 'auto');
    print(gcf, fullfile(output_dir, [base_name, '.png']), '-dpng', '-r600');
    print(gcf, fullfile(output_dir, [base_name, '.emf']), '-dmeta', '-r600');
    close(gcf);
    
    % ==================== 第二张图：闭环融合的东向位置比较 ====================
    figure('Position', [100, 100, 900, 500], 'Name', '闭环融合东向位置比较');
    
    hold on;
    plot(time, data.real_px, '-', 'LineWidth', 2.2, 'DisplayName', '参考值');
    plot(time, data.ukf_fused_px, '--', 'LineWidth', 2.0, 'DisplayName', '元学习模型辅助惯导');
    plot(time, data.baseline_ukf_fused_px, '-.', 'LineWidth', 2.0, 'DisplayName', '零气动力模型辅助惯导');
    plot(time, data.linear_drag_pos_x, ':', 'LineWidth', 2.0, 'DisplayName', '线性阻力气动力辅助惯导');
    % 注意：pure_ins_px使用更细的虚线以区分
    p5_p = plot(time, data.pure_ins_px, '--', 'LineWidth', 1.4, 'DisplayName', '纯惯导');
    p5_p.Color = [0.5, 0.5, 0.5];  % 使用灰色以区分
    
    xlabel('时间/s', 'FontName', 'SimSun', 'FontSize', 18);
    ylabel('东向位置/m', 'FontName', 'SimSun', 'FontSize', 18);
    title(sprintf('东向位置比较 - %s', wind_title), 'FontName', 'SimSun', 'FontSize', 20);
    legend('Location', 'best', 'FontName', 'SimSun', 'FontSize', 16);
    grid on;
    box on;
    set(gca, 'FontSize', 16);
    hold off;
    
    % 保存图片（高分辨率，PNG + EMF）
    base_name = sprintf('fig2_position_closed_loop_%s', extractBefore(filename, '.csv'));
    set(gcf, 'PaperPositionMode', 'auto');
    print(gcf, fullfile(output_dir, [base_name, '.png']), '-dpng', '-r600');
    print(gcf, fullfile(output_dir, [base_name, '.emf']), '-dmeta', '-r600');
    close(gcf);
    
    % ==================== 第三张图：x方向总力绘制比较 ====================
    figure('Position', [100, 100, 900, 500], 'Name', 'x方向总力比较');
    
    hold on;
    plot(time, data.real_fa_total_x, '-', 'LineWidth', 2.2, 'DisplayName', '真实值');
    plot(time, data.neural_f_total_x, '--', 'LineWidth', 2.0, 'DisplayName', '元学习模型');
    plot(time, data.baseline_f_total_x, '-.', 'LineWidth', 2.0, 'DisplayName', '零气动力模型');
    plot(time, data.linear_drag_f_total_x, ':', 'LineWidth', 2.0, 'DisplayName', '线性阻力模型');
    
    xlabel('时间 (s)', 'FontName', 'SimSun', 'FontSize', 18);
    ylabel('x方向总力 (N)', 'FontName', 'SimSun', 'FontSize', 18);
    title(sprintf('x方向总力比较 - %s', wind_title), 'FontName', 'SimSun', 'FontSize', 20);
    legend('Location', 'best', 'FontName', 'SimSun', 'FontSize', 16);
    grid on;
    box on;
    set(gca, 'FontSize', 16);
    hold off;
    
    % 保存图片（高分辨率，PNG + EMF）
    base_name = sprintf('fig3_total_force_x_%s', extractBefore(filename, '.csv'));
    set(gcf, 'PaperPositionMode', 'auto');
    print(gcf, fullfile(output_dir, [base_name, '.png']), '-dpng', '-r600');
    print(gcf, fullfile(output_dir, [base_name, '.emf']), '-dmeta', '-r600');
    close(gcf);
    
    % ==================== 第四张图：开环速度比较 ====================
    figure('Position', [100, 100, 900, 500], 'Name', '开环速度比较');
    
    hold on;
    plot(time, data.real_vx, '-', 'LineWidth', 2.2, 'DisplayName', '参考值');
    plot(time, data.open_loop_intelligent_vx, '--', 'LineWidth', 2.0, 'DisplayName', '元学习模型开环');
    plot(time, data.open_loop_baseline_vx, '-.', 'LineWidth', 2.0, 'DisplayName', '零气动力模型开环');
    plot(time, data.open_loop_linear_drag_vx, ':', 'LineWidth', 2.0, 'DisplayName', '线性阻力模型开环');
    
    xlabel('时间 (s)', 'FontName', 'SimSun', 'FontSize', 18);
    ylabel('东向速度 (m/s)', 'FontName', 'SimSun', 'FontSize', 18);
    title(sprintf('开环速度比较 - %s', wind_title), 'FontName', 'SimSun', 'FontSize', 20);
    legend('Location', 'best', 'FontName', 'SimSun', 'FontSize', 16);
    grid on;
    box on;
    set(gca, 'FontSize', 16);
    hold off;
    
    % 保存图片（高分辨率，PNG + EMF）
    base_name = sprintf('fig4_velocity_open_loop_%s', extractBefore(filename, '.csv'));
    set(gcf, 'PaperPositionMode', 'auto');
    print(gcf, fullfile(output_dir, [base_name, '.png']), '-dpng', '-r600');
    print(gcf, fullfile(output_dir, [base_name, '.emf']), '-dmeta', '-r600');
    close(gcf);
    
    % ==================== 第五张图：闭环速度比较 ====================
    figure('Position', [100, 100, 900, 500], 'Name', '闭环速度比较');
    
    hold on;
    plot(time, data.real_vx, '-', 'LineWidth', 2.2, 'DisplayName', '参考值');
    plot(time, data.ukf_fused_vx, '--', 'LineWidth', 2.0, 'DisplayName', '元学习模型闭环');
    plot(time, data.baseline_ukf_fused_vx, '-.', 'LineWidth', 2.0, 'DisplayName', '零气动力模型闭环');
    plot(time, data.linear_drag_ukf_fused_vx, ':', 'LineWidth', 2.0, 'DisplayName', '线性阻力模型闭环');
    % 注意：pure_ins_vx使用更细的虚线以区分
    p5_v = plot(time, data.pure_ins_vx, '--', 'LineWidth', 1.4, 'DisplayName', '纯惯导');
    p5_v.Color = [0.5, 0.5, 0.5];  % 使用灰色以区分
    
    xlabel('时间 (s)', 'FontName', 'SimSun', 'FontSize', 18);
    ylabel('东向速度 (m/s)', 'FontName', 'SimSun', 'FontSize', 18);
    title(sprintf('东向速度比较 - %s', wind_title), 'FontName', 'SimSun', 'FontSize', 20);
    legend('Location', 'best', 'FontName', 'SimSun', 'FontSize', 16);
    grid on;
    box on;
    set(gca, 'FontSize', 16);
    hold off;
    
    % 保存图片（高分辨率，PNG + EMF）
    base_name = sprintf('fig5_velocity_closed_loop_%s', extractBefore(filename, '.csv'));
    set(gcf, 'PaperPositionMode', 'auto');
    print(gcf, fullfile(output_dir, [base_name, '.png']), '-dpng', '-r600');
    print(gcf, fullfile(output_dir, [base_name, '.emf']), '-dmeta', '-r600');
    close(gcf);
    
    fprintf('完成文件: %s\n\n', filename);
end

fprintf('所有文件处理完成！\n');

