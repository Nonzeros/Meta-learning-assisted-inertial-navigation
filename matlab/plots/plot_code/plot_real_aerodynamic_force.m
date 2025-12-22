% 真实剩余气动力绘图脚本
% 功能：绘制XYZ三个方向的真实剩余气动力时间序列
% 作者：Auto-generated
% 日期：2025-12-19

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
    
    % 提取真实剩余气动力（XYZ三个方向）
    real_fa_x = data.real_fa_x;
    real_fa_y = data.real_fa_y;
    real_fa_z = data.real_fa_z;
    
    % ==================== 创建包含3个子图的图形 ====================
    figure('Position', [100, 100, 1000, 800], 'Name', '真实剩余气动力');
    
    % 子图1：X方向（东向）
    subplot(3, 1, 1);
    plot(time, real_fa_x, '-', 'LineWidth', 2.2, 'Color', [0.12, 0.47, 0.71]);
    xlabel('时间 (s)', 'FontName', 'SimSun', 'FontSize', 18);
    ylabel('X方向气动力 (N)', 'FontName', 'SimSun', 'FontSize', 18);
    title(sprintf('X方向（东向）真实剩余气动力 - %s', wind_title), 'FontName', 'SimSun', 'FontSize', 20);
    grid on;
    box on;
    set(gca, 'FontSize', 16);
    
    % 子图2：Y方向（北向）
    subplot(3, 1, 2);
    plot(time, real_fa_y, '-', 'LineWidth', 2.2, 'Color', [0.89, 0.47, 0.20]);
    xlabel('时间 (s)', 'FontName', 'SimSun', 'FontSize', 18);
    ylabel('Y方向气动力 (N)', 'FontName', 'SimSun', 'FontSize', 18);
    title(sprintf('Y方向（北向）真实剩余气动力 - %s', wind_title), 'FontName', 'SimSun', 'FontSize', 20);
    grid on;
    box on;
    set(gca, 'FontSize', 16);
    
    % 子图3：Z方向（天向）
    subplot(3, 1, 3);
    plot(time, real_fa_z, '-', 'LineWidth', 2.2, 'Color', [0.20, 0.63, 0.17]);
    xlabel('时间 (s)', 'FontName', 'SimSun', 'FontSize', 18);
    ylabel('Z方向气动力 (N)', 'FontName', 'SimSun', 'FontSize', 18);
    title(sprintf('Z方向（天向）真实剩余气动力 - %s', wind_title), 'FontName', 'SimSun', 'FontSize', 20);
    grid on;
    box on;
    set(gca, 'FontSize', 16);
    
    % 保存图片（高分辨率，PNG + EMF）
    base_name = sprintf('real_aerodynamic_force_%s', extractBefore(filename, '.csv'));
    set(gcf, 'PaperPositionMode', 'auto');
    print(gcf, fullfile(output_dir, [base_name, '.png']), '-dpng', '-r600');
    print(gcf, fullfile(output_dir, [base_name, '.emf']), '-dmeta', '-r600');
    
    fprintf('完成文件: %s\n\n', filename);
    close(gcf);
end

fprintf('所有真实剩余气动力图绘制完成！\n');



