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

% 选择两个风况进行对比
file_config = {
    'navigation_log_20251219_181003_custom_figure8_baseline_nowind.csv', '风速 0 m/s';
    'navigation_log_20251219_173957_custom_figure8_baseline_70p20sint.csv', '风速 8.5+2.4sin(t) m/s';
};

% 读取两个文件的数据
data_nowind = [];
data_sint = [];
time_nowind = [];
time_sint = [];
wind_title_nowind = '';
wind_title_sint = '';

fprintf('读取数据文件...\n');
for file_idx = 1:size(file_config, 1)
    filename = file_config{file_idx, 1};
    wind_title = file_config{file_idx, 2};
    
    fprintf('  读取文件: %s\n', filename);
    
    % 读取CSV文件
    filepath = fullfile(data_dir, filename);
    if ~exist(filepath, 'file')
        warning('文件不存在: %s', filepath);
        continue;
    end
    
    % 读取数据
    data_temp = readtable(filepath);
    
    % 提取时间，并从0时刻开始
    time_temp = data_temp.time;
    time_temp = time_temp - time_temp(1);  % 时间归零，从0开始
    
    if file_idx == 1  % no wind
        data_nowind = data_temp;
        time_nowind = time_temp;
        wind_title_nowind = wind_title;
    else  % 8.5+sin(t)
        data_sint = data_temp;
        time_sint = time_temp;
        wind_title_sint = wind_title;
    end
end

% 提取真实剩余气动力（XYZ三个方向）
% no wind 数据
real_fa_x_nowind = data_nowind.real_fa_x;
real_fa_y_nowind = data_nowind.real_fa_y;
real_fa_z_nowind = data_nowind.real_fa_z;

% 8.5+sin(t) 数据
real_fa_x_sint = data_sint.real_fa_x;
real_fa_y_sint = data_sint.real_fa_y;
real_fa_z_sint = data_sint.real_fa_z;

% ==================== 创建包含3个子图的图形 ====================
% 调整图形尺寸以适应双栏A4论文
figure('Position', [100, 100, 900, 750], 'Name', '真实剩余气动力对比');

% 子图1：X方向（东向）对比
subplot(3, 1, 1);
hold on;
plot(time_nowind, real_fa_x_nowind, '-', 'LineWidth', 2.2, 'Color', [0.12, 0.47, 0.71], 'DisplayName', wind_title_nowind);
plot(time_sint, real_fa_x_sint, '--', 'LineWidth', 2.2, 'Color', [0.89, 0.47, 0.20], 'DisplayName', wind_title_sint);
xlabel('时间/s', 'FontName', 'SimSun', 'FontSize', 18);
ylabel('东向气动力/N', 'FontName', 'SimSun', 'FontSize', 18);
title('东向参考剩余气动力对比', 'FontName', 'SimSun', 'FontSize', 20);
legend('Location', 'best', 'FontName', 'SimSun', 'FontSize', 16);
grid on;
box on;
set(gca, 'FontSize', 16);
hold off;

% 子图2：Y方向（北向）对比
subplot(3, 1, 2);
hold on;
plot(time_nowind, real_fa_y_nowind, '-', 'LineWidth', 2.2, 'Color', [0.12, 0.47, 0.71], 'DisplayName', wind_title_nowind);
plot(time_sint, real_fa_y_sint, '--', 'LineWidth', 2.2, 'Color', [0.89, 0.47, 0.20], 'DisplayName', wind_title_sint);
xlabel('时间/s', 'FontName', 'SimSun', 'FontSize', 18);
ylabel('北向气动力/N', 'FontName', 'SimSun', 'FontSize', 18);
title('北向参考剩余气动力对比', 'FontName', 'SimSun', 'FontSize', 20);
legend('Location', 'best', 'FontName', 'SimSun', 'FontSize', 16);
grid on;
box on;
set(gca, 'FontSize', 16);
hold off;

% 子图3：Z方向（天向）对比
subplot(3, 1, 3);
hold on;
plot(time_nowind, real_fa_z_nowind, '-', 'LineWidth', 2.2, 'Color', [0.12, 0.47, 0.71], 'DisplayName', wind_title_nowind);
plot(time_sint, real_fa_z_sint, '--', 'LineWidth', 2.2, 'Color', [0.89, 0.47, 0.20], 'DisplayName', wind_title_sint);
xlabel('时间/s', 'FontName', 'SimSun', 'FontSize', 18);
ylabel('天向气动力/N', 'FontName', 'SimSun', 'FontSize', 18);
title('天向参考剩余气动力对比', 'FontName', 'SimSun', 'FontSize', 20);
legend('Location', 'best', 'FontName', 'SimSun', 'FontSize', 16);
grid on;
box on;
set(gca, 'FontSize', 16);
hold off;

% 保存图片（高分辨率，PNG + EMF）
base_name = 'real_aerodynamic_force_comparison_nowind_vs_sint';
set(gcf, 'PaperPositionMode', 'auto');
print(gcf, fullfile(output_dir, [base_name, '.png']), '-dpng', '-r600');
print(gcf, fullfile(output_dir, [base_name, '.emf']), '-dmeta', '-r600');

fprintf('对比图已保存: %s.png 和 %s.emf\n', base_name, base_name);
fprintf('绘图完成！\n');


