% 真实飞行轨迹三维绘图脚本
% 功能：绘制无人机真实XYZ位置的三维飞行轨迹
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

% 读取指定的CSV文件
filename = 'navigation_log_20251219_173957_custom_figure8_baseline_70p20sint.csv';
filepath = fullfile(data_dir, filename);

fprintf('读取文件: %s\n', filename);

if ~exist(filepath, 'file')
    error('文件不存在: %s', filepath);
end

% 读取数据
data = readtable(filepath);

% 提取真实位置（XYZ三个方向）
real_px = data.real_px;
real_py = data.real_py;
real_pz = data.real_pz;

% 提取时间（用于颜色映射）
time = data.time;
time = time - time(1);  % 时间归零，从0开始

% ==================== 创建三维轨迹图 ====================
% 调整图形尺寸以适应双栏A4论文
figure('Position', [100, 100, 900, 750], 'Name', '真实飞行轨迹三维图');

% 绘制三维轨迹
% 使用plot3绘制完整轨迹，线宽2.4，深蓝色（稍微加粗以便在论文中清晰）
plot3(real_px, real_py, real_pz, '-', 'LineWidth', 2.4, ...
    'Color', [0.12, 0.47, 0.71], 'DisplayName', '飞行轨迹');
hold on;

% 标记起点和终点
plot3(real_px(1), real_py(1), real_pz(1), 'o', 'MarkerSize', 14, ...
    'MarkerFaceColor', [0, 0.8, 0], 'MarkerEdgeColor', [0, 0.6, 0], ...
    'LineWidth', 2.2, 'DisplayName', '起点');
plot3(real_px(end), real_py(end), real_pz(end), 's', 'MarkerSize', 14, ...
    'MarkerFaceColor', [0.8, 0, 0], 'MarkerEdgeColor', [0.6, 0, 0], ...
    'LineWidth', 2.2, 'DisplayName', '终点');

% 设置坐标轴标签和标题
xlabel('东向位置/m', 'FontName', 'SimSun', 'FontSize', 18);
ylabel('北向位置/m', 'FontName', 'SimSun', 'FontSize', 18);
zlabel('天向位置/m', 'FontName', 'SimSun', 'FontSize', 18);
title('无人机真实飞行轨迹 - 风速 8.5+2.4sin(t) m/s', 'FontName', 'SimSun', 'FontSize', 20);

% 设置坐标轴刻度字体
set(gca, 'FontSize', 16);

% 添加网格和边框
grid on;
box on;

% 设置视角（可选：调整视角以获得更好的观察角度）
view(45, 30);  % 方位角45度，仰角30度

% 添加图例
legend('Location', 'best', 'FontName', 'SimSun', 'FontSize', 16);

% 设置坐标轴等比例（可选，根据数据范围决定是否启用）
% axis equal;

% 保存图片（高分辨率，PNG + EMF）
% 确保高清输出，适合论文使用
base_name = sprintf('real_trajectory_3d_%s', extractBefore(filename, '.csv'));
set(gcf, 'PaperPositionMode', 'auto');
% 输出600 dpi PNG格式（高清位图）
print(gcf, fullfile(output_dir, [base_name, '.png']), '-dpng', '-r600');
% 输出EMF矢量格式（适合Word插入，不会模糊）
print(gcf, fullfile(output_dir, [base_name, '.emf']), '-dmeta', '-r600');

fprintf('已保存: %s.png 和 %s.emf\n', base_name, base_name);
fprintf('绘图完成！\n');

