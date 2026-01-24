% 100wind数据绘图脚本（带局部放大功能）
% 功能：绘制速度和位置时间序列对比图，并添加局部放大视图
% 作者：Auto-generated
% 日期：2025-12-15

clear; close all; clc;

% 设置字体
set(0,'DefaultAxesFontName','Times New Roman');
set(0,'DefaultTextFontName','Times New Roman');

% 文件路径配置
script_dir = fileparts(mfilename('fullpath'));
data_dir = fullfile(script_dir, '..', 'plot_data');
output_dir = fullfile(script_dir, 'plot_output');

% 确保输出目录存在
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% 读取100wind文件
filename = 'navigation_log_20251229_234229_custom_figure8_baseline_70p20sint.csv';
filepath = fullfile(data_dir, filename);

fprintf('读取文件: %s\n', filename);
data = readtable(filepath);

% 提取时间，并从0时刻开始
time = data.time;
time = time - time(1);  % 时间归零，从0开始

% ============ 设置放大区域的时间范围（可手动调整） ============
% 选择一个时间段进行放大显示，建议选择曲线变化较明显的区域
zoom_time_start = 12;   % 放大区域起始时间（秒）
zoom_time_end = 15;     % 放大区域结束时间（秒）

% 找到放大区域对应的数据索引
zoom_idx = (time >= zoom_time_start) & (time <= zoom_time_end);

%% ==================== 图1：东向速度比较（带放大） ====================
figure('Position', [100, 100, 1100, 650], 'Name', '东向速度比较');

% 主图（略窄一些，留出排版边距）
ax_main = axes('Position', [0.10, 0.12, 0.85, 0.80]);
hold on;

% 绘制所有曲线（使用闭环融合结果）
h1 = plot(time, data.real_vx, '-', 'LineWidth', 2.6, 'DisplayName', '参考值');
h2 = plot(time, data.ukf_fused_vx, '--', 'LineWidth', 2.3, 'DisplayName', '元学习模型');
h3 = plot(time, data.baseline_ukf_fused_vx, '-.', 'LineWidth', 2.3, 'DisplayName', '零气动力模型');
h4 = plot(time, data.linear_drag_ukf_fused_vx, ':', 'LineWidth', 2.3, 'DisplayName', '线性阻力模型');
h5 = plot(time, data.pure_ins_vx, '--', 'LineWidth', 1.8, 'DisplayName', '纯惯导');
h5.Color = [0.5, 0.5, 0.5];

% 标注放大区域（用矩形框）
% 首先确定y轴范围（基于所有曲线）
all_y_data = [data.real_vx; data.ukf_fused_vx; data.baseline_ukf_fused_vx; 
              data.linear_drag_ukf_fused_vx; data.pure_ins_vx];
y_min_all = min(all_y_data);
y_max_all = max(all_y_data);
y_range = y_max_all - y_min_all;

% 计算矩形框的y范围（基于放大区域内的数据）
y_data_zoom_box = [data.real_vx(zoom_idx); data.ukf_fused_vx(zoom_idx); 
                   data.linear_drag_ukf_fused_vx(zoom_idx); data.pure_ins_vx(zoom_idx)];
y_min_box = min(y_data_zoom_box);
y_max_box = max(y_data_zoom_box);
rect_y_start = y_min_box;
rect_height = y_max_box - y_min_box;

% 添加一些边距使矩形框更清晰
y_margin_box = rect_height * 0.05;  % 5%边距
rect_y_start = rect_y_start - y_margin_box;
rect_height = rect_height + 2 * y_margin_box;

% 绘制矩形框
rectangle('Position', [zoom_time_start, rect_y_start, zoom_time_end-zoom_time_start, rect_height], ...
    'EdgeColor', 'r', 'LineWidth', 1.5);

xlabel('时间 (s)', 'FontName', 'SimSun', 'FontSize', 19);
ylabel('东向速度 (m/s)', 'FontName', 'SimSun', 'FontSize', 19);
title('东向速度比较 - 风速 12.1 m/s', 'FontName', 'SimSun', 'FontSize', 21);
legend('Location', 'southwest', 'FontName', 'SimSun', 'FontSize', 17);
grid on;
box on;
set(gca, 'FontSize', 16);
hold off;

% 创建放大子图（inset axes）
ax_zoom = axes('Position', [0.58, 0.58, 0.36, 0.32]);  % [left, bottom, width, height]
hold on;

% 在放大图中绘制相同的曲线（但只显示放大区域，不含baseline）
% 明确指定颜色以保持与主图一致
p1_zoom = plot(time(zoom_idx), data.real_vx(zoom_idx), '-', 'LineWidth', 2.4);
p1_zoom.Color = h1.Color;  % 与主图参考值颜色一致

p2_zoom = plot(time(zoom_idx), data.ukf_fused_vx(zoom_idx), '--', 'LineWidth', 2.0);
p2_zoom.Color = h2.Color;  % 与主图元学习颜色一致

% 不绘制baseline（h3），因为误差太大会遮挡其他曲线
% 直接绘制线性阻力，并使用主图中h4的颜色
p4_zoom = plot(time(zoom_idx), data.linear_drag_ukf_fused_vx(zoom_idx), ':', 'LineWidth', 2.0);
p4_zoom.Color = h4.Color;  % 与主图线性阻力颜色一致

p5_zoom = plot(time(zoom_idx), data.pure_ins_vx(zoom_idx), '--', 'LineWidth', 1.6);
p5_zoom.Color = [0.5, 0.5, 0.5];  % 灰色

% 设置放大图的x和y范围，使其与矩形框完全匹配
xlim([zoom_time_start, zoom_time_end]);
ylim([rect_y_start, rect_y_start + rect_height]);

% 设置坐标轴刻度字体为Times New Roman（数字）
set(gca, 'FontName', 'Times New Roman', 'FontSize', 15);
% 设置坐标轴标签为宋体（中文）
xlabel('时间 (s)', 'FontName', 'SimSun', 'FontSize', 16);
ylabel('速度 (m/s)', 'FontName', 'SimSun', 'FontSize', 16);
% 去掉标题
grid on;
box on;
hold off;

% 保存图片（高分辨率，PNG + EMF）
set(gcf, 'PaperPositionMode', 'auto');
print(gcf, fullfile(output_dir, 'velocity_100wind_with_zoom.png'), '-dpng', '-r600');
print(gcf, fullfile(output_dir, 'velocity_100wind_with_zoom.emf'), '-dmeta', '-r600');
fprintf('已保存: velocity_100wind_with_zoom.png\n');

%% ==================== 图2：东向位置比较（带放大） ====================
figure('Position', [100, 100, 1100, 650], 'Name', '东向位置比较');

% 主图
ax_main2 = axes('Position', [0.10, 0.12, 0.85, 0.80]);
hold on;

% 绘制所有曲线
h1 = plot(time, data.real_px, '-', 'LineWidth', 2.6, 'DisplayName', '参考值');
h2 = plot(time, data.ukf_fused_px, '--', 'LineWidth', 2.3, 'DisplayName', '元学习模型');
h3 = plot(time, data.baseline_ukf_fused_px, '-.', 'LineWidth', 2.3, 'DisplayName', '零气动力模型');
h4 = plot(time, data.linear_drag_ukf_fused_px, ':', 'LineWidth', 2.3, 'DisplayName', '线性阻力模型');
h5 = plot(time, data.pure_ins_px, '--', 'LineWidth', 1.8, 'DisplayName', '纯惯导');
h5.Color = [0.5, 0.5, 0.5];

% 标注放大区域（用矩形框）
% 首先确定y轴范围（基于所有曲线）
all_y_data2 = [data.real_px; data.ukf_fused_px; data.baseline_ukf_fused_px; 
               data.linear_drag_ukf_fused_px; data.pure_ins_px];
y_min_all2 = min(all_y_data2);
y_max_all2 = max(all_y_data2);

% 计算矩形框的y范围（基于放大区域内的数据）
y_data_zoom_box2 = [data.real_px(zoom_idx); data.ukf_fused_px(zoom_idx); 
                    data.linear_drag_ukf_fused_px(zoom_idx); data.pure_ins_px(zoom_idx)];
y_min_box2 = min(y_data_zoom_box2);
y_max_box2 = max(y_data_zoom_box2);
rect_y_start2 = y_min_box2;
rect_height2 = y_max_box2 - y_min_box2;

% 添加一些边距使矩形框更清晰
y_margin_box2 = rect_height2 * 0.05;  % 5%边距
rect_y_start2 = rect_y_start2 - y_margin_box2;
rect_height2 = rect_height2 + 2 * y_margin_box2;

% 绘制矩形框
rectangle('Position', [zoom_time_start, rect_y_start2, zoom_time_end-zoom_time_start, rect_height2], ...
    'EdgeColor', 'r', 'LineWidth', 1.5);

xlabel('时间 (s)', 'FontName', 'SimSun', 'FontSize', 19);
ylabel('东向位置 (m)', 'FontName', 'SimSun', 'FontSize', 19);
title('东向位置比较 - 风速 12.1 m/s', 'FontName', 'SimSun', 'FontSize', 21);
legend('Location', 'northwest', 'FontName', 'SimSun', 'FontSize', 17);
grid on;
box on;
set(gca, 'FontSize', 16);
hold off;

% 创建放大子图
ax_zoom2 = axes('Position', [0.58, 0.58, 0.36, 0.32]);
hold on;

% 在放大图中绘制相同的曲线（但不含baseline）
% 明确指定颜色以保持与主图一致
p1_zoom2 = plot(time(zoom_idx), data.real_px(zoom_idx), '-', 'LineWidth', 2.4);
p1_zoom2.Color = h1.Color;  % 与主图参考值颜色一致

p2_zoom2 = plot(time(zoom_idx), data.ukf_fused_px(zoom_idx), '--', 'LineWidth', 2.0);
p2_zoom2.Color = h2.Color;  % 与主图元学习颜色一致

% 不绘制baseline（h3），因为误差太大会遮挡其他曲线
% 直接绘制线性阻力，并使用主图中h4的颜色
p4_zoom2 = plot(time(zoom_idx), data.linear_drag_ukf_fused_px(zoom_idx), ':', 'LineWidth', 2.0);
p4_zoom2.Color = h4.Color;  % 与主图线性阻力颜色一致

p5_zoom2 = plot(time(zoom_idx), data.pure_ins_px(zoom_idx), '--', 'LineWidth', 1.6);
p5_zoom2.Color = [0.5, 0.5, 0.5];  % 灰色

% 设置放大图的范围，使其与矩形框完全匹配
xlim([zoom_time_start, zoom_time_end]);
ylim([rect_y_start2, rect_y_start2 + rect_height2]);

% 设置坐标轴刻度字体为Times New Roman（数字）
set(gca, 'FontName', 'Times New Roman', 'FontSize', 15);
% 设置坐标轴标签为宋体（中文）
xlabel('时间 (s)', 'FontName', 'SimSun', 'FontSize', 16);
ylabel('位置 (m)', 'FontName', 'SimSun', 'FontSize', 16);
% 去掉标题
grid on;
box on;
hold off;

% 保存图片（高分辨率，PNG + EMF）
set(gcf, 'PaperPositionMode', 'auto');
print(gcf, fullfile(output_dir, 'position_100wind_with_zoom.png'), '-dpng', '-r600');
print(gcf, fullfile(output_dir, 'position_100wind_with_zoom.emf'), '-dmeta', '-r600');
fprintf('已保存: position_100wind_with_zoom.png\n');

fprintf('\n绘图完成！\n');
fprintf('提示：如需调整放大区域，请修改脚本中的 zoom_time_start 和 zoom_time_end 参数\n');

