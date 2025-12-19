% 三方向速度RMSE对比绘图脚本
% 功能：计算并绘制东北天三个方向的速度RMSE分组柱状图
% 作者：Auto-generated
% 日期：2025-12-15

clear; close all; clc;

% 设置字体（中文用宋体，英文和数字用Times New Roman）
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

% 文件列表和对应的风速标题
file_config = {
    'navigation_log_20251215_011436_custom_figure8_baseline_nowind.csv', '风速 0 m/s';
    'navigation_log_20251215_010407_custom_figure8_baseline_100wind.csv', '风速 12.1 m/s';
    'navigation_log_20251215_005345_custom_figure8_baseline_70wind.csv', '风速 8.5 m/s';
    'navigation_log_20251215_004317_custom_figure8_baseline_70p20sint.csv', '风速 8.5+sin(t) m/s';
    'navigation_log_20251215_003255_custom_figure8_baseline_35wind.csv', '风速 4.2 m/s';
};

% 适应阶段时间（秒）
adapt_time = 10.0;

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
    
    % 排除适应阶段
    valid_idx = time > adapt_time;
    
    % 提取参考速度（排除适应阶段）
    real_vx = data.real_vx(valid_idx);
    real_vy = data.real_vy(valid_idx);
    real_vz = data.real_vz(valid_idx);
    
    % 提取三种方法的速度（排除适应阶段）
    % 元学习
    meta_vx = data.ukf_fused_vx(valid_idx);
    meta_vy = data.ukf_fused_vy(valid_idx);
    meta_vz = data.ukf_fused_vz(valid_idx);
    
    % 线性阻力
    linear_vx = data.linear_drag_ukf_fused_vx(valid_idx);
    linear_vy = data.linear_drag_ukf_fused_vy(valid_idx);
    linear_vz = data.linear_drag_ukf_fused_vz(valid_idx);
    
    % 纯惯导
    pure_vx = data.pure_ins_vx(valid_idx);
    pure_vy = data.pure_ins_vy(valid_idx);
    pure_vz = data.pure_ins_vz(valid_idx);
    
    % 计算RMSE
    % 元学习
    rmse_meta_x = sqrt(mean((meta_vx - real_vx).^2));
    rmse_meta_y = sqrt(mean((meta_vy - real_vy).^2));
    rmse_meta_z = sqrt(mean((meta_vz - real_vz).^2));
    
    % 线性阻力
    rmse_linear_x = sqrt(mean((linear_vx - real_vx).^2));
    rmse_linear_y = sqrt(mean((linear_vy - real_vy).^2));
    rmse_linear_z = sqrt(mean((linear_vz - real_vz).^2));
    
    % 纯惯导
    rmse_pure_x = sqrt(mean((pure_vx - real_vx).^2));
    rmse_pure_y = sqrt(mean((pure_vy - real_vy).^2));
    rmse_pure_z = sqrt(mean((pure_vz - real_vz).^2));
    
    % 组织数据用于绘图
    % 行：三个方向（东、北、天）
    % 列：三种方法（元学习、线性阻力、纯惯导）
    rmse_data = [
        rmse_meta_x,   rmse_linear_x,   rmse_pure_x;    % 东向
        rmse_meta_y,   rmse_linear_y,   rmse_pure_y;    % 北向
        rmse_meta_z,   rmse_linear_z,   rmse_pure_z     % 天向
    ];
    
    % 绘制分组柱状图
    figure('Position', [100, 100, 900, 600], 'Name', 'RMSE对比');
    
    % 创建分组柱状图
    b = bar(rmse_data);
    
    % 设置柱子颜色 - 使用科研论文级配色方案
    % 参考Nature/Science期刊常用配色，色盲友好且高对比度
    b(1).FaceColor = [0.12, 0.47, 0.71];    % 元学习 - 深蓝色 RGB(31, 119, 180)
    b(2).FaceColor = [0.89, 0.47, 0.20];    % 线性阻力 - 暖橙色 RGB(227, 119, 51)
    b(3).FaceColor = [0.20, 0.63, 0.17];    % 纯惯导 - 深绿色 RGB(51, 160, 44)
    
    % 设置柱子边框，增强视觉效果
    b(1).EdgeColor = [0.10, 0.40, 0.65];
    b(2).EdgeColor = [0.80, 0.42, 0.18];
    b(3).EdgeColor = [0.18, 0.55, 0.15];
    b(1).LineWidth = 1.2;
    b(2).LineWidth = 1.2;
    b(3).LineWidth = 1.2;
    
    % 设置横轴标签
    set(gca, 'XTickLabel', {'东向', '北向', '天向'}, 'FontName', 'SimSun', 'FontSize', 11);
    
    % 设置纵轴标签和标题
    ylabel('RMSE (m/s)', 'FontName', 'Times New Roman', 'FontSize', 12);
    title(sprintf('三方向速度RMSE对比 - %s', wind_title), 'FontName', 'SimSun', 'FontSize', 14);
    
    % 添加图例
    legend({'元学习模型', '线性阻力模型', '纯惯导'}, ...
        'Location', 'best', 'FontName', 'SimSun', 'FontSize', 10);
    
    % 添加网格
    grid on;
    
    % 在柱子上显示数值
    % 获取柱子的x坐标
    for i = 1:3  % 三种方法
        x_data = b(i).XEndPoints;
        y_data = b(i).YEndPoints;
        
        % 在每个柱子上方显示数值
        for j = 1:length(x_data)
            text(x_data(j), y_data(j), sprintf('%.3f', y_data(j)), ...
                'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'bottom', ...
                'FontSize', 9, ...
                'FontName', 'Times New Roman');
        end
    end
    
    % 调整y轴范围以留出显示数值的空间
    ylim_current = ylim;
    ylim([0, ylim_current(2) * 1.15]);
    
    % 保存图片
    output_filename = sprintf('rmse_comparison_%s.png', extractBefore(filename, '.csv'));
    saveas(gcf, fullfile(output_dir, output_filename));
    
    % 输出RMSE数值到控制台
    fprintf('  东向RMSE - 元学习: %.4f, 线性阻力: %.4f, 纯惯导: %.4f\n', ...
        rmse_meta_x, rmse_linear_x, rmse_pure_x);
    fprintf('  北向RMSE - 元学习: %.4f, 线性阻力: %.4f, 纯惯导: %.4f\n', ...
        rmse_meta_y, rmse_linear_y, rmse_pure_y);
    fprintf('  天向RMSE - 元学习: %.4f, 线性阻力: %.4f, 纯惯导: %.4f\n', ...
        rmse_meta_z, rmse_linear_z, rmse_pure_z);
    fprintf('完成文件: %s\n\n', filename);
    
    close(gcf);
end

fprintf('所有RMSE对比图绘制完成！\n');

