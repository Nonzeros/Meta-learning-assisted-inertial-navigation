% 训练数据气动力绘图脚本
% 功能：读取training文件夹中的CSV文件，绘制fa（气动力）数据
% 作者：Auto-generated
% 日期：2025-12-20

clear; close all; clc;

% 设置字体（中文用宋体，英文和数字用Times New Roman）
set(0,'DefaultAxesFontName','Times New Roman');
set(0,'DefaultTextFontName','Times New Roman');
% 注意：MATLAB中设置中文字体需要在绘图时单独指定

% 文件路径配置（相对于脚本所在目录）
script_dir = fileparts(mfilename('fullpath'));
project_root = fullfile(script_dir, '..', '..', '..');
training_data_dir = fullfile(project_root, 'data', 'training');
output_dir = fullfile(script_dir, 'plot_output');

% 确保输出目录存在
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% 文件列表和对应的风速标题
file_config = {
    'custom_random3_baseline_nowind.csv', '风速 0 m/s';
    'custom_random3_baseline_10wind.csv', '风速 10 m/s';
    'custom_random3_baseline_20wind.csv', '风速 20 m/s';
    'custom_random3_baseline_30wind.csv', '风速 30 m/s';
    'custom_random3_baseline_40wind.csv', '风速 40 m/s';
    'custom_random3_baseline_50wind.csv', '风速 50 m/s';
};

% 遍历每个文件
for file_idx = 1:size(file_config, 1)
    filename = file_config{file_idx, 1};
    wind_title = file_config{file_idx, 2};
    
    fprintf('处理文件: %s\n', filename);
    
    % 读取CSV文件
    filepath = fullfile(training_data_dir, filename);
    if ~exist(filepath, 'file')
        warning('文件不存在: %s', filepath);
        continue;
    end
    
    % 读取数据
    % CSV列顺序固定：Unnamed: 0, t, p, p_d, v, v_d, q, R, w, T_sp, q_sp, hover_throttle, fa, pwm
    % 直接使用列索引访问：t在第2列，fa在第13列
    data = readtable(filepath);
    
    % 获取列数
    num_cols = width(data);
    
    % 读取时间列（第2列）
    if num_cols >= 2
        time_raw = data{:, 2};
    else
        error('数据表列数不足，无法读取时间列');
    end
    
    % 确保time是数值数组
    if iscell(time_raw)
        % 如果time是cell数组，需要将每个cell转换为数值
        time = zeros(length(time_raw), 1);
        for i = 1:length(time_raw)
            if ischar(time_raw{i}) || isstring(time_raw{i})
                time(i) = str2double(time_raw{i});
            else
                time(i) = double(time_raw{i});
            end
        end
    else
        time = double(time_raw);  % 确保是double类型
    end
    
    % 时间归零，从0开始
    time = time - time(1);
    
    % 解析fa列（字符串格式的列表 [fa_x, fa_y, fa_z]）
    num_rows = height(data);
    fa_x = zeros(num_rows, 1);
    fa_y = zeros(num_rows, 1);
    fa_z = zeros(num_rows, 1);
    
    % 读取fa列（第13列，如果列数不足则使用倒数第二列，因为最后一列通常是pwm）
    if num_cols >= 13
        fa_column = data{:, 13};  % fa列
    elseif num_cols >= 2
        fa_column = data{:, num_cols - 1};  % 倒数第二列（通常是fa）
    else
        error('数据表列数不足，无法读取fa列');
    end
    
    % 使用正则表达式高效解析fa列（格式："[1.2 1.5 1.6]" 或 "[1.2, 1.5, 1.6]"）
    for i = 1:num_rows
        % 处理fa列（可能是cell数组中的字符串）
        if iscell(fa_column)
            fa_str = char(fa_column{i});
        elseif isstring(fa_column)
            fa_str = char(fa_column(i));
        else
            fa_str = char(fa_column(i));
        end
        
        % 检查是否为空或NaN
        if isempty(fa_str) || strcmpi(fa_str, 'nan') || strcmpi(fa_str, 'none')
            fa_x(i) = NaN;
            fa_y(i) = NaN;
            fa_z(i) = NaN;
            continue;
        end
        
        % 移除方括号和空格
        fa_str = strrep(fa_str, '[', '');
        fa_str = strrep(fa_str, ']', '');
        fa_str = strtrim(fa_str);
        
        % 如果字符串为空，跳过
        if isempty(fa_str)
            fa_x(i) = NaN;
            fa_y(i) = NaN;
            fa_z(i) = NaN;
            continue;
        end
        
        % 分割字符串（支持空格或逗号分隔）
        if contains(fa_str, ',')
            % 逗号分隔
            parts = strsplit(fa_str, ',');
        else
            % 空格分隔（移除多个连续空格）
            fa_str = regexprep(fa_str, '\s+', ' ');
            parts = strsplit(fa_str, ' ');
        end
        
        % 移除空字符串
        parts = parts(~cellfun(@isempty, parts));
        
        % 提取数值（处理1个、2个或3个数值的情况）
        num_values = length(parts);
        if num_values >= 3
            % 标准格式：3个数值
            fa_x(i) = str2double(strtrim(parts{1}));
            fa_y(i) = str2double(strtrim(parts{2}));
            fa_z(i) = str2double(strtrim(parts{3}));
        elseif num_values == 2
            % 只有2个数值：假设第3个为0
            fa_x(i) = str2double(strtrim(parts{1}));
            fa_y(i) = str2double(strtrim(parts{2}));
            fa_z(i) = 0;
        elseif num_values == 1
            % 只有1个数值：可能是单一数值，或者格式异常
            single_val = str2double(strtrim(parts{1}));
            if ~isnan(single_val)
                % 如果只有一个数值，可能是某种特殊情况，全部设为该值或设为NaN
                fa_x(i) = single_val;
                fa_y(i) = 0;
                fa_z(i) = 0;
            else
                fa_x(i) = NaN;
                fa_y(i) = NaN;
                fa_z(i) = NaN;
            end
        else
            % 无法解析
            fa_x(i) = NaN;
            fa_y(i) = NaN;
            fa_z(i) = NaN;
        end
        
        % 检查解析是否成功
        if isnan(fa_x(i)) || isnan(fa_y(i)) || isnan(fa_z(i))
            % 不再显示警告，静默处理（因为可能有很多这样的行）
        end
    end
    
    % 移除NaN值所在的行（如果需要绘制连续数据）
    valid_idx = ~isnan(fa_x) & ~isnan(fa_y) & ~isnan(fa_z);
    if sum(valid_idx) < num_rows * 0.9  % 如果有效数据少于90%，显示警告
        warning('有 %.1f%% 的数据行fa格式异常，已跳过', (1 - sum(valid_idx)/num_rows) * 100);
    end
    
    % ==================== 创建包含3个子图的图形 ====================
    figure('Position', [100, 100, 1000, 800], 'Name', sprintf('训练数据气动力 - %s', wind_title));
    
    % 只绘制有效数据（排除NaN值）
    valid_idx = ~isnan(fa_x) & ~isnan(fa_y) & ~isnan(fa_z);
    
    if sum(valid_idx) == 0
        warning('文件 %s 中没有有效的fa数据，跳过绘图', filename);
        continue;
    end
    
    time_valid = time(valid_idx);
    fa_x_valid = fa_x(valid_idx);
    fa_y_valid = fa_y(valid_idx);
    fa_z_valid = fa_z(valid_idx);
    
    % 子图1：X方向（东向）
    subplot(3, 1, 1);
    plot(time_valid, fa_x_valid, '-', 'LineWidth', 2.2, 'Color', [0.12, 0.47, 0.71]);
    xlabel('时间 (s)', 'FontName', 'SimSun', 'FontSize', 18);
    ylabel('东向气动力 (N)', 'FontName', 'SimSun', 'FontSize', 18);
    title(sprintf('X方向（东向）气动力 - %s', wind_title), 'FontName', 'SimSun', 'FontSize', 20);
    grid on;
    box on;
    set(gca, 'FontSize', 16);
    
    % 子图2：Y方向（北向）
    subplot(3, 1, 2);
    plot(time_valid, fa_y_valid, '-', 'LineWidth', 2.2, 'Color', [0.89, 0.47, 0.20]);
    xlabel('时间 (s)', 'FontName', 'SimSun', 'FontSize', 18);
    ylabel('北向气动力 (N)', 'FontName', 'SimSun', 'FontSize', 18);
    title(sprintf('Y方向（北向）气动力 - %s', wind_title), 'FontName', 'SimSun', 'FontSize', 20);
    grid on;
    box on;
    set(gca, 'FontSize', 16);
    
    % 子图3：Z方向（天向）
    subplot(3, 1, 3);
    plot(time_valid, fa_z_valid, '-', 'LineWidth', 2.2, 'Color', [0.20, 0.63, 0.17]);
    xlabel('时间 (s)', 'FontName', 'SimSun', 'FontSize', 18);
    ylabel('天向气动力 (N)', 'FontName', 'SimSun', 'FontSize', 18);
    title(sprintf('Z方向（天向）气动力 - %s', wind_title), 'FontName', 'SimSun', 'FontSize', 20);
    grid on;
    box on;
    set(gca, 'FontSize', 16);
    
    % 保存图片（高分辨率，PNG + EMF）
    base_name = sprintf('training_fa_%s', extractBefore(filename, '.csv'));
    set(gcf, 'PaperPositionMode', 'auto');
    print(gcf, fullfile(output_dir, [base_name, '.png']), '-dpng', '-r600');
    print(gcf, fullfile(output_dir, [base_name, '.emf']), '-dmeta', '-r600');
    
    fprintf('完成文件: %s\n\n', filename);
    close(gcf);
end

fprintf('所有训练数据气动力图绘制完成！\n');

