% 气动力数据直方图绘制脚本
% 功能：遍历data/train目录下的所有Excel和CSV文件，读取第13列fa数据并绘制直方图
% 作者：Auto-generated
% 日期：2025-12-20

clear; close all; clc;

% 风速映射表：从文件名标记到实际风速值
wind_speed_map = containers.Map({
    'nowind', '10wind', '20wind', '30wind', '35wind', ...
    '40wind', '50wind', '70wind', '70p20sint', '100wind'
}, {
    '0', '1.3', '2.5', '3.7', '4.2', ...
    '4.9', '6.1', '8.5', '8.5+2.4sin(t)', '12.1'
});

% 设置字体（中文用宋体，英文和数字用Times New Roman）
set(0,'DefaultAxesFontName','Times New Roman');
set(0,'DefaultTextFontName','Times New Roman');
% 注意：MATLAB中设置中文字体需要在绘图时单独指定

% 文件路径配置（相对于脚本所在目录）
script_dir = fileparts(mfilename('fullpath'));
project_root = fullfile(script_dir, '..', '..', '..');
output_dir = fullfile(script_dir, 'plot_output');

% 确保输出目录存在
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% 检查训练数据目录是否存在（尝试多个可能的目录名）
train_data_dir = fullfile(project_root, 'data', 'train');
if ~exist(train_data_dir, 'dir')
    % 如果train目录不存在，尝试training目录
    train_data_dir = fullfile(project_root, 'data', 'training');
    if ~exist(train_data_dir, 'dir')
        error('训练数据目录不存在。请检查以下目录之一是否存在:\n  - %s\n  - %s', ...
            fullfile(project_root, 'data', 'train'), ...
            fullfile(project_root, 'data', 'training'));
    else
        fprintf('使用目录: %s\n', train_data_dir);
    end
else
    fprintf('使用目录: %s\n', train_data_dir);
end

% 获取所有数据文件（Excel和CSV）
excel_files = dir(fullfile(train_data_dir, '*.xlsx'));
excel_files = [excel_files; dir(fullfile(train_data_dir, '*.xls'))];
csv_files = dir(fullfile(train_data_dir, '*.csv'));
data_files = [excel_files; csv_files];

if isempty(data_files)
    error('在目录 %s 中未找到数据文件（.xlsx, .xls 或 .csv）', train_data_dir);
end

fprintf('找到 %d 个数据文件（%d 个Excel文件，%d 个CSV文件）\n', ...
    length(data_files), length(excel_files), length(csv_files));

% 初始化存储所有文件数据的结构
file_data = [];

% 遍历每个数据文件，收集所有数据
valid_file_count = 0;
for file_idx = 1:length(data_files)
    filename = data_files(file_idx).name;
    filepath = fullfile(train_data_dir, filename);
    
    [~, ~, ext] = fileparts(filename);
    file_type = upper(ext(2:end));  % 去掉点号
    
    fprintf('处理文件 %d/%d: %s (%s)\n', file_idx, length(data_files), filename, file_type);
    
    try
        % 初始化当前文件的fa数组
        fax = [];
        fay = [];
        faz = [];
        
        % 根据文件类型选择读取方式
        if strcmpi(ext, '.csv')
            % CSV文件：使用健壮的逐行解析方法
            fprintf('  使用健壮方法读取CSV文件\n');
            fa_column_str = read_malformed_csv(filepath, 13);
            num_rows = length(fa_column_str);
            
            % 解析每行的fa数据
            for i = 1:num_rows
                fa_str = char(fa_column_str(i));
                
                % 检查是否为空或NaN
                if isempty(fa_str) || strcmpi(fa_str, 'nan') || strcmpi(fa_str, 'none')
                    continue;
                end
                
                % 移除双引号（可能出现在字符串的开头和结尾）
                fa_str = strrep(fa_str, '"', '');
                fa_str = strtrim(fa_str);
                
                % 移除方括号
                fa_str = strrep(fa_str, '[', '');
                fa_str = strrep(fa_str, ']', '');
                fa_str = strtrim(fa_str);
                
                % 如果字符串为空，跳过
                if isempty(fa_str)
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
                
                % 移除空字符串并去除每个部分的前后空格
                parts = parts(~cellfun(@isempty, parts));
                parts = cellfun(@strtrim, parts, 'UniformOutput', false);
                parts = parts(~cellfun(@isempty, parts));
                
                % 提取数值（处理1个、2个或3个数值的情况）
                num_values = length(parts);
                if num_values >= 3
                    % 标准格式：3个数值
                    fax_val = str2double(parts{1});
                    fay_val = str2double(parts{2});
                    faz_val = str2double(parts{3});
                    
                    % 检查解析是否成功
                    if ~isnan(fax_val) && ~isnan(fay_val) && ~isnan(faz_val)
                        fax = [fax; fax_val];
                        fay = [fay; fay_val];
                        faz = [faz; faz_val];
                    end
                elseif num_values == 2
                    % 只有2个数值：假设第3个为0
                    fax_val = str2double(parts{1});
                    fay_val = str2double(parts{2});
                    
                    if ~isnan(fax_val) && ~isnan(fay_val)
                        fax = [fax; fax_val];
                        fay = [fay; fay_val];
                        faz = [faz; 0];
                    end
                elseif num_values == 1
                    % 只有1个数值：可能是单一数值，或者格式异常
                    single_val = str2double(parts{1});
                    if ~isnan(single_val)
                        fax = [fax; single_val];
                        fay = [fay; 0];
                        faz = [faz; 0];
                    end
                end
            end
        else
            % Excel文件：使用readtable
            fprintf('  使用readtable读取Excel文件\n');
            data = readtable(filepath);
            
            % 获取列数
            num_cols = width(data);
            
            if num_cols < 13
                warning('文件 %s 的列数不足13列（实际%d列），跳过', filename, num_cols);
                continue;
            end
            
            % 获取行数
            num_rows = height(data);
            
            % 读取第13列（fa数据）
            fa_column = data{:, 13};
            
            % 解析fa列
            for i = 1:num_rows
                % 处理fa列（可能是cell数组中的字符串）
                if iscell(fa_column)
                    fa_str = char(fa_column{i});
                elseif isstring(fa_column)
                    fa_str = char(fa_column(i));
                else
                    fa_str = char(string(fa_column(i)));
                end
                
                % 检查是否为空或NaN
                if isempty(fa_str) || strcmpi(fa_str, 'nan') || strcmpi(fa_str, 'none')
                    continue;
                end
                
                % 移除方括号和空格
                fa_str = strrep(fa_str, '[', '');
                fa_str = strrep(fa_str, ']', '');
                fa_str = strtrim(fa_str);
                
                % 如果字符串为空，跳过
                if isempty(fa_str)
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
                    fax_val = str2double(strtrim(parts{1}));
                    fay_val = str2double(strtrim(parts{2}));
                    faz_val = str2double(strtrim(parts{3}));
                    
                    % 检查解析是否成功
                    if ~isnan(fax_val) && ~isnan(fay_val) && ~isnan(faz_val)
                        fax = [fax; fax_val];
                        fay = [fay; fay_val];
                        faz = [faz; faz_val];
                    end
                elseif num_values == 2
                    % 只有2个数值：假设第3个为0
                    fax_val = str2double(strtrim(parts{1}));
                    fay_val = str2double(strtrim(parts{2}));
                    
                    if ~isnan(fax_val) && ~isnan(fay_val)
                        fax = [fax; fax_val];
                        fay = [fay; fay_val];
                        faz = [faz; 0];
                    end
                elseif num_values == 1
                    % 只有1个数值：可能是单一数值，或者格式异常
                    single_val = str2double(strtrim(parts{1}));
                    if ~isnan(single_val)
                        fax = [fax; single_val];
                        fay = [fay; 0];
                        faz = [faz; 0];
                    end
                end
            end
        end
        
        % 检查是否有有效数据
        if isempty(fax)
            warning('文件 %s 中没有有效的fa数据，跳过绘图', filename);
            continue;
        end
        
        fprintf('  成功提取 %d 个有效数据点\n', length(fax));
        fprintf('  fax范围: [%.4f, %.4f]\n', min(fax), max(fax));
        fprintf('  fay范围: [%.4f, %.4f]\n', min(fay), max(fay));
        fprintf('  faz范围: [%.4f, %.4f]\n', min(faz), max(faz));
        
        % 存储当前文件的数据
        [~, file_basename, ~] = fileparts(filename);
        
        % 从文件名提取风速标签
        wind_label = file_basename;  % 默认使用文件名
        wind_patterns = keys(wind_speed_map);
        for i = 1:length(wind_patterns)
            pattern = wind_patterns{i};
            if contains(file_basename, pattern)
                wind_value = wind_speed_map(pattern);
                wind_label = sprintf('风速 %s m/s', wind_value);
                break;
            end
        end
        
        valid_file_count = valid_file_count + 1;
        file_data(valid_file_count).filename = file_basename;
        file_data(valid_file_count).wind_label = wind_label;
        file_data(valid_file_count).fax = fax;
        file_data(valid_file_count).fay = fay;
        file_data(valid_file_count).faz = faz;
        
    catch ME
        warning('处理文件 %s 时出错: %s', filename, ME.message);
        continue;
    end
end

if isempty(file_data) || valid_file_count == 0
    error('未能从任何文件中提取到有效的fa数据');
end

fprintf('\n共成功处理 %d 个文件，开始绘制统一直方图...\n', length(file_data));

% ==================== 绘制统一的直方图 ====================
figure('Position', [100, 100, 1400, 900], 'Name', '气动力数据直方图 - 所有文件');

% 定义颜色方案（为每个文件分配不同颜色）
colors = lines(length(file_data));  % 使用lines颜色映射
% 或者使用自定义颜色
% colors = [
%     0.12, 0.47, 0.71;  % 蓝色
%     0.89, 0.47, 0.20;  % 橙色
%     0.20, 0.63, 0.17;  % 绿色
%     0.85, 0.33, 0.10;  % 红橙色
%     0.60, 0.20, 0.80;  % 紫色
%     0.20, 0.80, 0.80;  % 青色
% ];

% 子图1：fax直方图（所有文件）
subplot(3, 1, 1);
hold on;
legend_handles = [];
legend_labels = {};
for i = 1:length(file_data)
    h = histogram(file_data(i).fax, 50, 'FaceColor', colors(i,:), ...
        'EdgeColor', 'none', 'FaceAlpha', 0.6, 'DisplayName', file_data(i).wind_label, ...
        'Normalization', 'pdf');
    legend_handles = [legend_handles, h];
    legend_labels{end+1} = file_data(i).wind_label;
end
xlabel('东向剩余气动力/N', 'FontName', 'SimSun', 'FontSize', 18);
ylabel('概率密度(N^-^1)', 'FontName', 'SimSun', 'FontSize', 18);
title('各方向剩余气动力分布', 'FontName', 'SimSun', 'FontSize', 20);
legend(legend_handles, legend_labels, 'Location', 'best', 'FontName', 'SimSun', 'FontSize', 12);
grid on;
box on;
set(gca, 'FontSize', 16);
hold off;

% 子图2：fay直方图（所有文件）
subplot(3, 1, 2);
hold on;
legend_handles = [];
legend_labels = {};
for i = 1:length(file_data)
    h = histogram(file_data(i).fay, 50, 'FaceColor', colors(i,:), ...
        'EdgeColor', 'none', 'FaceAlpha', 0.6, 'DisplayName', file_data(i).wind_label, ...
        'Normalization', 'pdf');
    legend_handles = [legend_handles, h];
    legend_labels{end+1} = file_data(i).wind_label;
end
xlabel('北向剩余气动力/N', 'FontName', 'SimSun', 'FontSize', 18);
ylabel('概率密度(N^-^1)', 'FontName', 'SimSun', 'FontSize', 18);
% title('北向剩余气动力分布 - 所有文件', 'FontName', 'SimSun', 'FontSize', 20);
legend(legend_handles, legend_labels, 'Location', 'best', 'FontName', 'SimSun', 'FontSize', 12);
grid on;
box on;
set(gca, 'FontSize', 16);
hold off;

% 子图3：faz直方图（所有文件）
subplot(3, 1, 3);
hold on;
legend_handles = [];
legend_labels = {};
for i = 1:length(file_data)
    h = histogram(file_data(i).faz, 50, 'FaceColor', colors(i,:), ...
        'EdgeColor', 'none', 'FaceAlpha', 0.6, 'DisplayName', file_data(i).wind_label, ...
        'Normalization', 'pdf');
    legend_handles = [legend_handles, h];
    legend_labels{end+1} = file_data(i).wind_label;
end
xlabel('天向剩余气动力/N', 'FontName', 'SimSun', 'FontSize', 18);
ylabel('概率密度(N^-^1)', 'FontName', 'SimSun', 'FontSize', 18);
% title('天向剩余气动力分布', 'FontName', 'SimSun', 'FontSize', 20);
legend(legend_handles, legend_labels, 'Location', 'best', 'FontName', 'SimSun', 'FontSize', 12);
grid on;
box on;
set(gca, 'FontSize', 16);
hold off;

% 保存图片（高分辨率，PNG + EMF）
set(gcf, 'PaperPositionMode', 'auto');
print(gcf, fullfile(output_dir, 'fa_histogram_all_files.png'), '-dpng', '-r600');
print(gcf, fullfile(output_dir, 'fa_histogram_all_files.emf'), '-dmeta', '-r600');

fprintf('\n统一直方图已保存到: %s\n', output_dir);
fprintf('气动力数据直方图绘制完成！\n');

