function [fa_column] = read_malformed_csv(filepath, target_col)
%READ_MALFORMED_CSV 读取格式错误的CSV文件，其中向量列包含逗号但未加引号
%
% 问题描述：
% 当CSV文件中某一列包含向量（如 [-5.5817, 0.7544, 1.3637]）且未用引号包裹时，
% readtable()会错误地将向量中的逗号识别为列分隔符，导致向量被分割成多列。
%
% 解决方案：
% 逐行读取文件，手动解析每一行。通过计算目标列之前的逗号数量，并处理
% 向量中的逗号，来准确定位和提取目标列的内容。
%
% 输入参数：
%   filepath  - CSV文件路径
%   target_col - 目标列索引（从1开始，不包括可能的索引列）
%
% 输出参数：
%   fa_column - 字符串数组，包含每行的目标列内容
%
% 示例：
%   fa_column = read_malformed_csv('data.csv', 13);
%   % 返回每行第13列的完整向量字符串

    % 打开文件
    fid = fopen(filepath, 'r');
    if fid == -1
        error('无法打开文件: %s', filepath);
    end
    
    % 读取表头（第一行），用于调试
    header_line = fgetl(fid);
    if header_line == -1
        fclose(fid);
        error('文件为空或无法读取表头');
    end
    
    % 检查表头，确定是否有索引列
    % 如果第一列是 "Unnamed: 0" 或类似，则target_col需要调整
    header_parts = strsplit(header_line, ',');
    has_index_col = false;
    if length(header_parts) > 0
        first_col_name = strtrim(header_parts{1});
        if contains(first_col_name, 'Unnamed') || isempty(first_col_name) || ...
           (length(header_parts) > target_col && strcmpi(strtrim(header_parts{target_col+1}), 'fa'))
            % 有索引列，target_col保持不变（因为索引列是第0列）
            has_index_col = true;
        end
    end
    
    % 初始化输出
    fa_column = {};
    line_num = 0;
    
    % 逐行读取数据
    while ~feof(fid)
        line = fgetl(fid);
        if line == -1
            break;
        end
        
        line_num = line_num + 1;
        
        % 跳过空行
        if isempty(strtrim(line))
            fa_column{line_num} = '';
            continue;
        end
        
        % 策略：逐字符解析，跟踪括号嵌套和列位置
        % 1. 遍历每个字符，跟踪方括号深度
        % 2. 只将不在方括号内的逗号视为列分隔符
        % 3. 找到目标列后，如果向量被分割，继续合并后续列
        
        bracket_depth = 0;
        current_col = 1;
        col_start = 1;
        target_col_start = -1;
        target_content = '';
        
        i = 1;
        while i <= length(line)
            ch = line(i);
            
            % 跟踪方括号深度
            if ch == '['
                bracket_depth = bracket_depth + 1;
            elseif ch == ']'
                bracket_depth = bracket_depth - 1;
            end
            
            % 处理列分隔符（只在不在方括号内时）
            if ch == ',' && bracket_depth == 0
                if current_col == target_col
                    % 找到目标列
                    target_col_start = col_start;
                    target_content = line(col_start:i-1);
                    
                    % 检查向量是否被分割（以'['开始但未以']'结束）
                    trimmed = strtrim(target_content);
                    if startsWith(trimmed, '[') && ~endsWith(trimmed, ']')
                        % 向量被分割，需要继续合并后续列
                        col_start = i + 1;
                        current_col = current_col + 1;
                        % 继续循环以合并后续列
                    else
                        % 完整的列，退出
                        break;
                    end
                elseif current_col < target_col
                    % 还没到目标列
                    col_start = i + 1;
                    current_col = current_col + 1;
                elseif current_col > target_col && target_col_start > 0
                    % 正在合并被分割的向量
                    % 检查是否找到完整的向量
                    temp_content = line(target_col_start:i-1);
                    trimmed_temp = strtrim(temp_content);
                    if endsWith(trimmed_temp, ']')
                        target_content = temp_content;
                        break;
                    end
                    % 继续合并
                    col_start = i + 1;
                    current_col = current_col + 1;
                end
            end
            
            i = i + 1;
        end
        
        % 处理边界情况
        if isempty(target_content)
            if current_col == target_col
                % 目标列是最后一列
                target_content = line(col_start:end);
            elseif target_col_start > 0
                % 正在合并被分割的向量，但到达行尾
                target_content = line(target_col_start:end);
            else
                % 列数不足
                target_content = '';
            end
        end
        
        % 存储结果
        fa_column{line_num} = strtrim(target_content);
    end
    
    % 关闭文件
    fclose(fid);
    
    % 转换为字符串数组
    fa_column = string(fa_column);
    
    % 显示提取到的非空内容数量
    non_empty_count = sum(~cellfun(@isempty, cellstr(fa_column)));
    fprintf('  成功提取 %d/%d 行数据\n', non_empty_count, length(fa_column));
end

