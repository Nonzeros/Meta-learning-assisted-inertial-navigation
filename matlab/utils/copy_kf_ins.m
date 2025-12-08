function [kf_copy, ins_copy] = copy_kf_ins(kf, ins)
% 深拷贝UKF和INS对象，确保每个模型使用独立的实例
% 
% 输入:
%   kf - 原始UKF对象
%   ins - 原始INS对象
% 输出:
%   kf_copy - 拷贝的UKF对象
%   ins_copy - 拷贝的INS对象
%
% 注意：PSINS的kf和ins可能是handle类，需要手动拷贝所有属性

% 创建新的kf结构体/对象
kf_copy = struct();

% 获取kf的所有字段并拷贝
if isstruct(kf)
    % 如果是结构体，直接拷贝所有字段
    fields = fieldnames(kf);
    for i = 1:length(fields)
        field_name = fields{i};
        field_value = kf.(field_name);
        
        % 如果是矩阵或数组，进行深拷贝（使用copy确保独立）
        if isnumeric(field_value) || islogical(field_value)
            kf_copy.(field_name) = field_value + 0;  % 强制深拷贝数值数组（+0创建新数组）
        elseif isstruct(field_value)
            % 递归拷贝嵌套结构体
            kf_copy.(field_name) = copy_struct(field_value);
        elseif isa(field_value, 'function_handle')
            % 函数句柄直接拷贝（引用）
            kf_copy.(field_name) = field_value;
        else
            % 其他类型直接赋值
            kf_copy.(field_name) = field_value;
        end
    end
else
    % 如果是类对象，尝试使用copy方法（如果存在）
    try
        kf_copy = copy(kf);
    catch
        % 如果没有copy方法，手动拷贝属性
        props = properties(kf);
        for i = 1:length(props)
            prop_name = props{i};
            try
                prop_value = kf.(prop_name);
                if isnumeric(prop_value) || islogical(prop_value)
                    kf_copy.(prop_name) = prop_value;
                elseif isstruct(prop_value)
                    kf_copy.(prop_name) = copy_struct(prop_value);
                else
                    kf_copy.(prop_name) = prop_value;
                end
            catch
                % 如果属性不可访问，跳过
            end
        end
    end
end

% 创建新的ins结构体/对象
ins_copy = struct();

% 获取ins的所有字段并拷贝
if isstruct(ins)
    % 如果是结构体，直接拷贝所有字段
    fields = fieldnames(ins);
    for i = 1:length(fields)
        field_name = fields{i};
        field_value = ins.(field_name);
        
        % 如果是矩阵或数组，进行深拷贝（使用copy确保独立）
        if isnumeric(field_value) || islogical(field_value)
            ins_copy.(field_name) = field_value + 0;  % 强制深拷贝数值数组（+0创建新数组）
        elseif isstruct(field_value)
            % 递归拷贝嵌套结构体
            ins_copy.(field_name) = copy_struct(field_value);
        elseif isa(field_value, 'function_handle')
            % 函数句柄直接拷贝（引用）
            ins_copy.(field_name) = field_value;
        else
            % 其他类型直接赋值
            ins_copy.(field_name) = field_value;
        end
    end
else
    % 如果是类对象，尝试使用copy方法（如果存在）
    try
        ins_copy = copy(ins);
    catch
        % 如果没有copy方法，手动拷贝属性
        props = properties(ins);
        for i = 1:length(props)
            prop_name = props{i};
            try
                prop_value = ins.(prop_name);
                if isnumeric(prop_value) || islogical(prop_value)
                    ins_copy.(prop_name) = prop_value;
                elseif isstruct(prop_value)
                    ins_copy.(prop_name) = copy_struct(prop_value);
                else
                    ins_copy.(prop_name) = prop_value;
                end
            catch
                % 如果属性不可访问，跳过
            end
        end
    end
end

end

% 辅助函数：递归拷贝结构体
function s_copy = copy_struct(s)
    s_copy = struct();
    fields = fieldnames(s);
    for i = 1:length(fields)
        field_name = fields{i};
        field_value = s.(field_name);
        
        if isnumeric(field_value) || islogical(field_value)
            s_copy.(field_name) = field_value + 0;  % 强制深拷贝数值数组
        elseif isstruct(field_value)
            s_copy.(field_name) = copy_struct(field_value);
        elseif isa(field_value, 'function_handle')
            s_copy.(field_name) = field_value;
        else
            s_copy.(field_name) = field_value;
        end
    end
end

