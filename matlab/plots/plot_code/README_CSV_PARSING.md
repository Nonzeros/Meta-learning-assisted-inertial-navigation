# CSV解析问题及解决方案

## 问题描述

当CSV文件中某一列包含向量数据（如 `[-5.5817, 0.7544, 1.3637]`）且**未用引号包裹**时，`readtable()` 函数会错误地将向量中的逗号识别为列分隔符，导致：

1. **列数错误**：原本应该是13列，但被识别为15列或更多
2. **数据损坏**：向量被分割成多列，无法正确提取
3. **列索引失效**：第13列的实际内容可能被分散到第13、14、15列

### 为什么 readtable() 会失败？

`readtable()` 使用标准的CSV解析规则：
- 逗号（`,`）被视为列分隔符
- 只有在引号内的逗号才会被忽略
- 由于向量列未加引号，向量中的逗号被误认为是列分隔符

例如，原始CSV行：
```
1, 2, 3, ..., 12, [-5.5817, 0.7544, 1.3637], 14
```

`readtable()` 会将其解析为：
- 列1-12: 正常
- 列13: `[-5.5817`
- 列14: `0.7544`
- 列15: `1.3637]`
- 列16: `14`

## 解决方案

### 核心思路

**逐行读取 + 括号匹配**：
1. 使用 `fgetl()` 逐行读取文件
2. 逐字符解析，跟踪方括号的嵌套深度
3. 只将**不在方括号内**的逗号视为列分隔符
4. 准确提取目标列的内容

### 实现细节

`read_malformed_csv.m` 函数的工作原理：

```matlab
% 伪代码
for each line in file:
    bracket_depth = 0
    current_col = 1
    for each character in line:
        if character == '[':
            bracket_depth++
        elseif character == ']':
            bracket_depth--
        elseif character == ',' and bracket_depth == 0:
            % 这是真正的列分隔符
            if current_col == target_col:
                extract content
            current_col++
```

### 关键特性

1. **括号深度跟踪**：正确处理嵌套括号（虽然本例中不需要）
2. **列计数准确**：只计算列分隔符，忽略向量内的逗号
3. **不依赖固定列数**：即使向量被分割成任意多列，也能正确提取
4. **不使用eval**：安全可靠，避免代码注入风险

## 使用方法

```matlab
% 读取第13列的fa数据
fa_column = read_malformed_csv('data.csv', 13);

% 解析向量字符串
for i = 1:length(fa_column)
    fa_str = char(fa_column(i));
    % 移除方括号
    fa_str = strrep(fa_str, '[', '');
    fa_str = strrep(fa_str, ']', '');
    % 分割并转换为数值
    parts = strsplit(fa_str, ',');
    fax = str2double(parts{1});
    fay = str2double(parts{2});
    faz = str2double(parts{3});
end
```

## 优势

- ✅ **健壮性**：不依赖CSV的列数
- ✅ **准确性**：正确识别列边界
- ✅ **安全性**：不使用eval
- ✅ **通用性**：适用于任何包含未加引号向量的CSV文件


