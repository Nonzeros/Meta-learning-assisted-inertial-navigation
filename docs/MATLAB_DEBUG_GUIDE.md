# MATLAB 调试指南

## 概述

当 Python 调用 MATLAB 函数时，你可以在 MATLAB IDE 中设置断点并进行调试。这是调试 MATLAB 代码最强大的方式。

## 快速开始

### 方法 1：自动启用调试模式（推荐）

1. **在 Python 代码中启用调试**：
   
   打开 `src/experiment_runner.py`，找到以下代码：
   ```python
   ENABLE_MATLAB_DEBUG = False  # 默认关闭，需要调试时改为 True
   ```
   
   将其改为：
   ```python
   ENABLE_MATLAB_DEBUG = True  # 启用调试模式
   ```

2. **运行 Python 代码**：
   ```bash
   python src/main.py
   ```

3. **当 MATLAB 函数出错时**：
   - MATLAB 会自动停在错误处
   - Python 会等待 MATLAB 调试完成
   - 你可以在 MATLAB 命令窗口中检查变量、单步执行等

### 方法 2：手动设置断点（更灵活）

1. **打开 MATLAB IDE**，找到要调试的文件：
   - `matlab/utils/test_SINS_dynamic_UKF_153_forpython.m`

2. **在关键位置设置断点**：
   - 点击行号左侧，设置红色断点
   - 推荐断点位置：
     - `kffeedback` 调用前（第57行）
     - `ukf` 调用后（第29行）

3. **在 MATLAB 命令行启用自动断点**：
   ```matlab
   dbstop if error      % 自动在错误处断点
   dbstop if warning   % 自动在警告处断点（可选）
   ```

4. **运行 Python 代码**：
   ```bash
   python src/main.py
   ```

5. **当执行到断点时**：
   - MATLAB 会停在断点处
   - Python 会等待 MATLAB 返回
   - 你可以在 MATLAB 中调试

## 常用调试命令

在 MATLAB 命令窗口中（显示 `K>>` 提示符时），可以使用以下命令：

### 基本调试命令

```matlab
dbstep          % 单步执行（进入函数）
dbstep in        % 单步执行（进入函数）
dbstep out       % 单步执行（跳出函数）
dbcont           % 继续执行到下一个断点
dbquit           % 退出调试模式
dbstack          % 显示调用堆栈
```

### 变量检查命令

```matlab
whos             % 查看所有变量及其大小
size(kf.xk)      % 检查 kf.xk 的维度
size(kf.xfb)     % 检查 kf.xfb 的维度
size(kf.coef_fb) % 检查 kf.coef_fb 的维度
kf.n             % 检查状态数
kf.xk            % 查看 kf.xk 的值
kf.xfb           % 查看 kf.xfb 的值
```

### 条件断点

```matlab
% 在特定条件下断点
dbstop in test_SINS_dynamic_UKF_153_forpython at 30 if size(kf.xk,2)>1
```

## 调试技巧

### 1. 检查维度不匹配问题

当遇到 "无法执行赋值，因为左侧和右侧的元素数目不同" 错误时：

```matlab
% 在 kffeedback 调用前检查
K>> size(kf.xk)        % 应该是 [15, 1]，如果是 [15, 3] 则有问题
K>> size(kf.xfb)       % 应该是 [15, 1]
K>> size(kf.coef_fb)   % 应该是 [15, 1]
K>> kf.n               % 应该是 15
```

### 2. 检查对象状态

```matlab
% 检查 kf 对象的所有字段
K>> fieldnames(kf)

% 检查特定字段是否存在
K>> isfield(kf, 'xk')
K>> isfield(kf, 'xfb')
K>> isfield(kf, 'n')
```

### 3. 手动修复问题

如果发现 `kf.xk` 是矩阵而不是列向量：

```matlab
K>> kf.xk = kf.xk(:, 1);  % 取第一列
K>> size(kf.xk)           % 确认现在是 [15, 1]
K>> dbcont                 % 继续执行
```

### 4. 查看调用堆栈

```matlab
K>> dbstack
% 显示：
%   In test_SINS_dynamic_UKF_153_forpython.m at 30
%   In kffeedback.m at 70
```

## 常见问题

### Q: Python 卡住不动了？

**A**: 这是正常的！说明 MATLAB 停在断点处了。打开 MATLAB IDE，在命令窗口中调试。

### Q: 如何退出调试模式？

**A**: 在 MATLAB 命令窗口中输入：
```matlab
dbquit
```

### Q: 如何跳过当前断点？

**A**: 使用 `dbcont` 继续执行，或使用 `dbstep` 单步执行。

### Q: 如何只调试特定模型？

**A**: 可以在 Python 代码中只为一个引擎启用调试，或者只在特定的 MATLAB 文件中设置断点。

## 调试脚本

我们提供了一个调试设置脚本 `matlab/utils/debug_setup.m`，它会自动启用常用的调试选项。

在 MATLAB 命令行中运行：
```matlab
debug_setup
```

或者在 Python 中运行：
```python
eng.eval("debug_setup", nargout=0)
```

## 示例：调试 kffeedback 维度错误

1. **在 MATLAB 中设置断点**：
   - 打开 `test_SINS_dynamic_UKF_153_forpython.m`
   - 在第57行（`kffeedback` 调用前）设置断点

2. **启用自动错误断点**：
   ```matlab
   dbstop if error
   ```

3. **运行 Python 代码**

4. **当停在断点时，检查变量**：
   ```matlab
   K>> size(kf.xk)        % 检查维度
   K>> size(kf.xfb)       % 检查维度
   K>> kf.n               % 检查状态数
   ```

5. **如果发现问题，手动修复**：
   ```matlab
   K>> if size(kf.xk, 2) > 1
         kf.xk = kf.xk(:, 1);
      end
   K>> dbcont             % 继续执行
   ```

## 注意事项

1. **调试时 Python 会等待**：当 MATLAB 停在断点时，Python 会一直等待，直到你退出调试模式。

2. **多个 MATLAB 引擎**：如果使用了多个 MATLAB 引擎实例，每个引擎都是独立的，需要在每个引擎中分别设置断点。

3. **性能影响**：启用调试模式会显著降低执行速度，调试完成后记得关闭。

4. **断点持久性**：MATLAB 的断点在 MATLAB 会话关闭后不会保存，需要重新设置。

## 更多资源

- [MATLAB 调试文档](https://www.mathworks.com/help/matlab/debugging-code.html)
- [MATLAB Engine for Python 文档](https://www.mathworks.com/help/matlab/matlab_external/get-started-with-matlab-engine-for-python.html)



