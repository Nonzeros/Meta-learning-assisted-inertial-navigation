# 调试代码总结

## 重要说明

**⚠️ 关键事实**：Python 启动的 MATLAB 引擎是**独立的后台进程**，**无法直接弹出 MATLAB IDE 窗口**进行交互式调试。

但是，我们添加了以下调试功能来帮助你追踪问题：

## 已添加的调试代码

### 1. MATLAB 函数中的详细调试输出

**文件**：`matlab/utils/test_SINS_dynamic_UKF_153_forpython.m`

#### 1.1 调试标志设置（第14-30行）

```matlab
% ========== 调试标志：是否输出详细的UKF更新信息 ==========
ENABLE_DETAILED_DEBUG = true;  % 设置为true以启用详细调试输出
DEBUG_BASELINE_ONLY = true;     % 如果为true，只对baseline模型输出调试信息

% 检查是否是baseline模型（通过全局变量）
is_baseline = false;
if DEBUG_BASELINE_ONLY
    try
        global IS_BASELINE_MODEL;
        if exist('IS_BASELINE_MODEL', 'var') && IS_BASELINE_MODEL
            is_baseline = true;
        end
    catch
        % 如果全局变量不存在，通过dynamic_pos的值范围判断（备用方法）
        if max(abs(dynamic_pos)) < 1000
            is_baseline = true;
        end
    end
end
```

#### 1.2 UKF更新前的状态输出（第45-95行）

输出内容：
- 观测值（dynamic_pos，ENU米）
- INS预测位置（ins.pos，LLH弧度）
- 新息（innovation = ins.pos - pvt）
- UKF状态估计（kf.xk，前9个元素）
- 协方差矩阵对角元素（kf.Pxk）
- 观测噪声协方差（kf.Rk）

#### 1.3 UKF更新后的状态输出（第100-160行）

输出内容：
- 更新后的状态估计（kf.xk）
- 卡尔曼增益（kf.Kk，位置相关的3x3矩阵）
- 更新后的协方差矩阵对角元素
- INS的当前速度（ins.vn）
- **异常值检测**：如果速度或状态值异常大（>1e6），会显示警告

#### 1.4 kffeedback前后的状态输出（第170-220行）

输出内容：
- 变量维度检查
- 总反馈量（kf.xfb）
- 反馈后的速度、位置、姿态
- **异常值检测**：如果速度异常大（>1e4 米/秒），会显示警告

### 2. Python 代码中的调试标志设置

**文件**：`src/experiment_runner.py`

#### 2.1 启用 MATLAB 调试模式（第406-442行）

```python
ENABLE_MATLAB_DEBUG = True  # 设置为 True 以启用调试

if ENABLE_MATLAB_DEBUG:
    # 加载调试设置脚本
    debug_setup_path = os.path.join(project_root, "matlab", "utils", "debug_setup.m")
    eng_intelligent.eval(f"run('{debug_setup_path}')", nargout=0)
    eng_baseline.eval(f"run('{debug_setup_path}')", nargout=0)
    
    # 直接在 MATLAB 引擎中设置断点
    eng_intelligent.eval("dbstop('in', 'test_SINS_dynamic_UKF_153_forpython', 'at', '69')", nargout=0)
    eng_baseline.eval("dbstop('in', 'test_SINS_dynamic_UKF_153_forpython', 'at', '69')", nargout=0)
```

#### 2.2 设置 baseline 模型标识（第1092-1093行，第1111行）

```python
# 在调用 baseline 模型前设置全局变量
eng_baseline.eval("global IS_BASELINE_MODEL; IS_BASELINE_MODEL = true;", nargout=0)

# ... 调用 MATLAB 函数 ...

# 调用后清除标志
eng_baseline.eval("global IS_BASELINE_MODEL; IS_BASELINE_MODEL = false;", nargout=0)
```

#### 2.3 设置元学习模型标识（第897行）

```python
# 在调用元学习模型前设置全局变量
eng_intelligent.eval("global IS_BASELINE_MODEL; IS_BASELINE_MODEL = false;", nargout=0)
```

### 3. MATLAB 调试设置脚本

**文件**：`matlab/utils/debug_setup.m`

功能：
- 启用自动错误断点：`dbstop if error`
- 在关键位置设置断点（第69行）
- 显示所有警告

## 实际效果

### ✅ 可以实现的功能

1. **详细的调试输出**：
   - 在 Python 控制台看到每一步的详细数据
   - 包括状态值、协方差、卡尔曼增益等
   - 自动检测异常值并显示警告

2. **自动错误断点**：
   - 当 MATLAB 函数出错时，Python 会卡住等待
   - 可以在 Python 控制台看到详细的错误堆栈信息

### ❌ 无法实现的功能

1. **弹出 MATLAB IDE 窗口**：
   - Python 启动的 MATLAB 引擎是后台进程
   - 无法直接弹出 MATLAB IDE 进行交互式调试

2. **在 MATLAB IDE 中设置断点并调试**：
   - MATLAB IDE 中的断点只对 IDE 中的 MATLAB 会话有效
   - Python 启动的 MATLAB 引擎是另一个独立进程

## 如何使用

### 方法 1：查看详细调试输出（推荐）

1. 确保 `ENABLE_DETAILED_DEBUG = true`（在 MATLAB 文件中）
2. 运行 Python 代码
3. 在 Python 控制台查看输出，会看到类似：

```
========== [零动力学模型] UKF更新步骤详情 ==========
步骤1: UKF更新前的状态
  - 观测值 (dynamic_pos, ENU米): [x.xxxxxx, x.xxxxxx, x.xxxxxx]
  - INS预测位置 (ins.pos, LLH弧度): [x.xxxxxxxxxx, x.xxxxxxxxxx, x.xxxxxx]
  - 新息 (innovation = ins.pos - pvt): [x.xxxxxxxxxx, x.xxxxxxxxxx, x.xxxxxx]
  ...
步骤2: UKF更新后的状态
  ...
```

### 方法 2：使用自动错误断点

1. 确保 `ENABLE_MATLAB_DEBUG = True`（在 Python 文件中）
2. 运行 Python 代码
3. 当 MATLAB 函数出错时：
   - Python 会卡住等待
   - 在 Python 控制台会看到详细的错误信息
   - 错误信息包含变量维度、值等调试信息

## 调试输出示例

当运行代码时，你会看到类似这样的输出：

```
========== [零动力学模型] UKF更新步骤详情 ==========
步骤1: UKF更新前的状态
  - 观测值 (dynamic_pos, ENU米): [0.123456, -0.234567, 0.345678]
  - INS预测位置 (ins.pos, LLH弧度): [0.595865, -2.062345, 2.047000]
  - 新息 (innovation = ins.pos - pvt): [0.000123, -0.000234, 0.000345]
  - kf.xk (状态估计, 前9个): [1.234567e-06 -2.345678e-06 3.456789e-06 ...]
  - kf.Pxk 对角元素 (前9个): [3.384638e-17 3.384638e-17 ...]
  - kf.Rk 对角元素: [2.458172e-14, 2.458172e-14, 1.000000e+00]

步骤2: UKF更新后的状态
  - kf.xk (更新后, 前9个): [1.234567e-06 -2.345678e-06 3.456789e-06 ...]
  - kf.Kk (位置相关, 3x3):
    [1.234567e-10, 2.345678e-10, 3.456789e-10]
    [4.567890e-10, 5.678901e-10, 6.789012e-10]
    [7.890123e-10, 8.901234e-10, 9.012345e-10]
  - ins.vn (速度, 米/秒): [0.123456, -0.234567, 0.345678]
  ⚠️  警告：检测到异常大的状态值！
  - 最大绝对值: 1.234567e+06

步骤3: 准备调用 kffeedback
  - kf.n = 15
  - kf.xk 大小: [15, 1]
  - kf.xfb 大小: [15, 1]

步骤4: kffeedback后的状态
  - kf.xfb (总反馈量, 前9个): [1.234567e-06 -2.345678e-06 ...]
  - ins.vn (反馈后速度, 米/秒): [0.123456, -0.234567, 0.345678]
  ⚠️  警告：反馈后速度异常大！可能导致数值爆炸！
==========================================
```

## 总结

**已添加的代码**：
1. ✅ MATLAB 函数中的详细调试输出（4个步骤）
2. ✅ Python 代码中的调试标志设置
3. ✅ MATLAB 调试设置脚本
4. ✅ 异常值自动检测和警告

**实际效果**：
- ✅ 可以在 Python 控制台看到详细的调试信息
- ✅ 可以自动检测异常值
- ❌ **无法弹出 MATLAB IDE 窗口**（因为 MATLAB 引擎是后台进程）

**推荐使用方法**：
- 使用详细调试输出（方法1）来追踪每一步的数据变化
- 使用自动错误断点（方法2）来捕获错误信息



