# MATLAB 窗口和调试控制指南

## 哪些代码会影响 MATLAB 窗口和调试行为

### 1. **启动 MATLAB 引擎时是否显示窗口**

**位置**：`src/main.py` 第 131、133 行

```python
# 方式1：默认启动（可能会打开 MATLAB 窗口）
eng_intelligent = matlab.engine.start_matlab()
eng_baseline = matlab.engine.start_matlab()

# 方式2：无窗口模式启动（不显示 MATLAB 窗口）
eng_intelligent = matlab.engine.start_matlab("-nodesktop")
eng_baseline = matlab.engine.start_matlab("-nodesktop")
```

**影响**：
- 使用默认方式：MATLAB 引擎启动时可能会打开 MATLAB 窗口（取决于 MATLAB 配置）
- 使用 `-nodesktop`：不会打开 MATLAB 窗口，完全后台运行

**建议**：
- 需要交互式调试时：使用默认方式（不添加 `-nodesktop`）
- 不需要窗口时：使用 `-nodesktop` 参数

---

### 2. **启用自动错误断点**

**位置**：`src/experiment_runner.py` 第 404 行和 `matlab/utils/debug_setup.m` 第 16 行

```python
# Python 代码中
ENABLE_MATLAB_DEBUG = True  # 启用调试模式
```

```matlab
% MATLAB 代码中（debug_setup.m）
dbstop if error  % 自动在错误处断点
```

**影响**：
- 当 MATLAB 函数出错时，会自动停在错误处
- 如果 MATLAB 窗口是打开的，你可以在 MATLAB 命令窗口中输入调试命令（如 `dbstep`, `dbcont`, `dbquit` 等）
- Python 会一直等待，直到你退出调试模式

**如何控制**：
- 启用：设置 `ENABLE_MATLAB_DEBUG = True`
- 关闭：设置 `ENABLE_MATLAB_DEBUG = False`

---

### 3. **设置代码断点**

**位置**：`src/experiment_runner.py` 第 427-428 行

```python
eng_intelligent.eval("dbstop('in', 'test_SINS_dynamic_UKF_153_forpython', 'at', '69')", nargout=0)
eng_baseline.eval("dbstop('in', 'test_SINS_dynamic_UKF_153_forpython', 'at', '69')", nargout=0)
```

**影响**：
- 当执行到指定行（第69行）时，会停在断点处
- 如果 MATLAB 窗口是打开的，你可以在 MATLAB 命令窗口中交互式调试
- Python 会一直等待，直到你退出调试模式

**如何控制**：
- 这段代码只在 `ENABLE_MATLAB_DEBUG = True` 时执行
- 要启用：设置 `ENABLE_MATLAB_DEBUG = True`
- 要关闭：设置 `ENABLE_MATLAB_DEBUG = False`

---

## 完整的控制方案

### 场景 1：需要交互式调试（弹出 MATLAB 窗口，可以调试）

**设置**：
1. `src/main.py`：使用默认启动（不添加 `-nodesktop`）
   ```python
   eng_intelligent = matlab.engine.start_matlab()
   eng_baseline = matlab.engine.start_matlab()
   ```

2. `src/experiment_runner.py`：启用调试模式
   ```python
   ENABLE_MATLAB_DEBUG = True
   ```

**效果**：
- MATLAB 窗口会打开
- 当执行到断点或出错时，会停在断点处
- 你可以在 MATLAB 命令窗口中输入调试命令（`dbstep`, `dbcont`, `dbquit` 等）
- Python 会等待你完成调试

---

### 场景 2：不需要窗口，但需要错误信息（后台运行）

**设置**：
1. `src/main.py`：使用无窗口模式
   ```python
   eng_intelligent = matlab.engine.start_matlab("-nodesktop")
   eng_baseline = matlab.engine.start_matlab("-nodesktop")
   ```

2. `src/experiment_runner.py`：启用调试模式（可选）
   ```python
   ENABLE_MATLAB_DEBUG = True  # 仍然可以捕获错误信息
   ```

**效果**：
- 不会打开 MATLAB 窗口
- 如果出错，Python 会显示详细的错误信息
- 无法交互式调试，但可以通过错误信息定位问题

---

### 场景 3：完全关闭调试（最快运行速度）

**设置**：
1. `src/main.py`：使用无窗口模式（可选）
   ```python
   eng_intelligent = matlab.engine.start_matlab("-nodesktop")
   eng_baseline = matlab.engine.start_matlab("-nodesktop")
   ```

2. `src/experiment_runner.py`：关闭调试模式
   ```python
   ENABLE_MATLAB_DEBUG = False
   ```

3. `matlab/utils/test_SINS_dynamic_UKF_153_forpython.m`：关闭详细调试输出
   ```matlab
   ENABLE_DETAILED_DEBUG = false
   ```

**效果**：
- 不会打开 MATLAB 窗口
- 不会设置断点
- 运行速度最快
- 如果出错，只显示基本错误信息

---

## 总结表格

| 设置项 | 位置 | 启用时的影响 | 如何控制 |
|--------|------|-------------|----------|
| MATLAB 窗口 | `src/main.py` 第131、133行 | 打开 MATLAB 可视化窗口 | 添加 `-nodesktop` 参数关闭 |
| 自动错误断点 | `src/experiment_runner.py` 第404行 | 出错时自动停在错误处 | `ENABLE_MATLAB_DEBUG = True/False` |
| 代码断点 | `src/experiment_runner.py` 第427-428行 | 执行到指定行时停住 | 通过 `ENABLE_MATLAB_DEBUG` 控制 |
| 详细调试输出 | `matlab/utils/test_SINS_dynamic_UKF_153_forpython.m` 第16行 | 输出详细的调试信息到控制台 | `ENABLE_DETAILED_DEBUG = true/false` |

---

## 快速切换指南

### 需要交互式调试时：
```python
# src/main.py
eng_intelligent = matlab.engine.start_matlab()  # 不添加 -nodesktop
eng_baseline = matlab.engine.start_matlab()

# src/experiment_runner.py
ENABLE_MATLAB_DEBUG = True
```

### 不需要窗口，但需要错误信息：
```python
# src/main.py
eng_intelligent = matlab.engine.start_matlab("-nodesktop")
eng_baseline = matlab.engine.start_matlab("-nodesktop")

# src/experiment_runner.py
ENABLE_MATLAB_DEBUG = True  # 仍然可以捕获错误
```

### 完全关闭调试：
```python
# src/main.py
eng_intelligent = matlab.engine.start_matlab("-nodesktop")
eng_baseline = matlab.engine.start_matlab("-nodesktop")

# src/experiment_runner.py
ENABLE_MATLAB_DEBUG = False

# matlab/utils/test_SINS_dynamic_UKF_153_forpython.m
ENABLE_DETAILED_DEBUG = false
```

---

## 注意事项

1. **MATLAB 窗口和调试的关系**：
   - MATLAB 窗口本身不会自动进入调试模式
   - 只有当设置了断点（`dbstop`）或启用了 `dbstop if error` 时，才会停在断点处
   - 如果窗口是打开的，停在断点时你可以在窗口中输入调试命令

2. **交互式调试的限制**：
   - Python 启动的 MATLAB 引擎是独立进程
   - 即使窗口打开，也无法像在 MATLAB IDE 中那样完全交互式调试
   - 但可以在 MATLAB 命令窗口中输入基本调试命令（`dbstep`, `dbcont`, `dbquit` 等）

3. **性能影响**：
   - 启用调试模式会显著降低运行速度
   - 打开 MATLAB 窗口会占用更多内存
   - 建议在调试完成后关闭所有调试功能



