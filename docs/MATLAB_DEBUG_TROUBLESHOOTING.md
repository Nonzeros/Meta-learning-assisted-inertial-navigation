# MATLAB 调试问题排查指南

## 问题：在 MATLAB IDE 中设置断点，Python 调用时没有停下来

### 原因

**关键问题**：Python 启动的 MATLAB 引擎是**独立的后台进程**，不是 MATLAB IDE！

- Python 通过 `matlab.engine.start_matlab()` 启动的是独立的 MATLAB 进程
- 在 MATLAB IDE 中设置的断点只对 IDE 中的 MATLAB 会话有效
- Python 启动的 MATLAB 引擎是另一个独立的进程，不会受到 IDE 中断点的影响

### 解决方案

#### 方案 1：通过代码在 MATLAB 引擎中设置断点（推荐）

在 Python 代码中，通过 MATLAB 引擎直接设置断点：

```python
# 在 MATLAB 引擎中设置断点
eng.eval("dbstop('in', 'test_SINS_dynamic_UKF_153_forpython', 'at', '69')", nargout=0)
```

或者使用 `debug_setup.m` 脚本（已配置）：

```python
ENABLE_MATLAB_DEBUG = True  # 在 experiment_runner.py 中启用
```

#### 方案 2：使用 `dbstop if error` 自动在错误处停住

这是最可靠的方法，当 MATLAB 函数出错时会自动停在错误处：

```python
eng.eval("dbstop if error", nargout=0)
```

#### 方案 3：使用 fprintf 输出调试信息（最简单）

在 MATLAB 文件中添加调试输出：

```matlab
fprintf('[调试] kf.xk 大小: [%d, %d]\n', size(kf.xk, 1), size(kf.xk, 2));
```

这些输出会显示在 Python 控制台中。

### 如何验证断点是否生效

1. **检查 Python 控制台输出**：
   - 如果看到 `[调试]` 开头的输出，说明代码执行到了那里
   - 如果 Python 卡住不动，可能是停在断点处了

2. **检查 MATLAB 引擎状态**：
   ```python
   # 在 Python 中检查 MATLAB 引擎的断点状态
   status = eng.eval("dbstatus", nargout=1)
   print(status)
   ```

3. **使用错误断点**：
   - 启用 `dbstop if error` 后，当 MATLAB 函数出错时会自动停住
   - Python 会卡住等待，你可以在 Python 控制台看到错误信息

### 实际调试方法

由于 Python 启动的 MATLAB 引擎是后台进程，**无法直接在 MATLAB IDE 中交互式调试**。推荐使用以下方法：

#### 方法 1：使用 fprintf 输出调试信息（最实用）

在 MATLAB 文件中添加详细的调试输出：

```matlab
fprintf('[调试] 准备调用 kffeedback\n');
fprintf('[调试] kf.n = %d\n', kf.n);
fprintf('[调试] kf.xk 大小: [%d, %d]\n', size(kf.xk, 1), size(kf.xk, 2));
fprintf('[调试] kf.xfb 大小: [%d, %d]\n', size(kf.xfb, 1), size(kf.xfb, 2));
```

这些输出会显示在 Python 控制台中，你可以看到变量的值。

#### 方法 2：使用 try-catch 捕获错误并输出详细信息

```matlab
try
    [kf, ins] = kffeedback(kf, ins, 1, 'avp');
catch ME
    fprintf('========== 错误信息 ==========\n');
    fprintf('错误消息: %s\n', ME.message);
    fprintf('kf.xk 大小: [%d, %d]\n', size(kf.xk, 1), size(kf.xk, 2));
    fprintf('kf.xfb 大小: [%d, %d]\n', size(kf.xfb, 1), size(kf.xfb, 2));
    rethrow(ME);
end
```

#### 方法 3：使用 dbstop if error（自动在错误处停住）

```python
# 在 Python 中启用
eng.eval("dbstop if error", nargout=0)
```

当 MATLAB 函数出错时，Python 会卡住，你可以在 Python 控制台看到详细的错误堆栈信息。

### 为什么 Python 会卡住？

当 MATLAB 停在断点处时：
- MATLAB 引擎会等待调试命令
- Python 会一直等待 MATLAB 返回结果
- 这是**正常现象**，说明断点生效了

但是，由于是后台进程，你**无法直接在 MATLAB IDE 中交互式调试**。

### 推荐的调试流程

1. **启用调试模式**：
   ```python
   ENABLE_MATLAB_DEBUG = True
   ```

2. **在 MATLAB 文件中添加调试输出**：
   ```matlab
   fprintf('[调试] 变量值: ...\n');
   ```

3. **运行 Python 代码**，查看控制台输出

4. **如果出错**，查看 Python 控制台的错误信息

5. **根据错误信息修复代码**

### 总结

- ❌ **不能**：在 MATLAB IDE 中设置断点来调试 Python 启动的 MATLAB 引擎
- ✅ **可以**：通过代码在 MATLAB 引擎中设置断点
- ✅ **可以**：使用 `dbstop if error` 自动在错误处停住
- ✅ **可以**：使用 `fprintf` 输出调试信息到 Python 控制台
- ✅ **可以**：使用 try-catch 捕获错误并输出详细信息

最实用的方法是**使用 fprintf 输出调试信息**，这样可以直接在 Python 控制台看到变量的值，无需交互式调试。


