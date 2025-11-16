# 项目目录结构规划方案

## 当前项目分析

这是一个**动力学模型辅助导航**项目，使用Python调用MATLAB进行导航解算。当前项目包含：
- Python主程序和模块
- MATLAB脚本
- 训练数据和实验数据
- 训练好的模型文件
- 导航日志
- 处理后的数据

## 推荐的目录结构

```
dynamic_metalearning/
├── README.md                          # 项目说明文档
├── requirements.txt                   # Python依赖包列表
├── .gitignore                         # Git忽略文件配置
│
├── src/                               # 源代码目录
│   ├── __init__.py
│   ├── main.py                        # 主程序入口（原 navigation_dynamic_python_matlab_main.py）
│   │
│   ├── core/                          # 核心功能模块
│   │   ├── __init__.py
│   │   ├── navigation.py              # 导航核心逻辑
│   │   ├── dynamic_model.py           # 动力学模型（原 intelligent_dynamic_module.py）
│   │   └── ukf_filter.py              # UKF滤波相关
│   │
│   ├── models/                        # 机器学习模型
│   │   ├── __init__.py
│   │   ├── mlmodel.py                 # 原 script/mlmodel.py
│   │   └── pf_model.py                # 原 script/pf_model.py
│   │
│   ├── utils/                         # 工具函数
│   │   ├── __init__.py
│   │   ├── data_loader.py             # 数据加载工具（utils.load_data等）
│   │   ├── data_formatter.py          # 数据格式化工具（utils.format_data等）
│   │   └── visualization.py           # 可视化工具
│   │
│   └── matlab_interface/              # MATLAB接口模块
│       ├── __init__.py
│       ├── matlab_engine.py           # MATLAB引擎管理
│       └── matlab_functions.py        # MATLAB函数封装
│
├── matlab/                            # MATLAB脚本（保持不变）
│   ├── a2mat_subfun.m
│   ├── a2qua_subfun.m
│   ├── av2imu_main3.m
│   ├── llh2xyz_main.m
│   ├── llh2xyz_subfun.m
│   ├── pure_ins_solve.m
│   ├── SINS_dynamic_UKF153_init.m
│   ├── test_main.m
│   ├── test_SINS_dynamic_UKF_153_forpython.m
│   └── xyz2llh_subfun.m
│
├── configs/                           # 配置文件目录
│   ├── default.yaml                   # 默认配置
│   ├── experiment_config.yaml         # 实验配置
│   └── matlab_config.yaml             # MATLAB相关配置
│
├── data/                              # 数据目录（保持不变，但建议分类更清晰）
│   ├── raw/                           # 原始数据
│   │   ├── experiment/
│   │   ├── experiment2/
│   │   ├── training/
│   │   └── training-transfer/
│   │
│   ├── processed/                     # 处理后的数据（原 ProcessedData/）
│   │   └── ...
│   │
│   └── results/                       # 结果数据（原 result/）
│       └── ...
│
├── models/                            # 训练好的模型文件（保持不变）
│   └── *.pth                          # PyTorch模型文件
│
├── logs/                              # 日志目录（原 navigation_logs/）
│   ├── navigation/                    # 导航日志
│   ├── training/                      # 训练日志
│   └── experiments/                   # 实验日志
│
├── scripts/                           # 辅助脚本
│   ├── train.py                       # 训练脚本（如果有）
│   ├── evaluate.py                    # 评估脚本
│   └── visualize.py                   # 可视化脚本
│
├── tests/                             # 测试目录
│   ├── __init__.py
│   ├── test_navigation.py
│   ├── test_dynamic_model.py
│   └── test_utils.py
│
└── docs/                              # 文档目录
    ├── api/                           # API文档
    ├── tutorials/                     # 教程文档
    └── papers/                        # 相关论文
```

## 目录说明

### 1. `src/` - 源代码目录
- **目的**：所有Python源代码统一放在这里，符合Python项目标准
- **结构**：
  - `main.py`: 主程序入口，从根目录移入
  - `core/`: 核心功能模块，包括导航、动力学模型、滤波等
  - `models/`: 机器学习模型定义
  - `utils/`: 工具函数，包括数据加载、格式化等
  - `matlab_interface/`: MATLAB接口封装，便于管理和维护

### 2. `configs/` - 配置文件目录
- **目的**：集中管理所有配置参数
- **内容**：
  - 数据集路径
  - 模型参数（dim_a, solver_type, numPar等）
  - MATLAB路径配置
  - 实验参数

### 3. `data/` - 数据目录重组
- **目的**：更清晰的数据分类
- **结构**：
  - `raw/`: 原始实验数据
  - `processed/`: 处理后的数据
  - `results/`: 实验结果数据

### 4. `logs/` - 日志目录
- **目的**：统一管理所有日志
- **结构**：按类型分类（navigation, training, experiments）

### 5. `scripts/` - 辅助脚本
- **目的**：存放可执行的辅助脚本，如训练、评估等

### 6. `tests/` - 测试目录
- **目的**：单元测试和集成测试

### 7. `docs/` - 文档目录
- **目的**：项目文档、API文档、教程等

## 迁移步骤建议

### 阶段1：创建新目录结构
1. 创建所有新目录
2. 保持原有文件不动

### 阶段2：移动文件
1. **Python源代码**：
   - `navigation_dynamic_python_matlab_main.py` → `src/main.py`
   - `script/intelligent_dynamic_module.py` → `src/core/dynamic_model.py`
   - `script/mlmodel.py` → `src/models/mlmodel.py`
   - `script/pf_model.py` → `src/models/pf_model.py`
   - 创建 `src/utils/` 并移动utils相关代码

2. **数据目录**：
   - `ProcessedData/` → `data/processed/`
   - `result/` → `data/results/`
   - `data/experiment*` → `data/raw/experiment*/`

3. **日志目录**：
   - `navigation_logs/` → `logs/navigation/`

### 阶段3：更新导入路径
更新所有Python文件中的导入语句，例如：
- `import utils` → `from src.utils.data_loader import load_data`
- `import mlmodel` → `from src.models.mlmodel import ...`
- `from dynamic_model_imu_navigation.matlab_python_connetion.intelligent_dynamic_module import ...` → `from src.core.dynamic_model import ...`

### 阶段4：创建配置文件
将硬编码的参数提取到配置文件中：
- 数据集路径
- 模型参数
- MATLAB路径
- 实验参数

### 阶段5：创建项目文档
- 更新 `README.md`
- 创建 `requirements.txt`
- 创建 `.gitignore`

## 配置文件示例

### `configs/default.yaml`
```yaml
# 数据集配置
dataset:
  name: 'neural-fly'
  folder: 'data/raw/experiment2'
  features: ['v', 'q', 'pwm']
  label: 'fa'

# 模型配置
model:
  dim_a: 3
  stopping_epoch: 900
  model_name_template: "{dataset}_dim-a-{dim_a}_{'-'.join(features)}"

# 求解器配置
solver:
  type: 2  # 1=kf, 2=pf
  numPar: 30
  lambda1: 0.1
  loss_type: 'crossentropy-loss'

# MATLAB配置
matlab:
  engine_path: "F:\\code_for_guide\\Neural-Fly Enables Rapid Learning for Agile Flight in Strong Winds\\neural-fly-main2_mydata"
  matlab_scripts_path: "matlab"

# 实验配置
experiment:
  adapt_end_index: 100
  loops: 2000
  data_num: 2
  time_step: 0.02

# 物理参数
physics:
  mass: 2.6  # kg
  gravity: 9.8  # m/s^2
```

## 优势

1. **标准化**：符合Python项目最佳实践
2. **可维护性**：代码组织清晰，易于查找和修改
3. **可扩展性**：新功能可以轻松添加到对应目录
4. **可测试性**：独立的测试目录便于编写和运行测试
5. **配置管理**：配置文件集中管理，便于不同环境切换
6. **文档化**：专门的文档目录便于项目文档管理

## 注意事项

1. **MATLAB路径**：确保MATLAB脚本路径配置正确
2. **相对路径**：更新所有文件路径为相对路径
3. **导入路径**：需要更新所有Python导入语句
4. **向后兼容**：如果其他项目依赖当前结构，需要逐步迁移

## 下一步

1. 确认此目录结构是否符合需求
2. 如需调整，请说明具体需求
3. 确认后可以开始执行迁移（如果需要的话）


