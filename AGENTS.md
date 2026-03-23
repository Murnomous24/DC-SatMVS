# AGENTS.md

本文件提供给在本仓库执行任务的自动化编码代理（agentic coding agents）。
目标：在不破坏现有训练/推理流程的前提下，高效完成修改、验证与交付。

## 1) 仓库概览

- 项目类型：PyTorch 多视角卫星影像深度估计研究代码。
- 关键入口：
  - `train_TLC.py`：TLC/WHU-TLC 训练与测试
  - `train_WHU.py`：WHU 训练与测试
  - `predict.py`：推理与结果导出
- 关键目录：
  - `dataset/`：数据列表生成、读盘、预处理、数据集定义
  - `modules/`：warping、深度范围、特征融合等底层模块
  - `networks/`：模型与损失函数
  - `tools/`：指标、日志、RPC 相关工具

## 2) Cursor / Copilot 规则检查

已检查以下规则文件位置：

- `.cursor/rules/`
- `.cursorrules`
- `.github/copilot-instructions.md`

当前仓库中未发现以上规则文件。
若后续新增，请将其视为高优先级约束并同步更新本文件。

## 3) 环境与依赖（按仓库事实）

- README 历史环境：Python 3.7，PyTorch 1.7.1 + cu110。
- 代码中可见依赖：`torch`、`torchvision`、`tensorboardX`、`numpy`、`opencv-python`、`imageio`、`matplotlib`、`einops`。
- README 涉及地理库：`gdal`、`proj`、`geos`（并非仓库内自动安装）。
- 仓库没有 `requirements.txt`、`pyproject.toml`、`setup.py`，不要假设存在一键安装命令。

## 4) Build / Lint / Test 命令

说明：本仓库无传统 build、lint、pytest 配置；以下命令是可执行的最低限校验流程。

### 4.1 语法与导入健康检查（推荐）

```bash
python -m py_compile train_TLC.py train_WHU.py predict.py
python -m py_compile dataset/*.py modules/*.py networks/*.py tools/*.py
```

### 4.2 训练命令

```bash
# TLC (RPC)
python train_TLC.py --mode train --model samsat --geo_model rpc --dataset_root <DATA_ROOT> --batch_size 1 --gpu_id 0

# TLC (Pinhole)
python train_TLC.py --mode train --model samsat --geo_model pinhole --dataset_root <DATA_ROOT> --batch_size 1 --gpu_id 0

# WHU
python train_WHU.py --mode train --model samsat --geo_model pinhole --dataset_root <DATA_ROOT> --batch_size 1 --gpu_id 0
```

### 4.3 测试/评估命令

```bash
# TLC test
python train_TLC.py --mode test --model samsat --geo_model rpc --dataset_root <DATA_ROOT> --loadckpt <CKPT_PATH> --batch_size 1 --gpu_id 0

# WHU test
python train_WHU.py --mode test --model samsat --geo_model pinhole --dataset_root <DATA_ROOT> --loadckpt <CKPT_PATH> --batch_size 1 --gpu_id 0
```

### 4.4 推理命令

```bash
python predict.py --model samsat --geo_model rpc --dataset_root <DATA_ROOT> --loadckpt <CKPT_PATH> --batch_size 1 --gpu_id 0
```

默认输出目录：`./mvs_results/`。

### 4.5 单测（single test）约定

本仓库目前没有 pytest 用例；“单测”按“单样本最小集验证”执行：

1. 准备极小数据集（1 个 scene 或少量样本）。
2. 使用 `--batch_size 1` 运行一次 `--mode test` 或 `predict.py`。
3. 验证前向、指标打印和结果落盘。

示例：

```bash
python train_TLC.py --mode test --model samsat --geo_model rpc --dataset_root <MINI_DATA_ROOT> --loadckpt <CKPT_PATH> --batch_size 1 --gpu_id 0
```

若未来新增 pytest，用例粒度单测命令：

```bash
pytest path/to/test_file.py::test_name -q
```

## 5) 代码风格与实现约定

### 5.1 Imports

- 顺序：标准库 -> 第三方 -> 本地模块。
- 同组按字母序，避免随意穿插。
- 新增代码避免 `from xxx import *`。
- 避免循环导入；数据集分派使用 `dataset.find_dataset_def` 模式。

### 5.2 Formatting

- 遵循 PEP 8，4 空格缩进。
- 建议行宽约 88-100。
- 长表达式使用括号换行，保持可读对齐。
- 不做无关的大规模格式化提交。

### 5.3 Types

- 现有代码类型标注较少；新增公共函数建议补齐输入/返回类型。
- 对张量在注释或 docstring 中标注形状（如 `B,N,C,H,W`）。
- 与现有接口兼容优先，避免因类型注解引入行为变化。

### 5.4 Naming

- 类名：`PascalCase`。
- 函数/变量：`snake_case`。
- 常量：`UPPER_SNAKE_CASE`。
- 保持阶段键名约定：`stage1` / `stage2` / `stage3`。

### 5.5 Error Handling

- 对关键参数做早失败校验（`assert` 或显式异常）。
- 路径写入前检查并创建目录。
- 不静默吞异常，除非有明确降级逻辑。
- 错误信息应能直接定位参数或文件问题。

### 5.6 Logging / 可观测性

- 延续现有训练日志风格：epoch、iter、loss、耗时、指标。
- TensorBoard 继续使用 `save_scalars` 与 `save_images`。
- 新增分支仅添加必要日志，避免刷屏。

### 5.7 Device / 性能

- 代码默认 CUDA（大量 `.cuda()`），新增逻辑保持 device 一致。
- 避免循环内重复创建大张量，优先复用缓存。
- 谨慎使用 inplace 操作，确保 autograd 正常。

### 5.8 数据/模型接口契约

- 数据集返回键保持稳定：`imgs`、`cam_para`、`depth`、`mask`、`depth_values`、`out_view`、`out_name`。
- 模型输出至少包含 `outputs['stageX']['depth']` 与 `outputs['stageX']['photometric_confidence']`。
- ST-SatMVS 训练路径中还使用 `depth_filtered`。

## 6) 代理执行建议

1. 先读入口脚本与目标模块，确认数据流再修改。
2. 只做任务相关的最小修改，避免顺手重构。
3. 修改后先跑 `py_compile`，再跑最小测试/推理命令。
4. 涉及训练逻辑时，至少验证 1 个 batch 可前向与反向。
5. 输出需包含：改动点、原因、验证方式。

## 7) 已知注意点

- `train_WHU.py` 使用了 `args.which`，但参数定义处已注释；运行前先核对。
- 修改损失函数时，联动检查 `train_sample`、`test_sample`、`predict_sample` 键名一致性。
- `predict.py` 固定写出到 `./mvs_results/`，批量实验避免覆盖冲突。

## 8) 提交前检查

- 只提交任务相关改动。
- 参数默认值与 CLI 行为保持向后兼容（除非需求要求改变）。
- 至少完成语法检查 + 一条最小运行验证。
- 不引入私有路径、密钥、账号信息。
- 若行为变更，更新文档。

---

当仓库后续新增标准化 lint/test/format 配置，请优先更新本文件第 4、5 节。
