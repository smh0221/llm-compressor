# LLM Compressor 项目说明文档

> 版本：`0.12.1.dev0`（基于当前 `xmain` 分支）
> 适用对象：希望了解项目结构、快速上手、参与开发或自行编译安装的工程师。

---

## 一、项目概述

`llmcompressor`（包名 `llmcompressor`，PyPI 名 `llmcompressor`）是 vLLM 项目维护的**大语言模型压缩库**，用于把模型量化、剪枝后导出为 `compressed-tensors` 格式，供 vLLM 高效推理部署。

核心能力：

- 覆盖权重、激活、KV Cache、Attention 的全面量化算法与变换
- 与 Hugging Face Transformers 模型/仓库无缝集成
- 产物保存为 `compressed-tensors` 格式，直接被 vLLM 加载
- 支持 DDP（分布式数据并行）与磁盘卸载（disk offloading），可压缩超大模型

### 支持的精度与类型

| 类别 | 支持的方案 |
|------|-----------|
| 激活量化 | W8A8 (int8/fp8)、W4AFP8、Microscale (NVFP4、MXFP4、MXFP8) |
| 混合精度 | W4A16、W8A16、MXFP8A16、MXFP4A16、NVFP4A16 |
| Attention / KV Cache | FP8、NVFP4 |

### 支持的算法

Simple PTQ、GPTQ、AWQ、SmoothQuant、AutoRound、Rotation-based（SpinQuant、QuIP）。

> 注意：24 稀疏（Sparse compression）因缺乏硬件支持已不再被 LLM Compressor 支持。

---

## 二、代码结构

### 2.1 仓库顶层布局

```
llm-compressor/
├── src/llmcompressor/      # 主源码包（177 个 .py 文件）
├── examples/               # 各类量化/剪枝端到端示例（按算法、精度、模型类型分目录）
├── experimental/           # 实验性特性（attention、mistral 等）
├── tests/                  # 测试：unit / e2e / examples / lmeval / sparsity
├── docs/                   # 文档源码（zensical 构建，发布到 docs.vllm.ai）
├── tools/                  # 辅助脚本
├── setup.py                # 打包与依赖定义（核心安装入口）
├── pyproject.toml          # 构建后端、ruff/mypy/pytest 配置
├── Makefile                # 常用开发命令（style/quality/test/build/clean）
├── MANIFEST.in             # 打包包含/排除规则
├── README.md               # 项目主页说明
├── CONTRIBUTING.md         # 贡献指南
└── zensical.toml           # 文档站点配置
```

### 2.2 源码包 `src/llmcompressor/` 模块说明

包采用 `src` 布局（`package_dir={"": "src"}`），主要子模块如下（括号内为该目录 `.py` 文件数）：

| 模块 | 文件数 | 职责 |
|------|-------:|------|
| `entrypoints/` | 11 | **对外入口**。`oneshot`（一次性校准压缩）、`model_free`（无需 HF 模型定义的 PTQ 通道）、前后处理工具 |
| `modifiers/` | 63 | **压缩算法核心**。所有量化/剪枝/变换算法以 Modifier 形式实现，工厂模式可扩展 |
| `core/` | 8 | **压缩会话框架**。`CompressionSession`、生命周期、事件系统、状态管理 |
| `pipelines/` | 18 | **校准流水线**。basic / sequential / independent / data_free 等策略编排 |
| `recipe/` | 4 | **配方系统**。声明式描述压缩配置，支持分阶段执行与元数据跟踪 |
| `observers/` | 6 | **统计观测器**。min-max、MSE、imatrix 等，用于量化范围估计 |
| `modeling/` | 15 | **模型结构适配**。MoE 线性化、DeepSeek-V3.2、权重融合、norm 偏移、模型补丁 |
| `transformers/` | 21 | **HF Transformers 集成**。压缩张量工具、数据集封装、tracing 调试 |
| `pytorch/` | 10 | **PyTorch 工具**。模型加载/保存辅助、通用 PyTorch 工具 |
| `datasets/` | 2 | **校准数据集**。校准数据格式化、dataloader 构建、分布式分区 |
| `args/` | 5 | **参数解析**。`ModelArguments`、`DatasetArguments`、`RecipeArguments` |
| `utils/` | 9 | 通用工具：分布式、辅助函数、metric 日志等 |
| `logger.py` / `sentinel.py` / `typing.py` | - | 日志配置、哨兵对象、类型定义 |
| `version.py` | - | 由 `setuptools-scm` 自动生成，**请勿手动修改或纳入版本控制** |

### 2.3 `modifiers/` 子模块（算法清单）

```
modifiers/
├── factory.py / interface.py / modifier.py   # 基类、工厂、接口
├── quantization/      # 量化主入口（QuantizationModifier、calibration、gptq 集成）
├── gptq/              # GPTQ 算法实现
├── awq/               # 兼容性 shim（已迁移至 transform/awq）
├── smoothquant/       # 兼容性 shim（已迁移至 transform/smoothquant）
├── autoround/         # AutoRound 算法
├── transform/         # 旋转/变换类算法
│   ├── awq/           # AWQ（当前实现位置）
│   ├── smoothquant/   # SmoothQuant（当前实现位置）
│   ├── quip/          # QuIP
│   ├── spinquant/     # SpinQuant
│   └── imatrix/
├── pruning/           # 剪枝
│   ├── magnitude/  constant/  wanda/  sparsegpt/
├── obcq/              # OBCQ / SparseGPT 基类
├── logarithmic_equalization/
├── experimental/
└── utils/
```

> 重要：`modifiers/awq` 与 `modifiers/smoothquant` 现为**向后兼容垫片（shim）**，会发出 DeprecationWarning。新代码应从 `llmcompressor.modifiers.transform`（或 `transform.awq` / `transform.smoothquant`）导入。

### 2.4 核心调用链（oneshot 流程）

`entrypoints/oneshot.py` 中的 `Oneshot` 类驱动整个一次性校准生命周期：

1. **Preprocessing（预处理）**：加载模型与 tokenizer/processor；解绑共享的输入输出 embedding；为带量化配置的保存打补丁。
2. **Oneshot Calibration（校准）**：通过全局 `CompressionSession`，按 recipe 中定义的 Modifier（如 `GPTQModifier`、`SparseGPTModifier`）优化模型。
3. **Postprocessing（后处理）**：把模型、tokenizer/processor、配置保存到 `output_dir`。

入参由 `args/` 解析为 `model_args` / `dataset_args` / `recipe_args` 三组。

### 2.5 公开 API（顶层导出）

`src/llmcompressor/__init__.py` 导出的主要入口：

```python
from llmcompressor import (
    oneshot, Oneshot, model_free_ptq,        # 压缩入口
    active_session, create_session,           # 会话管理
    callbacks, reset_session,
    logger, configure_logger, LoggerConfig,
    __version__, version,
)
```

控制台脚本（`entry_points`）：

- `llmcompressor.trace` → tracing 调试工具
- `llmcompressor.reindex_fused_weights` → 已废弃

---

## 三、编译与安装方法

### 3.1 环境前置要求

来自 `setup.py` 与官方安装文档的声明：

- **操作系统**：Linux（GPU 支持推荐）
- **Python**：`>=3.10`
- **pip**：建议升级到最新版 `python -m pip install --upgrade pip`

> 说明：`setup.py` 中对依赖版本的上界（如 `torch>=2.10.0,<=2.12.0`、`transformers>=5.9.0`）针对的是该库目标发布版本的依赖约束。dev 构建（`BUILD_TYPE=dev`，默认）下仅校验下界。当前仓库实测环境为 Python 3.12 + CUDA torch 2.12，已正常安装（`pip show llmcompressor` 显示 `0.12.1.dev0`）。

### 3.2 安装方式

#### 方式一：从 PyPI 安装稳定版（普通用户）

```bash
pip install llmcompressor
# 指定版本
pip install llmcompressor==0.5.1
```

#### 方式二：从 GitHub main 分支安装最新开发版

```bash
pip install git+https://github.com/vllm-project/llm-compressor.git
```

#### 方式三：从本地克隆源码安装

普通安装：

```bash
pip install .
```

**开发模式（推荐，含开发依赖）**：

```bash
pip install -e ./[dev]
```

仅可编辑安装、不拉取依赖（当 `build.md` 所记录、依赖已就绪时使用）：

```bash
pip install -e . --no-deps
```

> 开发模式建议同时从源码安装配套库 `compressed-tensors`：
> ```bash
> git clone https://github.com/vllm-project/compressed-tensors.git
> pip install -e ./compressed-tensors
> ```

### 3.3 核心运行时依赖（`setup.py` install_requires）

| 依赖 | 用途 |
|------|------|
| `torch` | 深度学习框架 |
| `transformers` | HF 模型加载 |
| `compressed-tensors` | 压缩张量格式（核心配套库） |
| `datasets` | 校准数据集 |
| `accelerate` | 设备分发/卸载 |
| `auto-round` | AutoRound 算法 |
| `numpy` / `pyyaml` / `requests` / `tqdm` / `loguru` / `pillow` / `nvidia-ml-py` | 基础工具 |

可选 extras：

- `[dev]`：pytest、ruff、mypy、pre-commit、zensical（文档）、trl、torchvision、librosa 等
- `[qwen]`：`qwen_vl_utils`

### 3.4 构建 wheel 包

构建类型由环境变量 `BUILD_TYPE` 控制，取值 `release` / `nightly` / `dev`（默认 `dev`）：

```bash
# 默认 dev 构建
make build
# 等价于
python3 setup.py sdist bdist_wheel

# nightly 构建
make build BUILD_TYPE=nightly
```

版本号由 `setuptools-scm` 依据 git tag 自动推导并写入 `src/llmcompressor/version.py`。

---

## 四、开发常用命令（Makefile）

| 命令 | 作用 |
|------|------|
| `make style` | 用 ruff 自动格式化并修复 lint（对 `src tests examples setup.py`） |
| `make quality` | 仅检查 ruff lint 与格式，不修改 |
| `make test` | 运行 pytest（默认忽略 transformers/examples/sparsity，可用 `TARGETS=` 控制） |
| `make build` | 构建 sdist + wheel |
| `make clean` | 清理缓存、构建产物、`.pyc` 等 |

测试相关：

```bash
# 运行全部测试（耗时长，部分用例需多卡 GPU）
make test

# 只跑某类，通过 TARGETS 控制是否包含 transformers/examples/sparsity
make test TARGETS="transformers"
```

pytest 标记（`pyproject.toml`）：`smoke` / `sanity` / `regression` / `integration` / `unit` / `example` / `multi_gpu`。

### 代码规范

- **ruff**：行长 88，启用 `E/F/W/I` 规则；`isort` 一方包为 `llmcompressor`
- **mypy**：检查 `src/llmcompressor`
- 提交前请运行 `make style && make quality`，并安装 `pre-commit`

---

## 五、快速上手示例

把 `Qwen3-30B-A3B` 用 RTN 算法量化为 FP8（权重+激活）：

```python
from compressed_tensors.offload import dispatch_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_ID = "Qwen/Qwen3-30B-A3B"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# 配方：权重 FP8（block_size 128），激活推理时动态 FP8
recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_BLOCK",
    ignore=["lm_head", "re:.*mlp.gate$"],
)

# 应用量化
oneshot(model=model, recipe=recipe)

# 保存为 compressed-tensors 格式
SAVE_DIR = MODEL_ID.split("/")[1] + "-FP8-BLOCK"
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
```

用 vLLM 加载量化后的模型：

```python
from vllm import LLM
model = LLM("Qwen/Qwen3-30B-A3B-FP8-BLOCK")
output = model.generate("My name is")
```

更多端到端示例见 `examples/` 目录（按算法/精度/模型类型分门别类，每个子目录含 README）。

---

## 六、文档构建

文档使用 **zensical** 构建（配置 `zensical.toml`，源码在 `docs/`）：

```bash
pip install -e ".[dev]"
python docs/scripts/zensical_gen_files.py
zensical build      # 产物在 site/
```

在线文档：<https://docs.vllm.ai/projects/llm-compressor/en/latest/>

---

## 七、参考

- 仓库主页：<https://github.com/vllm-project/llm-compressor>
- 贡献指南：`CONTRIBUTING.md`
- 开发者教程：`docs/developer-tutorials/`（新增 Modifier、Observer 等）
- License：Apache 2.0
