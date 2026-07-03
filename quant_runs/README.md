# quant_runs/ — 量化运行脚本库

本目录承载本仓库（fork 自 [vllm-project/llm-compressor](https://github.com/vllm-project/llm-compressor)）上**多模型 × 多量化方法矩阵**的可运行脚本，与上游 `examples/` **完全隔离**，便于跟随上游 rebase 时保持干净。

---

## 一、底层库简介（llm-compressor）

`llmcompressor` 是 vLLM 维护的**大模型压缩库**：把模型量化/剪枝后导出为 `compressed-tensors` 格式，供 vLLM 高效推理。

- 量化方案：W8A8(int8/fp8)、W4AFP8、Microscale(NVFP4/MXFP4/MXFP8)、混合精度(W4A16/W8A16/…)、KV Cache(FP8/NVFP4)
- 算法：Simple PTQ、GPTQ、AWQ、SmoothQuant、AutoRound、Rotation(SpinQuant/QuIP)
- 入口：`oneshot`（一次性校准压缩）走「预处理 → 校准（按 recipe 中的 Modifier）→ 保存」三步；另有 `model_free_ptq`（无需校准数据的 PTQ 通道）

> `modifiers/awq`、`modifiers/smoothquant` 现为向后兼容垫片（会 DeprecationWarning），新代码应从 `llmcompressor.modifiers.transform` 导入。

---

## 二、环境构建与安装

本仓库以 `pip install -e . --no-deps` 安装，**setup.py 声明的运行依赖不会被自动安装**，需先用 `quant_runs/requirements.txt` 补齐。

> **本机 torch 为定制版本**（`2.9.0`），**不可替换为官方 torch**。故 `requirements.txt` 里 torch/torchvision 刻意钉在定制版本对应号，安装时须用 `--no-deps` 保护，详见下文。

```bash
# 0) 自备并激活一个 Python 环境（conda env 或 venv 均可），Python >= 3.10
conda activate <your-env>      # 或 source /path/to/venv/bin/activate

# 1) 安装运行依赖（--no-deps 绕过传递依赖对 torch 的强制，保护定制 torch 轮子）
pip install -r quant_runs/requirements.txt --no-deps

# 2) 以可编辑模式安装本地包（不带依赖，依赖由上一步提供）
pip install -e . --no-deps
```

依赖要点（详见 `requirements.txt`，已对齐本机实测可用组合，transformers 5.x 线）：

- **必须 `--no-deps` 安装**：`compressed-tensors>=0.17` 强制要求 `torch>=2.10.0`，而本机定制 torch 为 `2.9.0`。不加 `--no-deps` 会触发 `ResolutionImpossible` 冲突并试图卸载定制轮子。`--no-deps` 跳过依赖解析，保住 torch 2.9.0。
- **torch / torchvision 钉定制版本**：`torch==2.9.0`、`torchvision==0.24.0`。**切勿升级到 setup.py 要求的 2.10+**，否则覆盖定制 torch 轮子、多卡不可用。
- **numpy 受 Python 版本上限约束**：`numpy==2.2.6`。`numpy>=2.3` 要求 Python≥3.11、`>=2.5` 要求 Python≥3.12，本机 Python 3.10 装不了，故取 3.10 可用的最高版 2.2.6（偏离 setup.py 的钉版）。
- **transformers >= 5**：脚本用的 MoE `*ForConditionalGeneration` 类（如 `Qwen3_5MoeForConditionalGeneration`）仅 5.x 提供。
- **qwen_vl_utils + torchvision**：图文校准所需（`process_vision_info`）。`requirements.txt` 已含；亦可 `pip install llmcompressor[qwen]`（仅含 `qwen_vl_utils`，图文校准还需 torchvision）。
- **omegaconf**：`common/` 的配置合并（YAML config + CLI dotlist）所需，`requirements.txt` 已含。

> 已校验组合（x0.12.0 基线）：`torch==2.9.0`（定制版本）、`transformers==5.12.1`、`compressed-tensors==0.17.2a20260626`、`llmcompressor==0.12.1.dev`，Qwen3.5-35B-A3B GPTQ W8A8 可跑通。

### 多卡运行

- 卡选择用 `CUDA_VISIBLE_DEVICES`；某卡被占时避开它，如 `export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7` 避开卡 0。
- config 里 `device_map: auto` 会把大模型（如 35B）按层分片到多张卡（模型并行），单卡放不下时必需。

### torch 兼容垫片（common/torch_compat.py）

`compressed-tensors>=0.17` 用了 `torch.accelerator.get_memory_info`（torch 2.10 才有的 API），定制 torch 2.9 缺此 API。`common/torch_compat.py` 用等价的 `torch.cuda.mem_get_info` 补齐，`import quant_runs.common` 时自动生效（见 `common/__init__.py`）。此垫片仅落在 `quant_runs/` 内，不改动 llmcompressor 库源码，保持与上游隔离、rebase 干净。

---

## 三、目录结构与约定

- **一级目录 = 模型系列**：当前有 `qwen/`，后续按需新增 `glm/`、`deepseek/` …
- **二级 = 具体脚本（方法_精度）**：每个脚本可单独运行；config 加载等公共逻辑统一抽到 `common/`。

```
quant_runs/
├── README.md                      # 本文件：库简介、安装、目录约定、命名规范、运行方式
├── requirements.txt               # 运行依赖（因 `-e . --no-deps` 安装需手动补齐）
├── __init__.py                    # 使 quant_runs 成为可导入包（脚本 import common 用）
├── common/                        # 跨脚本公共模块
│   ├── __init__.py                # 导出 load_config / copy_auxiliary_files，并自动应用 torch 兼容垫片
│   ├── io.py                      # config 加载（OmegaConf 读 YAML）+ 文件系统辅助（拷贝非权重辅助文件）
│   └── torch_compat.py            # torch 兼容垫片，为定制 torch 2.9 补齐 compressed-tensors 0.17 依赖的新 API
└── qwen/
    ├── qwen3_5_moe_gptq_w8a8.py                    # 脚本
    ├── qwen3_5_moe_gptq_w8a8.config.yaml           # 配置（本机路径）
    └── run_qwen3_5_moe_gptq_w8a8.sh                # 运行包装（读 config）
```

每个脚本配套**三件套**：`<name>.py` + `<name>.config.yaml` + `run_<name>.sh`。config 用 YAML，键为 snake_case（与 `.py` 里 `cfg.xxx` 属性同名，如 `model_path`），首次使用前填好本机路径：

```bash
# 编辑 qwen3_5_moe_gptq_w8a8.config.yaml 填入 model_path / save_dir
```

glm/、deepseek/ 等后续按需补充（沿用下述命名约定）。

### 命名规范

文件名统一为 `<模型名>_<方法>_<精度>.py`（全小写 + 下划线）：

| 字段 | 取值示例 |
| --- | --- |
| 模型名 | `qwen3_5_moe`、`qwen3_next`、`glm4_7`、`deepseek_v4` |
| 方法 | `gptq`、`smoothquant`、`spinquant`、`awq`、`rtn`、`gptq_smoothquant`（组合用下划线连） |
| 精度 | `w8a8`、`w4a8`、`w8a16`、`w4a16`、`w4a4`、`w2a16` |

示例：`qwen3_5_moe_gptq_w8a8.py`、`glm4_7_smoothquant_w8a8.py`、`deepseek_v4_spinquant_w4a16.py`。

### 注释风格

脚本注释统一简洁中文、以单行为主，新增脚本沿用：

- 优先单行 `#` 注释，尽量不用多行/块注释，必要时才用
- 行尾最后一个标点不加（如句号、逗号）
- 段落用**裸名词标签**（如 `# 量化`、`# recipe 构造`、`# 保存压缩后权重`），不用 `# ----- ... -----` 分隔条
- 需补充缘由时用中文逗号「，」接在同一行，而非冒号或括号（如 `# 加载模型，MoE 须用 load_context 包裹以线性化专家张量`）
- 模块 docstring 为「一句标题 + 简短描述」（见 `qwen/qwen3_5_moe_gptq_w8a8.py` 顶部）
- 只在逻辑不够自解释处写注释，不为显而易见的代码堆砌注释

---

## 四、common/ 公共模块

跨脚本一致的逻辑统一抽到 `quant_runs/common/`，对外导出：

**`io.py` — config 加载 + 文件系统辅助**

| 函数 | 作用 |
| --- | --- |
| `load_config()` | 从 YAML config 加载并返回 OmegaConf 配置对象（可 `cfg.xxx` 属性访问）。CLI 只接受唯一键 `config=<path>` 指定 YAML 文件路径（缺失时友好退出）；`model_path` / `save_dir` 缺失时友好退出。**全部参数**（通用 + 脚本特有）都写在 YAML 里，不再有代码内置默认——脚本用到的键须在 YAML 写全，漏写则脚本访问时报 `ConfigAttributeError`。 |
| `copy_auxiliary_files(src_dir, dest_dir)` | 把源 checkpoint 中的**非权重辅助文件**（tokenizer、各类 config 等）拷到输出目录；跳过 `.safetensors`/`.bin` 权重分片及其 index（由 `save_pretrained` / `save_mtp_tensors_to_checkpoint` 产出），不覆盖已存在的同名文件。 |

**`torch_compat.py` — torch 兼容垫片**

| 函数 | 作用 |
| --- | --- |
| `apply_torch_compat()` | 为定制 torch 2.9 补齐 `torch.accelerator.get_memory_info`（compressed-tensors 0.17 依赖、torch 2.10 才有的 API），用等价的 `torch.cuda.mem_get_info` 实现。`import quant_runs.common` 时自动调用（见 `common/__init__.py`），脚本无需显式调用。仅在 API 缺失时打补丁，torch≥2.10 环境下为空操作。 |

**每个脚本一个 YAML**，写全该脚本用到的所有参数（通用 + 特有）。通用参数键：`model_path`、`save_dir`（二者必填）、`num_calibration_samples`、`max_seq_length`、`dataset_id`、`dataset_split`、`device_map`、`seed`、`preprocessing_num_workers`、`dataloader_num_workers`；脚本特有键（如 `smoothing_strength`）也一并写入。可参照同名 `*.config.yaml` 现有字段。

> 脚本顶部用 `sys.path.insert(0, <repo_root>)` 后再 `from quant_runs.common import ...`，因此无论从哪个 CWD、用 `python xxx.py` 还是 `run_*.sh` 启动都能找到 `common`（`quant_runs` 未随 `pip install -e .` 安装到环境）。

---

## 五、运行方式

每个脚本支持两种调用，配置一律写在 **YAML config** 里（每个脚本一个 YAML，含全部参数）。脚本**不内置 conda**，请先自行激活环境：

```bash
conda activate <your-env>      # 或 source /path/to/venv/bin/activate

# 1) 推荐：run_*.sh（读同名 .config.yaml，可 CONFIG= 覆盖配置路径）
bash quant_runs/qwen/run_qwen3_5_moe_gptq_w8a8.sh

# 2) 直接 .py + config（CLI 只接受 config=<path> 一个键）
python quant_runs/qwen/qwen3_5_moe_gptq_w8a8.py \
    config=quant_runs/qwen/qwen3_5_moe_gptq_w8a8.config.yaml
```

- config 用 YAML、**snake_case** 字段（如 `model_path`）；脚本用到的键须写全（无代码内置默认），`model_path` / `save_dir` 必填，脚本特有键（如 `smoothing_strength`）也一并写入。
- CLI **只接受唯一键** `config=<path>` 指定 YAML 文件路径；其余 `key=value` 会被忽略——所有设置改写进 YAML。
- 全部参数键见上文「四、common/ 公共模块」，或直接参照同名 `*.config.yaml` 现有字段。

---

## 六、设计说明

- **公共逻辑抽到 `common/`**：配置加载（OmegaConf 读 YAML）、辅助文件拷贝在所有脚本间一致，脚本本体只保留各自的模型加载、校准数据、recipe 等差异部分。全部参数写在每个脚本的 YAML 里，无代码内置默认。
- **MoE 模型** 必须用 `load_context(<ModelClass>)` 包裹 `from_pretrained`，并在 `oneshot` 传 `moe_calibrate_all_experts=True`，否则 3D 专家张量无法线性化为可量化的 2D Linear；MTP 层不经 `*ForConditionalGeneration` 加载，需用 `save_mtp_tensors_to_checkpoint` 从原 checkpoint 原样拷入输出目录。
- **产物输出**：脚本默认把量化结果写到输入目录同级（由 `save_dir` 控制）；量化产物本身不纳入版本控制。

---

## 七、Git 工作流（版本节点集成 / 基线跟随）

本仓库 fork 自上游，采用「基线跟随」策略：`main` 与上游 **100% 纯净同步**（严禁在 `main` 写业务代码）；日常开发在基于某个稳定 tag 创建的分支上（本仓库即 `x0.12.0`，基于上游 `0.12.0` tag）。上游发新版时**不把零碎更新持续 merge 进开发分支**，而是基于新 tag 建新分支、把旧分支业务代码整体迁移过去——每次升级都站在最新稳定基线上，避免长期积累冲突。

### 从 0.12.0 升级到 0.13.0

```bash
# 1) 同步上游并拉新 tag
git checkout main
git fetch upstream && git merge upstream/main && git push origin main
git fetch --all --tags

# 2) 基于新 tag 建开发分支并推送
git checkout -b x0.13.0 0.13.0
git push origin x0.13.0

# 3) 把旧分支业务代码合过来（大概率有冲突）
git merge x0.12.0
#    解决冲突：保留 0.13.0 上游新逻辑 + 自己的业务逻辑，删冲突标记后：
git add . && git commit -m "chore: merge business logic from x0.12.0 onto 0.13.0 base"
git push origin x0.13.0
```

随后到 GitHub **Settings → Default branch** 把默认分支切到 `x0.13.0`。

> **精准迁移**：若旧分支夹带临时/废弃提交，改用 `git cherry-pick`，只挑核心业务提交：
> `git log x0.12.0 --oneline` 找 hash → `git checkout x0.13.0` → `git cherry-pick <hash1> <hash2> …`；冲突时 `git add . && git cherry-pick --continue`（放弃用 `--abort`）。

### 注意事项

- **保留旧分支**：迁移后不要删 `x0.12.0`，留作历史快照 / 旧版修复。
- **更新分支保护**：默认分支变更后，到 `Settings → Rulesets` 把保护规则目标改到新分支（或设为 `Default`）。
- **检查 CI/CD**：`.github/workflows/` 若写死只在某分支触发，需相应更新。
- **保持 main 纯净**：`main` 仅用于同步上游，不在其上提交业务代码。
