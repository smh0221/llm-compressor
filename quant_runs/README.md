# quant_runs/ — 量化运行脚本库

本目录承载本仓库（fork 自 vllm-project/llm-compressor）上**多模型 × 多量化方法矩阵**的可运行脚本。与上游 `examples/` **完全隔离**，便于跟随上游 rebase 时保持干净。

## 目录约定

- **一级目录 = 模型系列**：`qwen/`、`glm/`、`deepseek/` …
- **二级 = 具体脚本（方法_精度）**：每个脚本**自包含、可单独运行**，暂不抽取公共模块。
- 每个模型系列目录下有一份 `README.md`，以表格形式记录该系列已覆盖的 模型 × 方法 × 精度 矩阵，便于横向对比。

```
quant_runs/
├── README.md                      # 本文件：目录约定、命名规范、运行方式、env、依赖
├── qwen/
│   ├── README.md                  # qwen 系列矩阵
│   └── qwen3_5_moe_gptq_w8a8.py
├── glm/
│   └── README.md                  # 占位：计划覆盖的 glm 模型/方法
└── deepseek/
    └── README.md                  # 占位：计划覆盖的 deepseek 模型/方法
```

## 命名规范

文件名统一为：`<模型名>_<方法>_<精度>.py`（全小写 + 下划线）

| 字段 | 取值示例 |
| --- | --- |
| 模型名 | `qwen3_5_moe`、`qwen3_next`、`glm4_7`、`deepseek_v4` |
| 方法 | `gptq`、`smoothquant`、`spinquant`、`awq`、`rtn`、`gptq_smoothquant`（组合用下划线连） |
| 精度 | `w8a8`、`w4a8`、`w8a16`、`w4a16`、`w4a4`、`w2a16` |

示例：
- `qwen3_5_moe_gptq_w8a8.py`
- `glm4_7_smoothquant_w8a8.py`
- `deepseek_v4_spinquant_w4a16.py`

## 运行环境

- conda env：**`smh_llmc`**
  - 已校验：`transformers==5.12.1`、`qwen_vl_utils` 均可用。
- 依赖要求（按脚本不同）：
  - **transformers >= 5**：MoE 的 `*ForConditionalGeneration` 类（如 `Qwen3_5MoeForConditionalGeneration`）才存在。
  - **qwen_vl_utils**：图文校准脚本需要（`process_vision_info`）。可 `pip install qwen_vl_utils` 或安装 `llmcompressor[qwen]`。

## 运行方式

所有脚本均通过 argparse 暴露**输入/输出路径**等可配置项。典型调用：

```bash
/root/miniconda/bin/conda run -n smh_llmc python quant_runs/qwen/qwen3_5_moe_gptq_w8a8.py \
    --model-path /path/to/Qwen3.5-35B-A3B \
    --save-dir   /path/to/Qwen3.5-35B-A3B-W8A8-gptq
```

每个脚本头部注释会写明：所需 env、依赖、最小运行示例。

## 设计说明

- **先各自独立**：每个脚本自包含，方便单独 review / 复制改写。待脚本增多、重复明显时，再考虑抽取 `quant_runs/common/`。
- **产物输出**：脚本默认把量化结果写到输入目录同级（由 `--save-dir` 控制），**不修改仓库 `.gitignore`**，量化产物是否纳入版本控制由用户自行决定。
