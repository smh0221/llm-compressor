# 环境构建 / 安装说明

本仓库（fork 自 vllm-project/llm-compressor）在 `smh_llmc` conda env 下使用。
由于采用 `pip install -e . --no-deps` 安装，**setup.py 声明的运行依赖不会被自动安装**，需先用 `quant_runs/requirements.txt` 补齐。

## 安装步骤

```bash
# 1. 安装运行依赖（quant_runs/ 脚本 + llmcompressor 库所需）
pip install -r quant_runs/requirements.txt

# 2. 以可编辑模式安装本地包（不带依赖，依赖由上一步提供）
pip install -e . --no-deps
```

## 说明

- 依赖清单与 pin 版本见 `quant_runs/requirements.txt`，版本对齐 `smh_llmc` env 实测可用值（transformers 5.x 线）。
- `requirements.txt` 已覆盖：
  - llmcompressor core（对应 setup.py `install_requires`）；
  - 图文校准所需的 `qwen-vl-utils` + `torchvision`（setup.py 的 `qwen` extra 仅含 `qwen_vl_utils`，但实跑图文校准还需 torchvision）。
- transformers **必须 >= 5**：脚本使用的 `Qwen3_5MoeForConditionalGeneration` 仅在 5.x 提供。
- torch / torchvision 在 `requirements.txt` 中去掉了 `+cuXXX` 本地构建标签以保证可移植；如需指定 CUDA wheel：
  ```bash
  pip install -r quant_runs/requirements.txt \
      --index-url https://download.pytorch.org/whl/cu126
  ```
