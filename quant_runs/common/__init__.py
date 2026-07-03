from quant_runs.common.io import load_config, copy_auxiliary_files
from quant_runs.common.torch_compat import apply_torch_compat

# 导入 common 即打上 torch 兼容垫片，保证脚本调用 from_pretrained 前 API 已补齐
apply_torch_compat()

__all__ = [
    "load_config",
    "copy_auxiliary_files",
]
