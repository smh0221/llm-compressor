"""quant_runs/ 脚本的共用辅助。

对外导出：所有量化脚本共用的 CLI/config 处理逻辑（预置通用 flag 的 argparse
parser、JSON-config 合并、save-dir 解析），以及拷贝非权重辅助文件的文件系统辅助。
"""

from quant_runs.common.cli import (
    build_arg_parser,
    merge_config,
    resolve_save_dir,
)
from quant_runs.common.io import copy_auxiliary_files

__all__ = [
    "build_arg_parser",
    "merge_config",
    "resolve_save_dir",
    "copy_auxiliary_files",
]
