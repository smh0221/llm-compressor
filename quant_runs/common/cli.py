"""quant_runs 脚本共用的 CLI / JSON-config 处理逻辑。

每个量化脚本都接受同一组「通用」flag（模型/输出路径、校准参数、数据集、
device map），外加各自的特有 flag（如 ``--smoothing-strength``）。本模块抽出共用部分：

  - ``build_arg_parser(description)`` —— 预置通用 flag（含 ``--config``）的
    ``ArgumentParser``；脚本在其上追加特有 flag 后再 ``parse_args()``。
  - ``merge_config(parser, args, defaults)`` —— 把 JSON config（kebab-case 键）
    合并进已解析的 args，优先级：显式 CLI flag > config 文件值 > 内置默认。
  - ``resolve_save_dir(model_path, save_dir, default_suffix)`` —— 计算输出目录，
    默认为输入目录同级的 ``<model_name>-<default_suffix>``。

所有 flag 默认值都是 ``None``，便于 ``merge_config`` 区分「用户没传」和真实取值，
只对未传的那些套用 config/内置默认。
"""

import argparse
import json
import os


# 通用 flag，默认值刻意设为 None，使 merge_config 能区分「未设置」与显式取值。
def build_arg_parser(description):
    """构建预置通用 flag 的 ArgumentParser。

    脚本调用本函数，在返回的 parser 上追加特有 flag，再依次
    ``parser.parse_args()`` 和 ``merge_config(...)``。
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "JSON config 文件路径。键名与 CLI flag 同名但去掉前缀 '--'"
            "（如 'model-path'、'save-dir'、'num-calibration-samples'）。"
            "显式 CLI flag 覆盖 config 值。"
        ),
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="输入模型目录（本地权重）。未在 --config 给出时必填。",
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help="输出目录。默认取脚本特定名称，放在输入目录同级。",
    )
    parser.add_argument(
        "--num-calibration-samples",
        type=int,
        default=None,
        help="校准样本数。(默认: 32)",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=None,
        help="校准序列最大长度。(默认: 8192)",
    )
    parser.add_argument(
        "--dataset-id",
        default=None,
        help="HuggingFace 数据集 id。(默认: lmms-lab/flickr30k)",
    )
    parser.add_argument(
        "--dataset-split",
        default=None,
        help="数据集 split 表达式。(默认: 'test[:<num-calibration-samples>]')",
    )
    parser.add_argument(
        "--device-map",
        default=None,
        help="传给 from_pretrained 的 device_map（如 'auto'）。(默认: None)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="校准集 shuffle 的随机种子。(默认: 42)",
    )
    parser.add_argument(
        "--preprocessing-num-workers",
        type=int,
        default=None,
        help="数据集预处理（map）的 worker 数。(默认: 2)",
    )
    parser.add_argument(
        "--dataloader-num-workers",
        type=int,
        default=None,
        help="校准 dataloader 的 worker 数。(默认: 2)",
    )
    return parser


# 所有脚本共用的内置默认值。脚本调用 merge_config 时可用自己的键扩展
# （如 'smoothing_strength'）。
COMMON_DEFAULTS = {
    "num_calibration_samples": 32,
    "max_seq_length": 8192,
    "dataset_id": "lmms-lab/flickr30k",
    "seed": 42,
    "preprocessing_num_workers": 2,
    "dataloader_num_workers": 2,
}


def merge_config(parser, args, defaults=None):
    """把 JSON config（kebab-case 键）合并进已解析的 CLI args。

    优先级：显式 CLI flag > config 文件值 > 内置默认。

    ``defaults`` 让脚本为自己的 flag 追加/覆盖 COMMON_DEFAULTS；合并后的默认表为
    COMMON_DEFAULTS 用 ``defaults`` 更新的结果。要求 ``--model-path`` 可解析
    （经 CLI 或 config），否则经 ``parser.error`` 报错退出。
    """
    merged_defaults = dict(COMMON_DEFAULTS)
    if defaults:
        merged_defaults.update(defaults)

    config = {}
    if args.config is not None:
        with open(args.config) as f:
            raw = json.load(f)
        # 把 kebab-case 键（'model-path'）规范成 argparse dest（'model_path'）。
        # 未知键直接报错，尽早发现拼写错误。
        valid_dests = {a.dest for a in parser._actions}
        for key, value in raw.items():
            dest = key.replace("-", "_")
            if dest not in valid_dests:
                raise ValueError(
                    f"Unknown config key '{key}' in {args.config}. "
                    f"Valid keys: {sorted(valid_dests - {'help', 'config'})}"
                )
            config[dest] = value

    # 每个选项：CLI flag 未传（仍为 None）时，依次回退到 config 值、内置默认。
    for dest, value in config.items():
        if getattr(args, dest, None) is None:
            setattr(args, dest, value)
    for dest, default in merged_defaults.items():
        if getattr(args, dest, None) is None:
            setattr(args, dest, default)

    if args.model_path is None:
        parser.error(
            "--model-path is required (provide it via CLI or in --config)."
        )

    return args


def resolve_save_dir(model_path, save_dir, default_suffix):
    """解析输出目录。

    给了 ``save_dir`` 则直接返回；否则返回输入目录同级的
    ``<model_name>-<default_suffix>``。
    """
    if save_dir is not None:
        return save_dir
    model_name = os.path.basename(model_path)
    return os.path.join(os.path.dirname(model_path), f"{model_name}-{default_suffix}")
