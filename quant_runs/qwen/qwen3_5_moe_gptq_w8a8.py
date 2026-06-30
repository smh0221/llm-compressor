#!/usr/bin/env python
"""Qwen3.5-35B-A3B（文本 MoE）—— 经 GPTQ 量化到 W8A8 (int8)。

权重为 int8 对称 per-channel 静态量化，激活为 int8 对称 per-token 动态量化，
使用图文（flickr30k）数据校准。
"""

import base64
import os
import sys
from io import BytesIO

import torch
from compressed_tensors.utils import save_mtp_tensors_to_checkpoint
from datasets import load_dataset

# 处理 qwen 视觉输入的可选工具（"qwen" extra，可 `pip install qwen_vl_utils`）
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3_5MoeForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.utils import load_context

# 把仓库根（本文件上两级）加入 sys.path，使任意 CWD/启动方式下都能 import quant_runs.common
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from quant_runs.common import (
    build_arg_parser,
    copy_auxiliary_files,
    merge_config,
    resolve_save_dir,
)


def parse_args():
    parser = build_arg_parser(
        description=(
            "Quantize Qwen3.5 MoE to W8A8 (int8) via GPTQ with image-text "
            "(flickr30k) calibration."
        ),
    )
    args = parser.parse_args()
    return merge_config(parser, args)


def main():
    args = parse_args()

    model_path = args.model_path.rstrip("/")
    save_dir = resolve_save_dir(model_path, args.save_dir, "W8A8-gptq")

    dataset_split = args.dataset_split
    if dataset_split is None:
        dataset_split = f"test[:{args.num_calibration_samples}]"

    # 加载模型，MoE 须用 load_context 包裹以线性化专家张量
    with load_context(Qwen3_5MoeForConditionalGeneration):
        model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
            model_path,
            device_map=args.device_map,
            dtype="auto",
            local_files_only=True,
        )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    # 加载校准数据集，flickr30k 图文数据集
    ds = load_dataset(args.dataset_id, split=dataset_split)
    ds = ds.shuffle(seed=args.seed)

    def preprocess_and_tokenize(example):
        # 预处理，把图像编码成 base64 data URI
        buffered = BytesIO()
        example["image"].save(buffered, format="PNG")
        encoded_image = base64.b64encode(buffered.getvalue())
        encoded_image_text = encoded_image.decode("utf-8")
        base64_qwen = f"data:image;base64,{encoded_image_text}"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": base64_qwen},
                    {"type": "text", "text": "What does the image show?"},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": f"{example['caption'][0]}"},
                ],
            },
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        # 分词
        return processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            max_length=args.max_seq_length,
            truncation=True,
        )

    ds = ds.map(preprocess_and_tokenize, remove_columns=ds.column_names)

    # 多模态 data collator
    def data_collator(batch):
        assert len(batch) == 1
        collated = {}
        for key, value in batch[0].items():
            if key == "pixel_values":
                collated[key] = torch.tensor(value, dtype=torch.bfloat16).squeeze(0)
            else:
                collated[key] = torch.tensor(value)
        return collated

    # recipe 构造
    recipe = [
        GPTQModifier(
            targets="Linear",
            scheme="W8A8",
            offload_hessians=True,
            ignore=[
                "re:.*lm_head",
                "re:visual.*",
                "re:model.visual.*",
                "re:.*mlp.gate$",
                "re:.*embed_tokens$",
                "re:.*shared_expert_gate$",
            ],
        ),
    ]

    # 量化
    oneshot(
        model=model,
        tokenizer=model_path,
        dataset=ds,
        recipe=recipe,
        max_seq_length=args.max_seq_length,
        num_calibration_samples=args.num_calibration_samples,
        trust_remote_code_model=True,
        data_collator=data_collator,
        moe_calibrate_all_experts=True,
        preprocessing_num_workers=args.preprocessing_num_workers,
        dataloader_num_workers=args.dataloader_num_workers,
        sequential_prefetch=True,
    )

    # 保存压缩后权重
    model.save_pretrained(save_dir, save_compressed=True)
    processor.save_pretrained(save_dir)

    # 从源目录拷贝非权重辅助文件，例如tokenizer、各类 config 等
    copy_auxiliary_files(model_path, save_dir)

    # MTP 层不经 Qwen3_5MoeForConditionalGeneration 加载，故从原 checkpoint 拷入到量化输出目录
    save_mtp_tensors_to_checkpoint(source_model=model_path, dest_dir=save_dir)

    print(f"Saved quantized model to: {save_dir}")


if __name__ == "__main__":
    main()
