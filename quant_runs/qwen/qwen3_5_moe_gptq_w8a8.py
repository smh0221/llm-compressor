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
    copy_auxiliary_files,
    load_config,
)


def parse_args():
    return load_config()


def main():
    cfg = parse_args()

    model_path = cfg.model_path.rstrip("/")
    save_dir = cfg.save_dir
    dataset_split = f"{cfg.dataset_split}[:{cfg.num_calibration_samples}]"

    # 加载模型，MoE 须用 load_context 包裹以线性化专家张量
    with load_context(Qwen3_5MoeForConditionalGeneration):
        model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
            model_path,
            device_map=cfg.device_map,
            dtype="auto",
            local_files_only=True,
        )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    # 加载校准数据集，flickr30k 图文数据集
    ds = load_dataset(cfg.dataset_id, split=dataset_split)
    ds = ds.shuffle(seed=cfg.seed)

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

        # apply_chat_template 已把图像替换为占位 token，text 里只剩占位符、无像素数据；
        # 真实像素由 process_vision_info 从 messages 解码得到，故须通过 images= 单独传入。
        # processor 内部再把 pixel_values 与 text 中的图像占位符按位置对齐。
        return processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            max_length=cfg.max_seq_length,
            truncation=True,
        )

    # 逐样本预处理，remove_columns 删除原始列（image/caption），只保留 processor 产出的字段。
    # map 后各字段为嵌套 list（Arrow 存储），tensor 化在 data_collator 里进行；下列 shape 指其逻辑形状：
    #   input_ids:         (1, seq_len)          文本 token 序列，图像位置为 <|image_pad|> 占位 token
    #   attention_mask:    (1, seq_len)          注意力掩码（padding=False，基本全 1）
    #   mm_token_type_ids: (1, seq_len)          多模态类型标记，逐 token 区分文本/图像位置
    #   pixel_values:      (1, num_patches, patch_dim)  打平的图像 patch（外层 1 来自 text=[text] 的 batch 维）
    #   image_grid_thw:    (1, 3)                每张图的 patch 网格 [T, H, W]
    ds = ds.map(preprocess_and_tokenize, remove_columns=ds.column_names)

    # 多模态 data collator：把 dataset 中一条样本的嵌套 list 转为 tensor 喂给模型。
    # 校准逐样本进行，故 batch_size 固定为 1（assert 保证）。
    def data_collator(batch):
        assert len(batch) == 1
        collated = {}
        for key, value in batch[0].items():
            if key == "pixel_values":
                # pixel_values 因 text=[text] 多带一层 batch 维 (1, num_patches, patch_dim)，
                # 单样本校准下该维冗余，squeeze(0) 压回 (num_patches, patch_dim)；
                # 同时转 bfloat16 以对齐模型（bf16 加载）的视觉塔精度，避免 dtype 不匹配。
                collated[key] = torch.tensor(value, dtype=torch.bfloat16).squeeze(0)
            else:
                # input_ids / attention_mask / mm_token_type_ids 等文本类字段，
                # 保留 (1, seq_len) 的 batch 维（模型 forward 本就期望该维），直接转 tensor。
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
        max_seq_length=cfg.max_seq_length,
        num_calibration_samples=cfg.num_calibration_samples,
        trust_remote_code_model=True,
        data_collator=data_collator,
        moe_calibrate_all_experts=True,
        preprocessing_num_workers=cfg.preprocessing_num_workers,
        dataloader_num_workers=cfg.dataloader_num_workers,
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
