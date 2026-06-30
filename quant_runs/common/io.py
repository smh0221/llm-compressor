"""quant_runs 脚本共用的文件系统辅助。

量化后 ``save_pretrained`` 会写出压缩权重分片及其 index，但大量*非权重*辅助文件
（tokenizer、各类 config、生成参数等）只存在于源 checkpoint。本模块负责把它们拷过去，
且不覆盖刚写好的（量化后）权重。
"""

import os
import shutil


def copy_auxiliary_files(src_dir, dest_dir):
    """把 src_dir 中的非权重辅助文件拷到 dest_dir。

    跳过 safetensors/bin 权重分片及其 index 文件（这些由 save_pretrained /
    save_mtp_tensors_to_checkpoint 产出，不可被原始未量化版本覆盖）。
    dest_dir 中已存在的同名文件保持不动。
    """
    skip_exts = (".safetensors", ".bin")
    skip_names = {
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    }
    os.makedirs(dest_dir, exist_ok=True)
    for name in os.listdir(src_dir):
        src_path = os.path.join(src_dir, name)
        if not os.path.isfile(src_path):
            continue
        if name in skip_names or name.endswith(skip_exts):
            continue
        dest_path = os.path.join(dest_dir, name)
        if os.path.exists(dest_path):
            continue
        shutil.copy2(src_path, dest_path)
