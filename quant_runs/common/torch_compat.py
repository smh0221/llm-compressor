"""torch 兼容垫片，为 torch 2.9 定制版本补齐 compressed-tensors 0.17 依赖的新 API。"""

import torch


def apply_torch_compat():
    # compressed-tensors>=0.17 用 torch.accelerator.get_memory_info，该 API 仅 torch>=2.10 提供
    # torch 2.9 定制版本缺此 API，用等价的 torch.cuda.mem_get_info 补齐，二者均返回 (free, total)
    if not hasattr(torch.accelerator, "get_memory_info"):

        def get_memory_info(device=None):
            return torch.cuda.mem_get_info(device)

        torch.accelerator.get_memory_info = get_memory_info
