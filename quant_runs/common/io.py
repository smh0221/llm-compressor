import os
import shutil
from omegaconf import OmegaConf


def load_config():
    config_path = OmegaConf.from_cli().pop("config", None)
    if config_path is None:
        raise SystemExit("config=<path.yaml> is required on the CLI.")

    cfg = OmegaConf.load(config_path)

    necessary_keys = ["model_path", "save_dir"]
    missing = [k for k in necessary_keys if cfg.get(k) is None]
    if missing:
        raise SystemExit(f"{', '.join(missing)} required in the YAML config ({config_path}).")

    return cfg


def copy_auxiliary_files(src_dir, dest_dir):
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
