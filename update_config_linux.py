#!/usr/bin/env python3
"""
update_config_linux.py
将 config.yaml 中的 processed 路径改为 Linux 原生路径
原路径：/mnt/d/.../processed/   (Windows 9p，慢)
新路径：~/data/processed/       (Linux ext4，快)

用法：python update_config_linux.py
"""
import os
import shutil
import yaml
import pathlib

HOME = pathlib.Path.home()
LINUX_PROCESSED = str(HOME / "data" / "processed")
CONFIG_PATH = "configs/config.yaml"

def main():
    if not os.path.exists(CONFIG_PATH):
        print(f"找不到 {CONFIG_PATH}，请在项目根目录运行")
        return

    # 备份
    shutil.copy(CONFIG_PATH, CONFIG_PATH + ".bak")

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    old_processed = cfg["data"].get("processed_dir", "")
    
    # 更新路径（只改 processed 相关，raw_root 保持不变）
    cfg["data"]["processed_dir"] = LINUX_PROCESSED + "/"
    cfg["data"]["h5_path"]       = os.path.join(LINUX_PROCESSED, "dataset.h5")
    cfg["data"]["meta_path"]     = os.path.join(LINUX_PROCESSED, "metadata.json")
    cfg["inference"]["norm_stats"] = os.path.join(LINUX_PROCESSED, "norm_stats.npz")
    cfg["inference"]["ood_stats"]  = os.path.join(LINUX_PROCESSED, "ood_stats.npz")

    with open(CONFIG_PATH, "w") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    print("config.yaml 路径已更新：")
    print(f"  processed_dir : {cfg['data']['processed_dir']}")
    print(f"  h5_path       : {cfg['data']['h5_path']}")
    print(f"  meta_path     : {cfg['data']['meta_path']}")
    print(f"  norm_stats    : {cfg['inference']['norm_stats']}")
    print()
    print(f"raw_root 保持不变（{cfg['data']['raw_root']}）")
    print()

    # 检查文件是否已经复制过去
    h5_exists = os.path.exists(cfg["data"]["h5_path"])
    meta_exists = os.path.exists(cfg["data"]["meta_path"])

    if h5_exists and meta_exists:
        h5_size = os.path.getsize(cfg["data"]["h5_path"]) / 1e9
        print(f"✅ HDF5 文件已存在 ({h5_size:.2f} GB)")
        print("可以直接运行：python main.py --mode train")
    else:
        print("⚠️  HDF5 文件还未复制，请先执行：")
        print(f"  mkdir -p {LINUX_PROCESSED}")
        print(f"  cp {old_processed}dataset.h5 {LINUX_PROCESSED}/")
        print(f"  cp {old_processed}metadata.json {LINUX_PROCESSED}/")
        print(f"  cp {old_processed}norm_stats.npz {LINUX_PROCESSED}/  # 如果存在")
        print()
        print("复制完成后再运行：python main.py --mode train")


if __name__ == "__main__":
    main()