#!/usr/bin/env python3
"""
main.py  (v5 - 适配 7.8GB RAM + RTX 4070 Laptop)
改动：
  - 全量 get_all_samples() → HDF5 流式读取
  - train 不再把样本堆内存，改用 HDF5Dataset
  - 新增 --mode check：环境检查，不跑训练
"""
import argparse
import logging
import os
import sys
import time

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

logger = logging.getLogger("main")


def setup_logging(cfg: dict) -> None:
    log_dir = cfg["train"]["log_dir"]
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                os.path.join(log_dir, f"run_{time.strftime('%Y%m%d_%H%M%S')}.log"),
                encoding="utf-8"
            ),
        ],
    )


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# =====================================================================
# 环境检查（--mode check）
# =====================================================================

def run_check(cfg: dict) -> None:
    """验证运行环境，打印硬件报告"""
    import torch

    print("\n" + "=" * 60)
    print("RehabGuardian 环境检查")
    print("=" * 60)

    # GPU
    cuda_ok = torch.cuda.is_available()
    print(f"CUDA 可用:     {cuda_ok}")
    if cuda_ok:
        props = torch.cuda.get_device_properties(0)
        vram  = props.total_memory / 1e9
        print(f"GPU:           {props.name}")
        print(f"VRAM:          {vram:.1f} GB")
        if vram < 6:
            print("⚠️  VRAM 不足 6GB，batch_size 请设为 8")
        elif vram < 8:
            print("✅  VRAM 充足，batch_size=16 稳定运行")

    # RAM
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / 1e9
        ram_avail = psutil.virtual_memory().available / 1e9
        print(f"RAM 总量:      {ram_gb:.1f} GB")
        print(f"RAM 可用:      {ram_avail:.1f} GB")
        if ram_gb < 8:
            print("⚠️  WSL2 RAM 较少，已切换到 HDF5 流式模式（无需全量加载）")
            print("   建议：在 C:\\Users\\Lenovo\\.wslconfig 中添加:")
            print("   [wsl2]")
            print("   memory=14GB")
            print("   swap=8GB")
    except ImportError:
        pass

    # 磁盘
    h5_path = cfg["data"]["h5_path"]
    h5_dir  = os.path.dirname(h5_path)
    try:
        import shutil
        total, used, free = shutil.disk_usage(h5_dir if os.path.exists(h5_dir)
                                              else os.path.dirname(h5_dir))
        print(f"磁盘可用:      {free/1e9:.1f} GB  ({h5_dir})")
        if free < 3e9:
            print("⚠️  磁盘空间不足，HDF5 约需 2GB")
    except Exception:
        pass

    # HDF5 文件
    if os.path.exists(h5_path):
        size_gb = os.path.getsize(h5_path) / 1e9
        print(f"HDF5 文件:     ✅ 已存在 ({size_gb:.2f} GB)")
    else:
        print(f"HDF5 文件:     ❌ 不存在，请先运行: python preprocess.py")

    # 关键包
    for pkg in ["torch", "h5py", "nimblephysics", "scipy", "numpy"]:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "?")
            print(f"{pkg:<18} {ver}")
        except ImportError:
            print(f"{pkg:<18} ❌ 未安装")

    # nimblephysics 专项
    try:
        import nimblephysics as nimble
        print("nimblephysics: ✅")
    except Exception as e:
        print(f"nimblephysics: ❌ {e}")

    print("=" * 60)


# =====================================================================
# 训练（HDF5 版本）
# =====================================================================

def run_train(cfg: dict) -> None:
    import torch
    from torch.utils.data import DataLoader
    from models.stgcn      import STGCN
    from models.fno        import FNO1d
    from models.risk_model import RiskMLP
    from utils.normalizer  import Normalizer
    from data.hdf5_dataset import HDF5RehabDataset, build_datasets
    from trainers.train_pipeline import RehabGuardianTrainer
    from utils.reproducibility   import StructuredLogger, verify_dataset_version

    h5_path   = cfg["data"]["h5_path"]
    meta_path = cfg["data"]["meta_path"]
    lock_path = os.path.join(cfg["data"]["processed_dir"], "data_version_lock.json")

    # 验证 HDF5
    if not os.path.exists(h5_path):
        logger.error(f"HDF5 文件不存在: {h5_path}")
        logger.error("请先运行: python preprocess.py")
        sys.exit(1)

    # 数据版本验证
    verify_dataset_version(lock_path, [h5_path])

    future_k = cfg["fno"].get("future_k", 10)

    # Normalizer（从 HDF5 计算，不需要全量加载）
    norm = Normalizer()

    # Dataset（HDF5 流式，fit 时采样 10 万帧）
    logger.info("初始化 Dataset（流式读取，不全量加载）...")
    ds_train = HDF5RehabDataset(
        h5_path, meta_path, "train", norm,
        seq_len=cfg["data"]["sequence_length"],
        future_k=future_k,
        vis_len=cfg["stgcn"]["num_frames"],
        augment=True,
    )   # norm 在此 fit
    ds_val = HDF5RehabDataset(
        h5_path, meta_path, "val", norm,
        seq_len=cfg["data"]["sequence_length"],
        future_k=future_k,
        vis_len=cfg["stgcn"]["num_frames"],
        augment=False,
    )

    # 保存归一化参数
    os.makedirs(cfg["data"]["processed_dir"], exist_ok=True)
    norm.save(os.path.join(cfg["data"]["processed_dir"], "norm_stats.npz"))

    bs          = cfg["train"]["batch_size"]
    n_workers   = cfg["train"].get("num_workers", 0)
    pin_memory  = cfg["train"].get("pin_memory", True) and torch.cuda.is_available()

    dl_train = DataLoader(
        ds_train, batch_size=bs, shuffle=True,
        num_workers=n_workers, pin_memory=pin_memory,
        persistent_workers=(n_workers > 0), drop_last=True,
    )
    dl_val = DataLoader(
        ds_val, batch_size=bs, shuffle=False,
        num_workers=n_workers, pin_memory=pin_memory,
        persistent_workers=(n_workers > 0),
    )

    logger.info(f"训练集: {len(ds_train):,} 窗口 | 验证集: {len(ds_val):,} 窗口")
    logger.info(f"batch_size={bs} | num_workers={n_workers} | pin_memory={pin_memory}")

    # 构建模型
    stgcn = STGCN(
        num_nodes       = cfg["stgcn"]["num_nodes"],
        in_channels     = cfg["stgcn"]["in_channels"],
        hidden_channels = cfg["stgcn"]["hidden_channels"],
        num_frames      = cfg["stgcn"]["num_frames"],
        output_dim      = cfg["stgcn"]["output_dim"],
        dropout         = cfg["stgcn"]["dropout"],
    )
    fno = FNO1d(
        input_dim  = cfg["fno"]["input_dim"],
        output_dim = cfg["fno"]["output_dim"],
        future_k   = future_k,
        modes      = cfg["fno"]["modes"],
        width      = cfg["fno"]["width"],
        depth      = cfg["fno"]["depth"],
        seq_len    = cfg["fno"]["seq_len"],
    )
    risk = RiskMLP(
        input_dim   = cfg["risk"]["input_dim"],
        hidden_dim  = cfg["risk"]["hidden_dim"],
        num_classes = cfg["risk"]["num_classes"],
        seq_len     = cfg["risk"].get("seq_len", 20),
        use_physio  = cfg["risk"].get("use_physio", True),
    )

    logger.info(f"ST-GCN: {stgcn.count_params()/1e6:.2f}M 参数")
    logger.info(f"FNO:    {fno.count_params()/1e6:.3f}M 参数")
    logger.info(f"Risk:   {risk.count_params():,} 参数")
    logger.info(f"FNO 初始时延: {fno.current_lag_ms:.1f}ms")

    # 结构化日志
    slog = StructuredLogger(
        os.path.join(cfg["train"]["log_dir"],
                     f"run_{time.strftime('%Y%m%d_%H%M%S')}.json")
    )
    import json
    with open(meta_path) as f:
        meta = json.load(f)
    slog.start(
        config=cfg,
        data_version={"fingerprint": meta.get("fingerprint",""), "frames": meta.get("total_frames",0)},
        model_params={"stgcn": stgcn.count_params(), "fno": fno.count_params(), "risk": risk.count_params()},
    )

    # 训练（使用已有 DataLoader，而非内部重建）
    trainer = RehabGuardianTrainer(cfg)
    history = trainer.train_with_loaders(
        stgcn_model = stgcn,
        fno_model   = fno,
        risk_model  = risk,
        dl_train    = dl_train,
        dl_val      = dl_val,
        slog        = slog,
        future_k    = future_k,
    )

    slog.finish(early_stopped=history.get("early_stopped", False))
    final_lag = history["learned_lag_ms"][-1] if history["learned_lag_ms"] else 0
    logger.info(
        f"训练完成 | best_val={min(history['val_loss']):.4f} | "
        f"学习时延={final_lag:.1f}ms"
    )


# =====================================================================
# 推理
# =====================================================================

def run_infer(cfg: dict) -> None:
    from inference.inference import RehabGuardianInference
    from inference.heytap_health_adapter import HeytapHealthAdapter

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    engine  = RehabGuardianInference(cfg, device=device)
    adapter = HeytapHealthAdapter(mock=True)

    engine.update_health_data(adapter.get_heart_rate(), adapter.get_sleep_score())

    result = None
    latencies = []
    for i in range(60):
        skeleton = np.random.randn(33, 3).astype(np.float32)
        result   = engine.run(skeleton, mode="pose")
        latencies.append(result["latency_ms"])
        if i % 20 == 0:
            logger.info(
                f"帧{i:02d}: {result['risk_text']} "
                f"{result['latency_ms']:.1f}ms "
                f"conf={result.get('confidence',0):.2f} "
                f"state={result.get('risk_label','?')}"
            )

    if latencies:
        logger.info(
            f"延迟统计: mean={np.mean(latencies):.1f}ms "
            f"p95={np.percentile(latencies,95):.1f}ms"
        )

    if result is not None:
        adapter.write_risk_analysis(
            start_time_ms=int(time.time() * 1000),
            duration_ms=int(60 / 60 * 1000),
            risk_score=result["risk_score"],
            risk_level=result["risk_label"],
        )


# =====================================================================
# 导出
# =====================================================================

def run_export(cfg: dict) -> None:
    from export.export_onnx import export_all
    export_all(cfg)


# =====================================================================
# 入口
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="RehabGuardian 10.0")
    parser.add_argument("--mode",   default="train",
                        choices=["preprocess", "train", "infer", "export", "check"])
    parser.add_argument("--config", default=os.path.join(ROOT, "configs/config.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg)

    np.random.seed(cfg["project"]["seed"])
    try:
        import torch
        torch.manual_seed(cfg["project"]["seed"])
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark     = False
    except ImportError:
        pass

    logger.info("=" * 60)
    logger.info(f"RehabGuardian {cfg['project']['version']}  模式: {args.mode}")
    logger.info("=" * 60)

    if   args.mode == "check":
        run_check(cfg)
    elif args.mode == "preprocess":
        logger.info("请直接运行: python preprocess.py")
        logger.info("（支持更多参数，如 --skip_markers）")
    elif args.mode == "train":
        run_train(cfg)
    elif args.mode == "infer":
        run_infer(cfg)
    elif args.mode == "export":
        run_export(cfg)

    logger.info("任务完成 ✅")


if __name__ == "__main__":
    main()