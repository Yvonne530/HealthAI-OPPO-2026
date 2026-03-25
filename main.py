#!/usr/bin/env python3
"""
main.py (v6 - 修复 num_workers 死锁 + epoch 子集 + tqdm 进度条)
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
    fmt = "%(asctime)s %(levelname)s %(name)s - %(message)s"
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(
            os.path.join(log_dir, f"run_{time.strftime('%Y%m%d_%H%M%S')}.log"),
            encoding="utf-8"
        ),
    ]
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ================================================================
# 环境检查
# ================================================================

def run_check(cfg: dict) -> None:
    import torch
    print("\n" + "=" * 55 + "\nRehabGuardian 环境检查\n" + "=" * 55)
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"GPU:  {p.name}  VRAM:{p.total_memory/1e9:.1f}GB")
    try:
        import psutil
        print(f"RAM:  {psutil.virtual_memory().available/1e9:.1f}GB 可用 / {psutil.virtual_memory().total/1e9:.1f}GB 总量")
    except ImportError:
        pass
    h5 = cfg["data"]["h5_path"]
    print(f"HDF5: {'✅ 已存在' if os.path.exists(h5) else '❌ 不存在，请先 python preprocess.py'}")
    if os.path.exists(h5):
        print(f"  大小: {os.path.getsize(h5)/1e9:.2f} GB")
    nw = cfg["train"].get("num_workers", 0)
    if nw > 0:
        print(f"⚠️  num_workers={nw}，WSL2+h5py 可能死锁，建议改为 0")
    else:
        print(f"num_workers: {nw} ✅")
    ratio = cfg["train"].get("epoch_subset_ratio", 1.0)
    print(f"epoch_subset_ratio: {ratio} ({'抽样模式' if ratio<1 else '全量模式'})")
    print("=" * 55)


# ================================================================
# 训练（完整版 + tqdm + EpochSubsetSampler）
# ================================================================

def run_train(cfg: dict) -> None:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader

    try:
        from tqdm import tqdm
        HAS_TQDM = True
    except ImportError:
        HAS_TQDM = False
        logger.warning("tqdm 未安装，无进度条（pip install tqdm）")

    from models.stgcn          import STGCN
    from models.fno            import FNO1d
    from models.risk_model     import RiskMLP
    from utils.normalizer      import Normalizer
    from data.hdf5_dataset     import HDF5RehabDataset, EpochSubsetSampler
    from trainers.train_pipeline import (
        EarlyStopping, e2e_forward, warmup_cosine_scheduler, _create_opt_and_sched
    )
    from utils.reproducibility import StructuredLogger, verify_dataset_version

    h5_path   = cfg["data"]["h5_path"]
    meta_path = cfg["data"]["meta_path"]

    if not os.path.exists(h5_path):
        logger.error(f"HDF5 不存在: {h5_path}\n请先运行: python preprocess.py")
        sys.exit(1)

    verify_dataset_version(
        os.path.join(cfg["data"]["processed_dir"], "data_version_lock.json"),
        [h5_path]
    )

    future_k   = cfg["fno"].get("future_k", 10)
    n_workers  = cfg["train"].get("num_workers", 0)
    pin_memory = cfg["train"].get("pin_memory", True) and torch.cuda.is_available()
    subset_r   = cfg["train"].get("epoch_subset_ratio", 1.0)

    if n_workers > 0:
        logger.warning(f"num_workers={n_workers} 在 WSL2+h5py 可能死锁，强制改为 0")
        n_workers = 0

    norm = Normalizer()
    logger.info("初始化 Dataset（流式读取，不全量加载）...")

    ds_train = HDF5RehabDataset(
        h5_path, meta_path, "train", norm,
        seq_len=cfg["data"]["sequence_length"],
        future_k=future_k,
        vis_len=cfg["stgcn"]["num_frames"],
        augment=True,
    )
    ds_val = HDF5RehabDataset(
        h5_path, meta_path, "val", norm,
        seq_len=cfg["data"]["sequence_length"],
        future_k=future_k,
        vis_len=cfg["stgcn"]["num_frames"],
        augment=False,
    )

    os.makedirs(cfg["data"]["processed_dir"], exist_ok=True)
    norm.save(os.path.join(cfg["data"]["processed_dir"], "norm_stats.npz"))

    bs = cfg["train"]["batch_size"]

    # EpochSubsetSampler：每 epoch 随机抽 subset_r 比例
    train_sampler = EpochSubsetSampler(ds_train, ratio=subset_r,
                                       seed=cfg["project"]["seed"])
    val_sampler   = EpochSubsetSampler(ds_val, ratio=min(1.0, subset_r * 2),
                                       seed=cfg["project"]["seed"])

    dl_train = DataLoader(ds_train, batch_size=bs, sampler=train_sampler,
                          num_workers=0, pin_memory=pin_memory, drop_last=True)
    dl_val   = DataLoader(ds_val,   batch_size=bs, sampler=val_sampler,
                          num_workers=0, pin_memory=pin_memory)

    n_train_w = len(train_sampler)
    n_val_w   = len(val_sampler)
    n_train_b = n_train_w // bs
    n_val_b   = n_val_w   // bs

    logger.info(
        f"训练集: {len(ds_train):,} 总窗口 → 每epoch {n_train_w:,} 窗口 ({subset_r*100:.0f}%)"
        f" = {n_train_b} batch"
    )
    logger.info(f"batch_size={bs} num_workers=0 pin_memory={pin_memory}")
    logger.info(f"预估每epoch时间: {n_train_b * 0.025 / 60:.1f}~{n_train_b * 0.06 / 60:.1f} 分钟")

    # 构建模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stgcn = STGCN(
        num_nodes=cfg["stgcn"]["num_nodes"],
        in_channels=cfg["stgcn"]["in_channels"],
        hidden_channels=cfg["stgcn"]["hidden_channels"],
        num_frames=cfg["stgcn"]["num_frames"],
        output_dim=cfg["stgcn"]["output_dim"],
        dropout=cfg["stgcn"]["dropout"],
    ).to(device)
    fno = FNO1d(
        input_dim=cfg["fno"]["input_dim"],
        output_dim=cfg["fno"]["output_dim"],
        future_k=future_k,
        modes=cfg["fno"]["modes"],
        width=cfg["fno"]["width"],
        depth=cfg["fno"]["depth"],
        seq_len=cfg["fno"]["seq_len"],
    ).to(device)
    risk = RiskMLP(
        input_dim=cfg["risk"]["input_dim"],
        hidden_dim=cfg["risk"]["hidden_dim"],
        num_classes=cfg["risk"]["num_classes"],
        seq_len=cfg["risk"].get("seq_len", 20),
        use_physio=cfg["risk"].get("use_physio", True),
    ).to(device)

    logger.info(f"ST-GCN: {stgcn.count_params()/1e6:.2f}M  FNO: {fno.count_params()/1e6:.2f}M  "
                f"Risk: {risk.count_params():,}  lag={fno.current_lag_ms:.0f}ms")

    n_epochs   = cfg["train"]["num_epochs"]
    lr, wd     = cfg["train"]["learning_rate"], cfg["train"]["weight_decay"]
    warmup_ep  = cfg.get("scheduled_sampling", {}).get("warmup_epochs", 5)
    st_cfg     = cfg.get("stage_training", {})
    stage1_end = int(n_epochs * st_cfg.get("stage1_ratio", 0.30))
    stage2_end = int(n_epochs * st_cfg.get("stage2_ratio", 0.60))

    opt_s1, sc_s1 = _create_opt_and_sched(stgcn.parameters(), lr, wd, n_epochs, warmup_ep)
    opt_s2, sc_s2 = _create_opt_and_sched(
        list(fno.parameters()) + list(risk.parameters()), lr, wd, n_epochs, warmup_ep)
    opt_s3, sc_s3 = _create_opt_and_sched(
        list(stgcn.parameters()) + list(fno.parameters()) + list(risk.parameters()),
        lr * 0.1, wd, n_epochs, warmup_ep)

    w     = cfg["train"]["loss_weights"]
    early = EarlyStopping(patience=cfg["train"]["patience"])

    slog = StructuredLogger(
        os.path.join(cfg["train"]["log_dir"],
                     f"run_{time.strftime('%Y%m%d_%H%M%S')}.json")
    )
    import json as _json
    with open(meta_path) as f:
        meta_info = _json.load(f)
    slog.start(
        config=cfg,
        data_version={"fingerprint": meta_info.get("fingerprint",""),
                      "frames": meta_info.get("total_frames",0),
                      "subset_ratio": subset_r},
        model_params={"stgcn": stgcn.count_params(), "fno": fno.count_params(),
                      "risk": risk.count_params()},
    )

    os.makedirs(cfg["train"]["checkpoint_dir"], exist_ok=True)
    best_val = float("inf")
    history  = {"train_loss":[], "val_loss":[], "val_grf_mae":[],
                "val_risk_acc":[], "learned_lag_ms":[]}

    for epoch in range(1, n_epochs + 1):
        t0      = time.time()
        stage   = 1 if epoch <= stage1_end else (2 if epoch <= stage2_end else 3)
        sched_p = max(0.3, 1.0 - 0.7*(epoch-1)/max(n_epochs-1,1))

        if   stage == 1: opt, sched = opt_s1, sc_s1
        elif stage == 2: opt, sched = opt_s2, sc_s2
        else:            opt, sched = opt_s3, sc_s3

        for p in stgcn.parameters(): p.requires_grad_(stage in (1,3))
        for p in fno.parameters():   p.requires_grad_(stage in (2,3))
        for p in risk.parameters():  p.requires_grad_(stage in (2,3))

        # 每 epoch 用不同随机子集
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch + 10000)

        # ---- Train ----
        stgcn.train(); fno.train(); risk.train()
        tr_loss = 0.0; n_steps = 0

        it = tqdm(dl_train, desc=f"Ep{epoch:03d}[S{stage}]",
                  leave=False, dynamic_ncols=True) if HAS_TQDM else dl_train

        for batch in it:
            opt.zero_grad()
            batch_a = {k: batch[k] for k in
                ["pose_seq","ja_gt","markers_gt","grf_future",
                 "grf_mask","risk_label","physics_weight"]}
            _, losses = e2e_forward(
                stgcn, fno, risk, batch_a, device,
                scheduled_p=sched_p, future_k=future_k,
            )
            if stage == 1:
                total = (w.get("joint_angles",0.30)*losses["ja"]
                       + w.get("marker",0.10)*losses["mk"]
                       + 0.05*losses["sym"])
            else:
                total = (w.get("joint_angles",0.30)*losses["ja"]
                       + w.get("marker",0.10)*losses["mk"]
                       + w.get("grf",0.35)*losses["grf"]
                       + w.get("risk",0.20)*losses["risk"]
                       + w.get("conf",0.05)*losses["conf"]
                       + 0.05*losses["sym"]
                       + 0.05*losses["pinn"])
            total.backward()
            nn.utils.clip_grad_norm_(stgcn.parameters(), 1.0)
            nn.utils.clip_grad_norm_(fno.parameters(),   1.0)
            nn.utils.clip_grad_norm_(risk.parameters(),  1.0)
            opt.step()
            tr_loss += total.item(); n_steps += 1

            if HAS_TQDM:
                it.set_postfix(loss=f"{total.item():.4f}")

        tr_loss /= max(n_steps, 1)

        # ---- Validation ----
        stgcn.eval(); fno.eval(); risk.eval()
        val_loss = 0.0; grf_mae = 0.0; rc = 0; nv = 0

        with torch.no_grad():
            for batch in dl_val:
                batch_a = {k: batch[k] for k in
                    ["pose_seq","ja_gt","markers_gt","grf_future",
                     "grf_mask","risk_label","physics_weight"]}
                preds, losses = e2e_forward(
                    stgcn, fno, risk, batch_a, device,
                    scheduled_p=0.0, future_k=future_k,
                )
                v = (w.get("joint_angles",0.30)*losses["ja"]
                   + w.get("grf",0.35)*losses["grf"]
                   + w.get("risk",0.20)*losses["risk"])
                val_loss += v.item()
                grf_mae  += (preds["pred_grf"] -
                             batch["grf_future"].to(device)).abs().mean().item()
                rc += (preds["logits"].argmax(-1) ==
                       batch["risk_label"].to(device)).sum().item()
                nv += 1

        val_loss /= max(nv, 1)
        grf_mae  /= max(nv, 1)
        risk_acc  = rc / max(nv * bs, 1)
        lag_ms    = fno.current_lag_ms
        elapsed   = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_grf_mae"].append(grf_mae)
        history["val_risk_acc"].append(risk_acc)
        history["learned_lag_ms"].append(lag_ms)
        slog.log_epoch(tr_loss, val_loss, grf_mae, risk_acc, lag_ms)

        logger.info(
            f"Ep {epoch:03d}/{n_epochs} [S{stage}] | "
            f"tr={tr_loss:.4f} val={val_loss:.4f} "
            f"grf_mae={grf_mae:.4f} risk_acc={risk_acc:.3f} "
            f"lag={lag_ms:.1f}ms | {elapsed:.1f}s"
        )
        sched.step()

        if val_loss < best_val:
            best_val = val_loss
            for m, nm in [(stgcn,"stgcn_best.pth"),
                          (fno,  "fno_best.pth"),
                          (risk, "risk_best.pth")]:
                torch.save(m.state_dict(),
                           os.path.join(cfg["train"]["checkpoint_dir"], nm))
            logger.info(f"  ✅ 最佳模型保存 val={best_val:.4f}")

        if early(val_loss):
            logger.info("早停触发")
            break

    slog.finish(early_stopped=early.stop)
    logger.info(
        f"训练完成 | best_val={best_val:.4f} "
        f"最终时延={fno.current_lag_ms:.1f}ms"
    )


# ================================================================
# 推理
# ================================================================

def run_infer(cfg: dict) -> None:
    from inference.inference import RehabGuardianInference
    from inference.heytap_health_adapter import HeytapHealthAdapter
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    engine  = RehabGuardianInference(cfg, device=device)
    adapter = HeytapHealthAdapter(mock=True)
    engine.update_health_data(adapter.get_heart_rate(), adapter.get_sleep_score())

    result = None; lats = []
    for i in range(60):
        skeleton = np.random.randn(33,3).astype(np.float32)
        result   = engine.run(skeleton, mode="pose")
        lats.append(result["latency_ms"])
        if i % 20 == 0:
            logger.info(f"帧{i}: {result['risk_text']} {result['latency_ms']:.1f}ms")

    if lats:
        logger.info(f"延迟: mean={np.mean(lats):.1f}ms p95={np.percentile(lats,95):.1f}ms")
    if result:
        adapter.write_risk_analysis(
            int(time.time()*1000), 1000,
            result["risk_score"], result["risk_label"]
        )


def run_export(cfg: dict) -> None:
    from export.export_onnx import export_all
    export_all(cfg)


# ================================================================
# 入口
# ================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",   default="train",
                        choices=["preprocess","train","infer","export","check"])
    parser.add_argument("--config", default=os.path.join(ROOT,"configs/config.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg)

    np.random.seed(cfg["project"]["seed"])
    try:
        import torch
        torch.manual_seed(cfg["project"]["seed"])
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True   # 固定模型结构时可加速
    except ImportError:
        pass

    logger.info("=" * 55)
    logger.info(f"RehabGuardian {cfg['project']['version']}  模式: {args.mode}")
    logger.info("=" * 55)

    if   args.mode == "check":      run_check(cfg)
    elif args.mode == "preprocess": logger.info("请运行: python preprocess.py")
    elif args.mode == "train":      run_train(cfg)
    elif args.mode == "infer":      run_infer(cfg)
    elif args.mode == "export":     run_export(cfg)

    logger.info("任务完成 ✅")


if __name__ == "__main__":
    main()