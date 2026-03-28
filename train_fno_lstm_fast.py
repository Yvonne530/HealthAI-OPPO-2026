#!/usr/bin/env python3
"""
train_fno_lstm_fast.py
快速训练 FNO-LSTM 专用脚本（用于 MNN 导出）

策略：
  - 只训练 FNO 分支（STGCN/Risk 冻结）
  - FNO 强制 use_lstm=True
  - 快速收敛参数（25 epoch，较高初始学习率）
  - 预计耗时：3-5 小时
  
用法：
  python train_fno_lstm_fast.py
  python train_fno_lstm_fast.py --epochs 30 --batch 32

输出：
  checkpoints/fno_bestgrf_PhaseA_lstm.pth
"""
import argparse, logging, os, sys, time
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
import yaml
from torch.utils.data import DataLoader, Subset

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.makedirs("logs", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

LOG_FILE = f"logs/train_fno_lstm_{time.strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
logger = logging.getLogger("train_fno_lstm_fast")


def main():
    parser = argparse.ArgumentParser(description="Fast FNO-LSTM training for MNN export")
    parser.add_argument("--config", default=os.path.join(ROOT, "configs/config.yaml"))
    parser.add_argument("--batch", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--epoch_frac", type=float, default=0.15)
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--out_ckpt", default=os.path.join(ROOT, "checkpoints/fno_bestgrf_PhaseA_lstm.pth"))
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg["project"]["seed"])
    np.random.seed(cfg["project"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    logger.info(f"Device: {device} | Log: {LOG_FILE}")

    # ---- Dataset ----
    from utils.normalizer import Normalizer
    from data.hdf5_dataset import HDF5RehabDataset

    linux_h5 = os.path.expanduser("~/data/processed/dataset.h5")
    linux_meta = os.path.expanduser("~/data/processed/metadata.json")

    if os.path.exists(linux_h5):
        h5_path = linux_h5
        meta_path = linux_meta
        logger.info("✅ Using Linux local files (fast)")
    else:
        h5_path = cfg["data"]["h5_path"]
        meta_path = cfg["data"]["meta_path"]
        logger.info("⚠️  Using /mnt/d/ paths (slower)")

    norm = Normalizer()
    if not os.path.exists(h5_path):
        logger.error(f"HDF5 not found: {h5_path}")
        sys.exit(1)

    ds_train = HDF5RehabDataset(
        h5_path, meta_path, "train", norm,
        seq_len=cfg["fno"]["seq_len"],
        future_k=cfg["fno"]["future_k"],
        vis_len=cfg["stgcn"]["num_frames"],
        augment=True,
    )
    ds_val = HDF5RehabDataset(
        h5_path, meta_path, "val", norm,
        seq_len=cfg["fno"]["seq_len"],
        future_k=cfg["fno"]["future_k"],
        vis_len=cfg["stgcn"]["num_frames"],
        augment=False,
    )

    n_tr = len(ds_train)
    n_val = len(ds_val)
    n_per_ep = max(50, int(n_tr * args.epoch_frac))
    n_per_val_ep = max(50, int(n_val * args.val_frac))

    sec_per_batch = 0.08 if os.path.exists(linux_h5) else 0.25
    est_min = (n_per_ep + n_per_val_ep) / args.batch * sec_per_batch / 60

    logger.info(f"Train windows: {n_tr:,} → per epoch sample {n_per_ep:,} ({args.epoch_frac*100:.0f}%)")
    logger.info(f"Val windows: {n_val:,} → per epoch sample {n_per_val_ep:,} ({args.val_frac*100:.0f}%)")
    logger.info(f"Batch={args.batch}, epochs={args.epochs}, lr={args.lr} | Est. per epoch: ~{est_min:.0f} min")

    train_indices = np.random.choice(n_tr, n_per_ep, replace=False)
    val_indices = np.random.choice(n_val, n_per_val_ep, replace=False)
    ds_train_sub = Subset(ds_train, train_indices)
    ds_val_sub = Subset(ds_val, val_indices)

    dl_train = DataLoader(ds_train_sub, batch_size=args.batch, shuffle=True, num_workers=0)
    dl_val = DataLoader(ds_val_sub, batch_size=args.batch, shuffle=False, num_workers=0)

    # ---- Models ----
    from models.stgcn import STGCN
    from models.fno import FNO1d
    from models.risk_model import RiskMLP
    from utils.kinematics import finite_diff, compute_com

    # Load pretrained STGCN and Risk (keep frozen)
    stgcn = STGCN(
        num_nodes=cfg["stgcn"]["num_nodes"],
        in_channels=cfg["stgcn"]["in_channels"],
        hidden_channels=cfg["stgcn"]["hidden_channels"],
        num_frames=cfg["stgcn"]["num_frames"],
        output_dim=cfg["stgcn"]["output_dim"],
        dropout=cfg["stgcn"]["dropout"],
    ).to(device)
    stgcn_ckpt = os.path.join(ROOT, "checkpoints/stgcn_bestgrf_PhaseA.pth")
    if os.path.exists(stgcn_ckpt):
        stgcn.load_state_dict(torch.load(stgcn_ckpt, map_location=device))
        logger.info(f"✅ Loaded STGCN from {stgcn_ckpt}")
    stgcn.eval()
    for p in stgcn.parameters():
        p.requires_grad = False

    # FNO with LSTM (trainable)
    fno = FNO1d(
        input_dim=cfg["fno"]["input_dim"],
        output_dim=cfg["fno"]["output_dim"],
        future_k=cfg["fno"]["future_k"],
        modes=cfg["fno"]["modes"],
        width=cfg["fno"]["width"],
        depth=cfg["fno"]["depth"],
        seq_len=cfg["fno"]["seq_len"],
        use_lstm=True,  # FORCE LSTM
    ).to(device)
    logger.info(f"FNO initialized with use_lstm=True | {fno.count_params():,} trainable params")

    # Risk (keep frozen)
    risk = RiskMLP(
        input_dim=cfg["risk"]["input_dim"],
        hidden_dim=cfg["risk"]["hidden_dim"],
        num_classes=cfg["risk"]["num_classes"],
        seq_len=cfg["risk"].get("seq_len", 20),
        use_physio=cfg["risk"].get("use_physio", True),
    ).to(device)
    risk_ckpt = os.path.join(ROOT, "checkpoints/risk_bestgrf_PhaseA.pth")
    if os.path.exists(risk_ckpt):
        risk.load_state_dict(torch.load(risk_ckpt, map_location=device))
        logger.info(f"✅ Loaded Risk from {risk_ckpt}")
    risk.eval()
    for p in risk.parameters():
        p.requires_grad = False

    # ---- Loss & Optimizer ----
    opt = optim.Adam(fno.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
    loss_mse = nn.MSELoss()

    best_val_loss = float("inf")
    best_ep = 0

    # ---- Training Loop ----
    logger.info(f"\n{'='*80}")
    logger.info(f"Starting FNO-LSTM training: {args.epochs} epochs")
    logger.info(f"{'='*80}\n")

    for ep in range(1, args.epochs + 1):
        # Train
        fno.train()
        train_loss = 0.0
        n_batch = 0
        t0 = time.time()

        for bidx, batch in enumerate(dl_train):
            opt.zero_grad()

            pose_seq = batch["pose_seq"].to(device)  # (B, 5, 33, 3)
            bio_seq = batch["bio_seq"].to(device)  # (B, 20, 72)
            grf_future = batch["grf_future"].to(device)  # (B, 10, 12)
            markers_gt = batch["markers_gt"].to(device)  # (B, 84)
            ja_gt = batch["ja_gt"].to(device)  # (B, 23)

            with torch.no_grad():
                ja_pred, marker_pred = stgcn(pose_seq)

            # FNO forward
            grf_pred, _ = fno(bio_seq)
            loss_grf = loss_mse(grf_pred, grf_future)
            loss = loss_grf

            loss.backward()
            torch.nn.utils.clip_grad_norm_(fno.parameters(), 1.0)
            opt.step()

            train_loss += loss.item()
            n_batch += 1

            if (bidx + 1) % max(1, len(dl_train) // 5) == 0:
                logger.info(
                    f"  [Ep {ep:2d}] batch {bidx+1:3d}/{len(dl_train):3d} | "
                    f"loss_grf={loss_grf.item():.5f} | lr={opt.param_groups[0]['lr']:.2e}"
                )

        train_loss_avg = train_loss / n_batch
        t_ep = time.time() - t0

        # Val
        fno.eval()
        val_loss = 0.0
        n_val_batch = 0
        with torch.no_grad():
            for batch in dl_val:
                bio_seq = batch["bio_seq"].to(device)
                grf_future = batch["grf_future"].to(device)

                grf_pred, _ = fno(bio_seq)
                loss_grf = loss_mse(grf_pred, grf_future)
                val_loss += loss_grf.item()
                n_val_batch += 1

        val_loss_avg = val_loss / n_val_batch

        sched.step()

        # Log
        logger.info(
            f"Epoch {ep:2d}/{args.epochs} | "
            f"train_loss={train_loss_avg:.5f} | "
            f"val_loss={val_loss_avg:.5f} | "
            f"time={t_ep:.0f}s"
        )

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            best_ep = ep
            torch.save(fno.state_dict(), args.out_ckpt)
            logger.info(f"  ✨ NEW BEST! Saved to {args.out_ckpt}")

        if ep - best_ep >= 8:
            logger.info(f"\nEarly stopping (no improvement for 8 epochs)")
            break

    logger.info(f"\n{'='*80}")
    logger.info(f"Training complete.")
    logger.info(f"Best epoch: {best_ep} | Best val_loss: {best_val_loss:.5f}")
    logger.info(f"Checkpoint: {args.out_ckpt}")
    logger.info(f"{'='*80}\n")


if __name__ == "__main__":
    main()
