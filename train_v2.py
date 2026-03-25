#!/usr/bin/env python3
"""
train_v2.py  ← 这是最终稳定版，直接替换之前所有训练脚本
解决：
  - num_workers=0（WSL2死锁）
  - 验证集也只采 10%，并打进度（不再静止62分钟）
  - 文件移到 Linux 路径后自动使用（快3-5x）
  - 每 batch 打印进度，不依赖 tqdm
  - 自动从上次断点继续

用法：
  python train_v2.py
  python train_v2.py --epoch_frac 0.05   # 更快看到每epoch结果
"""
import argparse, logging, os, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
import yaml
from torch.utils.data import DataLoader, Subset

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.makedirs("logs", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

LOG_FILE = f"logs/train_{time.strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
logger = logging.getLogger("train_v2")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default=os.path.join(ROOT, "configs/config.yaml"))
    parser.add_argument("--epoch_frac", type=float, default=0.10)
    parser.add_argument("--val_frac",   type=float, default=0.10)   # 验证集也只用10%
    parser.add_argument("--batch",      type=int,   default=None)
    parser.add_argument("--epochs",     type=int,   default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg["project"]["seed"])
    np.random.seed(cfg["project"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    logger.info(f"设备: {device} | 日志: {LOG_FILE}")

    bs       = args.batch  or cfg["train"]["batch_size"]
    n_epochs = args.epochs or cfg["train"]["num_epochs"]

    # ---- 优先使用 Linux 路径 ----
    linux_h5   = os.path.expanduser("~/data/processed/dataset.h5")
    linux_meta = os.path.expanduser("~/data/processed/metadata.json")
    linux_norm = os.path.expanduser("~/data/processed/norm_stats.npz")

    if os.path.exists(linux_h5):
        h5_path   = linux_h5
        meta_path = linux_meta
        norm_save = linux_norm
        logger.info("✅ 使用 Linux 本地文件（快速）")
    else:
        h5_path   = cfg["data"]["h5_path"]
        meta_path = cfg["data"]["meta_path"]
        norm_save = cfg["inference"]["norm_stats"]
        logger.info("⚠️  使用 /mnt/d/ 路径（较慢）。建议先复制文件：")
        logger.info("   mkdir -p ~/data/processed")
        logger.info("   cp /mnt/d/Camargo2021_Formatted_No_Arm/processed/dataset.h5 ~/data/processed/")
        logger.info("   cp /mnt/d/Camargo2021_Formatted_No_Arm/processed/metadata.json ~/data/processed/")

    if not os.path.exists(h5_path):
        logger.error(f"HDF5 不存在: {h5_path}")
        sys.exit(1)

    # ---- Dataset ----
    from utils.normalizer  import Normalizer
    from data.hdf5_dataset import HDF5RehabDataset

    norm = Normalizer()
    if os.path.exists(norm_save):
        norm.load(norm_save)
        logger.info(f"加载归一化参数: {norm_save}")

    ds_train = HDF5RehabDataset(h5_path, meta_path, "train", norm,
        seq_len=cfg["fno"]["seq_len"], future_k=cfg["fno"]["future_k"],
        vis_len=cfg["stgcn"]["num_frames"], augment=True)
    if not norm._fitted:
        norm.save(norm_save)

    ds_val = HDF5RehabDataset(h5_path, meta_path, "val", norm,
        seq_len=cfg["fno"]["seq_len"], future_k=cfg["fno"]["future_k"],
        vis_len=cfg["stgcn"]["num_frames"], augment=False)

    n_tr = len(ds_train); n_val = len(ds_val)
    n_per_ep     = max(50, int(n_tr  * args.epoch_frac))
    n_per_val_ep = max(50, int(n_val * args.val_frac))

    # 预估时间（RTX 4070 Laptop，num_workers=0，Linux文件系统）
    sec_per_batch = 0.08 if os.path.exists(linux_h5) else 0.25
    est_min = (n_per_ep + n_per_val_ep) / bs * sec_per_batch / 60

    logger.info(f"训练窗口: {n_tr:,} → 每epoch采样 {n_per_ep:,} ({args.epoch_frac*100:.0f}%)")
    logger.info(f"验证窗口: {n_val:,} → 每epoch采样 {n_per_val_ep:,} ({args.val_frac*100:.0f}%)")
    logger.info(f"batch={bs} | num_workers=0 | 预估每epoch: ~{est_min:.0f}分钟")

    # ---- 模型 ----
    from models.stgcn      import STGCN
    from models.fno        import FNO1d
    from models.risk_model import RiskMLP
    from utils.kinematics  import finite_diff, compute_com

    stgcn = STGCN(
        num_nodes=cfg["stgcn"]["num_nodes"], in_channels=cfg["stgcn"]["in_channels"],
        hidden_channels=cfg["stgcn"]["hidden_channels"],
        num_frames=cfg["stgcn"]["num_frames"], output_dim=cfg["stgcn"]["output_dim"],
        dropout=cfg["stgcn"]["dropout"],
    ).to(device)
    fno = FNO1d(
        input_dim=cfg["fno"]["input_dim"], output_dim=cfg["fno"]["output_dim"],
        future_k=cfg["fno"]["future_k"], modes=cfg["fno"]["modes"],
        width=cfg["fno"]["width"], depth=cfg["fno"]["depth"], seq_len=cfg["fno"]["seq_len"],
    ).to(device)
    risk = RiskMLP(
        input_dim=cfg["risk"]["input_dim"], hidden_dim=cfg["risk"]["hidden_dim"],
        num_classes=cfg["risk"]["num_classes"],
        seq_len=cfg["risk"].get("seq_len", 20),
        use_physio=cfg["risk"].get("use_physio", True),
    ).to(device)

    # ---- 加载 checkpoint ----
    ckpt_dir = cfg["train"]["checkpoint_dir"]
    start_ep = 1
    for m, name in [(stgcn,"stgcn_best.pth"),(fno,"fno_best.pth"),(risk,"risk_best.pth")]:
        p = os.path.join(ckpt_dir, name)
        if os.path.exists(p):
            try:
                m.load_state_dict(torch.load(p, map_location=device))
                logger.info(f"  ✅ 加载 {name}")
            except Exception as e:
                logger.warning(f"  ⚠️  {name}: {e}")
    state_p = os.path.join(ckpt_dir, "train_state.npz")
    if os.path.exists(state_p):
        st = np.load(state_p, allow_pickle=True)
        start_ep = int(st.get("epoch", 0)) + 1
        logger.info(f"从 epoch {start_ep} 继续")

    logger.info(f"ST-GCN={stgcn.count_params()/1e6:.2f}M FNO={fno.count_params()/1e3:.0f}K "
                f"Risk={risk.count_params():,} lag={fno.current_lag_ms:.0f}ms")

    # ---- 优化器 ----
    lr = cfg["train"]["learning_rate"]
    wd = cfg["train"]["weight_decay"]
    w  = cfg["train"]["loss_weights"]
    stage1 = int(n_epochs * 0.30)
    stage2 = int(n_epochs * 0.60)

    opt_s1 = optim.AdamW(stgcn.parameters(), lr=lr, weight_decay=wd)
    opt_s2 = optim.AdamW(list(fno.parameters())+list(risk.parameters()), lr=lr, weight_decay=wd)
    opt_s3 = optim.AdamW(list(stgcn.parameters())+list(fno.parameters())+list(risk.parameters()),
                         lr=lr*0.1, weight_decay=wd)

    best_val = float("inf"); patience_cnt = 0
    patience  = cfg["train"]["patience"]
    history   = {"tr":[], "val":[], "grf":[], "lag":[]}

    logger.info("=" * 60)
    logger.info(f"开始训练 epoch {start_ep}~{n_epochs} | 每200batch打印一次进度")
    logger.info("=" * 60)

    for epoch in range(start_ep, n_epochs + 1):
        t0 = time.time()
        stage = 1 if epoch <= stage1 else (2 if epoch <= stage2 else 3)
        opt   = [opt_s1, opt_s2, opt_s3][stage-1]

        for p in stgcn.parameters(): p.requires_grad_(stage in (1,3))
        for p in fno.parameters():   p.requires_grad_(stage in (2,3))
        for p in risk.parameters():  p.requires_grad_(stage in (2,3))
        sched_p = max(0.3, 1.0 - 0.7*(epoch-1)/max(n_epochs-1,1))

        # ---- 本epoch子集 ----
        idx_tr  = np.random.choice(n_tr,  n_per_ep,     replace=False)
        idx_val = np.random.choice(n_val, n_per_val_ep, replace=False)

        dl_tr = DataLoader(Subset(ds_train, idx_tr.tolist()),
                           batch_size=bs, shuffle=True,
                           num_workers=0, pin_memory=(str(device)=="cuda"), drop_last=True)
        dl_val_ep = DataLoader(Subset(ds_val, idx_val.tolist()),
                               batch_size=bs, shuffle=False,
                               num_workers=0, pin_memory=(str(device)=="cuda"))
        n_bat = len(dl_tr); n_vbat = len(dl_val_ep)

        # ---- Train ----
        stgcn.train(); fno.train(); risk.train()
        tr_loss = 0.0; tr_grf = 0.0; t_batch = time.time()

        for bi, batch in enumerate(dl_tr):
            opt.zero_grad()
            ps  = batch["pose_seq"].to(device)
            jag = batch["ja_gt"].to(device)
            mkg = batch["markers_gt"].to(device)
            gf  = batch["grf_future"].to(device)
            gm  = batch["grf_mask"].to(device)
            rl  = batch["risk_label"].to(device)
            pw  = batch["physics_weight"].to(device)
            B   = ps.shape[0]

            pja, pmk = stgcn(ps)
            lj = F.mse_loss(pja, jag)
            lm = F.mse_loss(pmk, mkg)
            ls = stgcn.compute_symmetry_loss(pja)

            juse = (jag if torch.rand(1).item() < sched_p else pja.detach())
            jexp = juse.unsqueeze(1).expand(-1,20,-1)
            fni  = torch.cat([jexp, finite_diff(jexp,1/60,1),
                              finite_diff(jexp,1/60,2), compute_com(jexp)], -1)
            pgrff, gal = fno(fni, torch.cat([torch.zeros(B,20,12,device=device), gf], 1))
            gt  = gal if (gal is not None and gal.shape[1]==cfg["fno"]["future_k"]) else gf
            mm  = gm.unsqueeze(-1).expand_as(gt)
            pww = pw.reshape(-1,1,1).expand_as(gt)
            lgrf = ((pgrff-gt)**2*mm*pww).sum()/(mm*pww+1e-8).sum()

            rf = torch.cat([pja.detach(), pgrff[:,0].detach()],-1).unsqueeze(1).expand(-1,20,-1)
            hr_t = torch.full((B, 1), 70.0, device=device)
            sl_t = torch.full((B, 1), 80.0, device=device)
            lg, cf = risk(rf, hr_t, sl_t)
            cw = (1.0/(torch.bincount(rl.clamp(0,2),minlength=3).float()+1.0))
            cw = cw/cw.sum()*3.0
            lrisk = F.cross_entropy(lg, rl, weight=cw.to(device))
            lcnf  = F.binary_cross_entropy(cf, (lg.argmax(-1)==rl).float().unsqueeze(-1))

            if stage == 1:
                tot = w.get("joint_angles",0.30)*lj + w.get("marker",0.10)*lm + 0.05*ls
            else:
                tot = (w.get("joint_angles",0.30)*lj + w.get("marker",0.10)*lm
                     + w.get("grf",0.35)*lgrf + w.get("risk",0.20)*lrisk
                     + w.get("conf",0.05)*lcnf + 0.05*ls)

            tot.backward()
            nn.utils.clip_grad_norm_([p for g in [stgcn,fno,risk]
                                      for p in g.parameters() if p.requires_grad], 1.0)
            opt.step()
            tr_loss += tot.item(); tr_grf += lgrf.item()

            if (bi+1) % 200 == 0 or bi == n_bat-1:
                el = time.time()-t_batch
                fps = (bi+1)*B/el
                eta = (n_bat-bi-1)*B/max(fps,1)
                logger.info(f"  Ep{epoch:03d}[S{stage}][训练] "
                            f"[{bi+1:5d}/{n_bat}] "
                            f"loss={tr_loss/(bi+1):.4f} grf={tr_grf/(bi+1):.4f} "
                            f"{fps:.0f}samp/s ETA:{eta/60:.1f}min")

        tr_avg = tr_loss/max(n_bat,1)

        # ---- Validation（有进度显示）----
        stgcn.eval(); fno.eval(); risk.eval()
        vl=0.0; vg=0.0; rk=0; rt=0; nv=0
        logger.info(f"  Ep{epoch:03d} 开始验证 ({n_vbat} batch)...")

        with torch.no_grad():
            for vbi, batch in enumerate(dl_val_ep):
                jag_v = batch["ja_gt"].to(device)
                gf_v  = batch["grf_future"].to(device)
                gm_v  = batch["grf_mask"].to(device)
                rl_v  = batch["risk_label"].to(device)

                pja_v, _ = stgcn(batch["pose_seq"].to(device))
                jx_v = pja_v.unsqueeze(1).expand(-1,20,-1)
                fi_v = torch.cat([jx_v, finite_diff(jx_v,1/60,1),
                                  finite_diff(jx_v,1/60,2), compute_com(jx_v)], -1)
                pg_v, _ = fno(fi_v)
                mm_v = gm_v.unsqueeze(-1).expand_as(pg_v)
                vg  += ((pg_v-gf_v)**2*mm_v).sum()/(mm_v.sum()+1e-8)
                vj   = F.mse_loss(pja_v, jag_v)
                rf_v = torch.cat([pja_v, pg_v[:,0]],-1).unsqueeze(1).expand(-1,20,-1)
                B_v = rf_v.shape[0]
                hr_v = torch.full((B_v, 1), 70.0, device=device)
                sl_v = torch.full((B_v, 1), 80.0, device=device)
                lv, _ = risk(rf_v, hr_v, sl_v)
                cw_v = (1.0/(torch.bincount(rl_v.clamp(0,2),minlength=3).float()+1.0))
                cw_v = cw_v/cw_v.sum()*3.0
                vl  += (0.30*vj + 0.35*((pg_v-gf_v)**2*mm_v).sum()/(mm_v.sum()+1e-8)
                      + 0.20*F.cross_entropy(lv, rl_v, weight=cw_v.to(device))).item()
                rk  += (lv.argmax(-1)==rl_v).sum().item()
                rt  += rl_v.shape[0]; nv += 1

                # 验证进度（每100batch）
                if (vbi+1) % 100 == 0 or vbi == n_vbat-1:
                    logger.info(f"  Ep{epoch:03d}[验证] [{vbi+1:4d}/{n_vbat}]")

        val_loss = vl/max(nv,1); val_grf = vg/max(nv,1)
        risk_acc = rk/max(rt,1); lag_ms = fno.current_lag_ms
        elapsed  = time.time()-t0

        history["tr"].append(tr_avg); history["val"].append(val_loss)
        history["grf"].append(float(val_grf)); history["lag"].append(lag_ms)

        logger.info(
            f"★ Epoch {epoch:03d}/{n_epochs} [S{stage}] | "
            f"train={tr_avg:.4f}  val={val_loss:.4f}  "
            f"grf_mae={float(val_grf):.4f}  risk_acc={risk_acc:.3f}  "
            f"lag={lag_ms:.0f}ms | {elapsed/60:.1f}min"
        )

        if val_loss < best_val:
            best_val = val_loss; patience_cnt = 0
            for m, nm in [(stgcn,"stgcn_best.pth"),(fno,"fno_best.pth"),(risk,"risk_best.pth")]:
                torch.save(m.state_dict(), os.path.join(ckpt_dir, nm))
            np.savez(os.path.join(ckpt_dir,"train_state.npz"), epoch=epoch, best_val=best_val)
            logger.info(f"  ✅ 最佳模型保存 (val={best_val:.4f})")
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                logger.info(f"早停 (patience={patience})")
                break

        # 余弦学习率
        scale = 0.5*(1+np.cos(np.pi*epoch/n_epochs))
        for pg in opt.param_groups: pg["lr"] = lr * max(scale, 0.01)

    np.savez("logs/history.npz", **{k:np.array(v) for k,v in history.items()})
    logger.info(f"训练完成 | best_val={best_val:.4f} | lag={fno.current_lag_ms:.0f}ms")
    logger.info("推理测试：python main.py --mode infer")


if __name__ == "__main__":
    main()