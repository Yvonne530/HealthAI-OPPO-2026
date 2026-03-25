#!/usr/bin/env python3
"""
continue_train.py
从已有 checkpoint 继续训练（不重新训练S1/S2）

策略（基于你 epoch51 的结果分析）：
  Phase A（约20 epoch）：S2 续训
    - lr=0.0003（比原来0.001低3倍，减少振荡）
    - patience=25（S2振荡型，需要更宽松）
    - 监控 grf_mae 和 risk_acc 双指标
  Phase B（约20 epoch）：S3 联合微调
    - lr=0.0001（全部参数，极低学习率）
    - 目标：grf_mae<0.35，risk_acc>0.83

用法：
  python continue_train.py                  # 标准续训
  python continue_train.py --phase b_only   # 直接跳到S3联合微调
  python continue_train.py --lr 0.0002      # 自定义学习率
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

LOG_FILE = f"logs/continue_{time.strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
logger = logging.getLogger("continue")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      default=os.path.join(ROOT, "configs/config.yaml"))
    parser.add_argument("--phase",       default="ab", choices=["ab","a_only","b_only"],
                        help="ab=先S2续训再S3微调 | a_only=只S2 | b_only=直接S3")
    parser.add_argument("--lr",          type=float, default=None)
    parser.add_argument("--phase_a_ep",  type=int,   default=20,  help="Phase A epoch数")
    parser.add_argument("--phase_b_ep",  type=int,   default=30,  help="Phase B epoch数")
    parser.add_argument("--patience",    type=int,   default=25)
    parser.add_argument("--batch",       type=int,   default=24)
    parser.add_argument("--epoch_frac",  type=float, default=0.10)
    parser.add_argument("--val_frac",    type=float, default=0.10)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg["project"]["seed"])
    np.random.seed(cfg["project"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    logger.info(f"设备: {device} | 日志: {LOG_FILE}")

    # ---- 路径 ----
    linux_h5   = os.path.expanduser("~/data/processed/dataset.h5")
    linux_meta = os.path.expanduser("~/data/processed/metadata.json")
    linux_norm = os.path.expanduser("~/data/processed/norm_stats.npz")
    h5_path    = linux_h5   if os.path.exists(linux_h5)   else cfg["data"]["h5_path"]
    meta_path  = linux_meta if os.path.exists(linux_meta) else cfg["data"]["meta_path"]
    norm_save  = linux_norm if os.path.exists(linux_norm) else cfg["inference"]["norm_stats"]
    logger.info(f"数据: {'Linux本地(快)' if os.path.exists(linux_h5) else '/mnt/d/(慢)'}")

    # ---- Dataset ----
    from utils.normalizer  import Normalizer
    from data.hdf5_dataset import HDF5RehabDataset
    from utils.kinematics  import finite_diff, compute_com

    norm = Normalizer()
    if os.path.exists(norm_save):
        norm.load(norm_save)

    ds_train = HDF5RehabDataset(h5_path, meta_path, "train", norm,
        seq_len=cfg["fno"]["seq_len"], future_k=cfg["fno"]["future_k"],
        vis_len=cfg["stgcn"]["num_frames"], augment=True)
    ds_val = HDF5RehabDataset(h5_path, meta_path, "val", norm,
        seq_len=cfg["fno"]["seq_len"], future_k=cfg["fno"]["future_k"],
        vis_len=cfg["stgcn"]["num_frames"], augment=False)

    n_tr = len(ds_train); n_val = len(ds_val)
    n_ep = max(50, int(n_tr  * args.epoch_frac))
    n_vp = max(50, int(n_val * args.val_frac))
    logger.info(f"训练{n_tr:,} → 每epoch{n_ep:,} | 验证{n_val:,} → {n_vp:,}")

    # ---- 模型 ----
    from models.stgcn      import STGCN
    from models.fno        import FNO1d
    from models.risk_model import RiskMLP

    stgcn = STGCN(
        num_nodes=cfg["stgcn"]["num_nodes"], in_channels=cfg["stgcn"]["in_channels"],
        hidden_channels=cfg["stgcn"]["hidden_channels"],
        num_frames=cfg["stgcn"]["num_frames"], output_dim=cfg["stgcn"]["output_dim"],
        dropout=cfg["stgcn"]["dropout"]).to(device)
    fno = FNO1d(
        input_dim=cfg["fno"]["input_dim"], output_dim=cfg["fno"]["output_dim"],
        future_k=cfg["fno"]["future_k"], modes=cfg["fno"]["modes"],
        width=cfg["fno"]["width"], depth=cfg["fno"]["depth"],
        seq_len=cfg["fno"]["seq_len"]).to(device)
    risk = RiskMLP(
        input_dim=cfg["risk"]["input_dim"], hidden_dim=cfg["risk"]["hidden_dim"],
        num_classes=cfg["risk"]["num_classes"],
        seq_len=cfg["risk"].get("seq_len",20),
        use_physio=cfg["risk"].get("use_physio",True)).to(device)

    # ---- 加载 checkpoint ----
    ckpt_dir = cfg["train"]["checkpoint_dir"]
    for m, name in [(stgcn,"stgcn_best.pth"),(fno,"fno_best.pth"),(risk,"risk_best.pth")]:
        p = os.path.join(ckpt_dir, name)
        if os.path.exists(p):
            m.load_state_dict(torch.load(p, map_location=device, weights_only=True))
            logger.info(f"  ✅ 加载 {name}")
        else:
            logger.error(f"  ❌ {name} 不存在，请先训练")
            sys.exit(1)

    logger.info(f"ST-GCN={stgcn.count_params()/1e6:.2f}M FNO={fno.count_params()/1e3:.0f}K "
                f"Risk={risk.count_params():,} lag={fno.current_lag_ms:.0f}ms")

    w = cfg["train"]["loss_weights"]
    bs = args.batch

    # ================================================================
    # 训练函数（复用）
    # ================================================================
    def run_phase(phase_name, n_epochs, lr, freeze_stgcn=False, patience=25):
        logger.info("=" * 60)
        logger.info(f"Phase {phase_name} | lr={lr} | epochs={n_epochs} | "
                    f"{'冻结ST-GCN' if freeze_stgcn else '全部可训'} | patience={patience}")
        logger.info("=" * 60)

        if freeze_stgcn:
            params = list(fno.parameters()) + list(risk.parameters())
            for p in stgcn.parameters(): p.requires_grad_(False)
            for p in fno.parameters():   p.requires_grad_(True)
            for p in risk.parameters():  p.requires_grad_(True)
        else:
            params = list(stgcn.parameters()) + list(fno.parameters()) + list(risk.parameters())
            for m in [stgcn, fno, risk]:
                for p in m.parameters(): p.requires_grad_(True)

        opt = optim.AdamW(params, lr=lr, weight_decay=1e-4)
        # 余弦退火（在本 phase 内）
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=lr*0.05)

        best_val = float("inf")
        best_grf = float("inf")
        best_risk_acc = 0.0
        pat_cnt = 0
        phase_history = []

        for ep in range(1, n_epochs + 1):
            t0 = time.time()
            stgcn.train(); fno.train(); risk.train()

            idx_tr = np.random.choice(n_tr, n_ep, replace=False)
            dl_tr  = DataLoader(Subset(ds_train, idx_tr.tolist()),
                                batch_size=bs, shuffle=True,
                                num_workers=0, pin_memory=(str(device)=="cuda"),
                                drop_last=True)
            n_bat  = len(dl_tr)
            tr_loss = 0.0; tr_grf = 0.0; t_b = time.time()

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
                lj  = F.mse_loss(pja, jag)
                lm  = F.mse_loss(pmk, mkg)
                ls  = stgcn.compute_symmetry_loss(pja)

                juse = pja if not freeze_stgcn else pja.detach()
                jexp = juse.unsqueeze(1).expand(-1,20,-1)
                fni  = torch.cat([jexp, finite_diff(jexp,1/60,1),
                                  finite_diff(jexp,1/60,2), compute_com(jexp)], -1)
                gall  = torch.cat([torch.zeros(B,20,12,device=device), gf], 1)
                pgrff, gal = fno(fni, gall)
                gt = gal if (gal is not None and gal.shape[1]==cfg["fno"]["future_k"]) else gf
                mm  = gm.unsqueeze(-1).expand_as(gt)
                pw_ = pw.reshape(-1,1,1).expand_as(gt)
                lgrf = ((pgrff-gt)**2*mm*pw_).sum()/(mm*pw_+1e-8).sum()

                rf = torch.cat([pja.detach(), pgrff[:,0].detach()],-1).unsqueeze(1).expand(-1,20,-1)
                hr_t = torch.full((B,1), 70.0, device=device)
                sl_t = torch.full((B,1), 80.0, device=device)
                lg, cf = risk(rf, hr_t, sl_t)
                cw = (1.0/(torch.bincount(rl.clamp(0,2),minlength=3).float()+1.0))
                cw = cw/cw.sum()*3.0
                lrisk = F.cross_entropy(lg, rl, weight=cw.to(device))
                lcnf  = F.binary_cross_entropy(cf,(lg.argmax(-1)==rl).float().unsqueeze(-1))

                tot = (w.get("joint_angles",0.30)*lj + w.get("marker",0.10)*lm
                     + w.get("grf",0.35)*lgrf       + w.get("risk",0.20)*lrisk
                     + w.get("conf",0.05)*lcnf       + 0.05*ls)

                tot.backward()
                nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                tr_loss += tot.item(); tr_grf += lgrf.item()

                if (bi+1) % 200 == 0 or bi == n_bat-1:
                    fps = (bi+1)*B/(time.time()-t_b)
                    eta = (n_bat-bi-1)*B/max(fps,1)
                    logger.info(f"  {phase_name} Ep{ep:02d}[训练][{bi+1:4d}/{n_bat}] "
                                f"loss={tr_loss/(bi+1):.4f} grf={tr_grf/(bi+1):.4f} "
                                f"{fps:.0f}s/s ETA:{eta/60:.1f}min")

            tr_avg = tr_loss/max(n_bat,1)
            sched.step()

            # ---- Validation ----
            stgcn.eval(); fno.eval(); risk.eval()
            idx_val = np.random.choice(n_val, n_vp, replace=False)
            dl_val = DataLoader(Subset(ds_val, idx_val.tolist()),
                                batch_size=bs, shuffle=False,
                                num_workers=0, pin_memory=(str(device)=="cuda"))
            vl=0.0; vg=0.0; rk=0; rt=0; nv=0; n_vbat=len(dl_val)

            with torch.no_grad():
                for vbi, batch in enumerate(dl_val):
                    jag_v = batch["ja_gt"].to(device)
                    gf_v  = batch["grf_future"].to(device)
                    gm_v  = batch["grf_mask"].to(device)
                    rl_v  = batch["risk_label"].to(device)
                    B_v   = jag_v.shape[0]

                    pja_v, _ = stgcn(batch["pose_seq"].to(device))
                    jx_v = pja_v.unsqueeze(1).expand(-1,20,-1)
                    fi_v = torch.cat([jx_v, finite_diff(jx_v,1/60,1),
                                      finite_diff(jx_v,1/60,2), compute_com(jx_v)], -1)
                    pg_v_tuple = fno(fi_v)
                    pg_v = pg_v_tuple[0] if isinstance(pg_v_tuple, tuple) else pg_v_tuple
                    mm_v = gm_v.unsqueeze(-1).expand_as(pg_v)
                    vg_  = ((pg_v-gf_v)**2*mm_v).sum()/(mm_v.sum()+1e-8)
                    vj   = F.mse_loss(pja_v, jag_v)
                    rf_v = torch.cat([pja_v, pg_v[:,0]],-1).unsqueeze(1).expand(-1,20,-1)
                    hr_v = torch.full((B_v,1),70.0,device=device)
                    sl_v = torch.full((B_v,1),80.0,device=device)
                    lv, _ = risk(rf_v, hr_v, sl_v)
                    cw_v = (1.0/(torch.bincount(rl_v.clamp(0,2),minlength=3).float()+1.0))
                    cw_v = cw_v/cw_v.sum()*3.0
                    v_risk = F.cross_entropy(lv, rl_v, weight=cw_v.to(device))
                    vl += (0.30*vj+0.35*vg_+0.20*v_risk).item()
                    vg += vg_.item()
                    rk += (lv.argmax(-1)==rl_v).sum().item()
                    rt += B_v; nv += 1

                    if (vbi+1) % 100 == 0 or vbi == n_vbat-1:
                        logger.info(f"  {phase_name} Ep{ep:02d}[验证][{vbi+1:3d}/{n_vbat}]")

            val_loss = vl/max(nv,1); val_grf = vg/max(nv,1)
            risk_acc = rk/max(rt,1); lag_ms = fno.current_lag_ms
            elapsed  = time.time()-t0

            phase_history.append({"val":val_loss,"grf":val_grf,"risk":risk_acc})

            logger.info(
                f"★ {phase_name} Ep{ep:02d}/{n_epochs} | "
                f"train={tr_avg:.4f}  val={val_loss:.4f}  "
                f"grf_mae={val_grf:.4f}  risk_acc={risk_acc:.3f}  "
                f"lag={lag_ms:.0f}ms | {elapsed/60:.1f}min  lr={sched.get_last_lr()[0]:.5f}"
            )

            # 双指标保存（val_loss 和 grf_mae 都有改善才算最优）
            improved = val_loss < best_val
            if val_grf < best_grf:
                best_grf = val_grf
                torch.save(fno.state_dict(),   os.path.join(ckpt_dir, f"fno_bestgrf_{phase_name}.pth"))
                torch.save(stgcn.state_dict(), os.path.join(ckpt_dir, f"stgcn_bestgrf_{phase_name}.pth"))
                torch.save(risk.state_dict(),  os.path.join(ckpt_dir, f"risk_bestgrf_{phase_name}.pth"))

            if improved:
                best_val = val_loss
                best_risk_acc = max(best_risk_acc, risk_acc)
                pat_cnt = 0
                for m, nm in [(stgcn,"stgcn_best.pth"),(fno,"fno_best.pth"),(risk,"risk_best.pth")]:
                    torch.save(m.state_dict(), os.path.join(ckpt_dir, nm))
                logger.info(f"  ✅ 最佳模型 val={best_val:.4f} grf={best_grf:.4f} risk={risk_acc:.3f}")
            else:
                pat_cnt += 1
                logger.info(f"  patience {pat_cnt}/{patience}")
                if pat_cnt >= patience:
                    logger.info(f"  早停")
                    break

        logger.info(f"{phase_name} 完成 | best_val={best_val:.4f} best_grf={best_grf:.4f}")
        return best_val, best_grf, best_risk_acc

    # ================================================================
    # 执行各 Phase
    # ================================================================
    lr_a = args.lr or 0.0003   # S2续训用较低LR
    lr_b = (args.lr or 0.0001) if args.phase == "b_only" else 0.0001  # S3微调

    if args.phase in ("ab", "a_only"):
        logger.info("\n【Phase A：S2续训（降低LR减少振荡）】")
        val_a, grf_a, risk_a = run_phase(
            "PhaseA", args.phase_a_ep, lr_a,
            freeze_stgcn=True, patience=args.patience
        )
        logger.info(f"Phase A 结果: val={val_a:.4f} grf={grf_a:.4f} risk={risk_a:.3f}")

        # 重新加载最优 checkpoint 进入 Phase B
        for m, name in [(stgcn,"stgcn_best.pth"),(fno,"fno_best.pth"),(risk,"risk_best.pth")]:
            p = os.path.join(ckpt_dir, name)
            if os.path.exists(p):
                m.load_state_dict(torch.load(p, map_location=device, weights_only=True))

    if args.phase in ("ab", "b_only"):
        logger.info("\n【Phase B：S3联合微调（全部参数，极低LR）】")
        val_b, grf_b, risk_b = run_phase(
            "PhaseB", args.phase_b_ep, lr_b,
            freeze_stgcn=False, patience=args.patience
        )
        logger.info(f"Phase B 结果: val={val_b:.4f} grf={grf_b:.4f} risk={risk_b:.3f}")

    logger.info("=" * 60)
    logger.info("续训完成！")
    logger.info(f"最终模型保存在 checkpoints/")
    logger.info("推理：python main.py --mode infer")
    logger.info("导出：python main.py --mode export")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()