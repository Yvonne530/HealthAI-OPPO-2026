#!/usr/bin/env python3
"""
relabel.py
重新生成 HDF5 中的 risk_label 列（无需重新读取 .b3d 文件）

你的 HDF5 已有的数据（不变）：
  joint_angles, joint_vel, joint_acc, grf_left, grf_right,
  com, grf_mask, physics_weight, seq_idx

本脚本只做一件事：读取 joint_angles + grf，重新调用修复后的
TeacherLabeler，覆盖写回 risk_label。

预计耗时：5-10 分钟（读+计算+写 2.7M 帧）
不需要 nimblephysics，不需要读 .b3d

用法：
  python relabel.py
  python relabel.py --h5_path /mnt/d/.../processed/dataset.h5
  python relabel.py --dry_run    # 只统计，不写入
"""
import argparse
import logging
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("relabel.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("relabel")


def main():
    parser = argparse.ArgumentParser(description="重新生成 HDF5 风险标签（无需重读 .b3d）")
    parser.add_argument("--h5_path", default="/mnt/d/Camargo2021_Formatted_No_Arm/processed/dataset.h5")
    parser.add_argument("--batch",   type=int, default=65536, help="批量处理大小")
    parser.add_argument("--dry_run", action="store_true", help="只统计分布，不写入")
    args = parser.parse_args()

    try:
        import h5py
    except ImportError:
        logger.error("请安装 h5py: pip install h5py"); sys.exit(1)

    if not os.path.exists(args.h5_path):
        logger.error(f"HDF5 文件不存在: {args.h5_path}")
        logger.error("请先运行: python preprocess.py")
        sys.exit(1)

    from utils.teacher_labeler import TeacherLabeler, THRESH_MED, THRESH_HIGH
    labeler = TeacherLabeler()

    # ----------------------------------------------------------------
    # 打开 HDF5
    # ----------------------------------------------------------------
    logger.info(f"打开 HDF5: {args.h5_path}")
    with h5py.File(args.h5_path, "a") as h5:   # "a" = 读写模式
        N = h5["joint_angles"].shape[0]
        logger.info(f"总帧数: {N:,}")

        # ---- 1. 统计现有标签分布 ----
        old_labels = h5["risk_label"][:]
        old_dist   = np.bincount(np.clip(old_labels, 0, 2), minlength=3)
        logger.info(f"现有标签分布: LOW={old_dist[0]:,} MED={old_dist[1]:,} HIGH={old_dist[2]:,}")

        # ---- 2. 批量重新打标 ----
        new_labels = np.zeros(N, dtype=np.int16)
        new_scores = np.zeros(N, dtype=np.float32)

        t0 = time.time()
        n_batches = (N + args.batch - 1) // args.batch

        for b_idx in range(n_batches):
            s = b_idx * args.batch
            e = min(s + args.batch, N)

            ja_batch   = h5["joint_angles"][s:e]   # (B, 23)
            grf_l_batch= h5["grf_left"][s:e]       # (B, 6)
            grf_r_batch= h5["grf_right"][s:e]      # (B, 6)
            grf_batch  = np.concatenate([grf_l_batch, grf_r_batch], axis=1)  # (B, 12)

            for i in range(e - s):
                lbl, sc = labeler.label_frame(ja_batch[i], grf_batch[i])
                new_labels[s + i] = lbl
                new_scores[s + i] = sc

            # 进度
            elapsed = time.time() - t0
            fps     = (e) / elapsed
            eta     = (N - e) / fps if fps > 0 else 0
            if b_idx % max(1, n_batches // 20) == 0:
                logger.info(
                    f"  {e:>10,}/{N:,} ({100*e/N:.1f}%) "
                    f"速度:{fps:.0f}帧/s ETA:{eta/60:.1f}min"
                )

        elapsed_total = time.time() - t0

        # ---- 3. 统计新标签分布 ----
        new_dist = np.bincount(np.clip(new_labels, 0, 2), minlength=3)
        logger.info(f"\n新标签分布: LOW={new_dist[0]:,}  MED={new_dist[1]:,}  HIGH={new_dist[2]:,}")
        logger.info(
            f"比例: LOW={new_dist[0]/N*100:.1f}% "
            f"MED={new_dist[1]/N*100:.1f}% "
            f"HIGH={new_dist[2]/N*100:.1f}%"
        )
        logger.info(f"耗时: {elapsed_total/60:.1f} 分钟")

        if new_dist[1] == 0:
            logger.warning("⚠️  MEDIUM 仍为 0，阈值可能需要进一步调整")
            logger.warning(
                "分数统计：\n"
                f"  score 均值={new_scores.mean():.3f} "
                f"p25={np.percentile(new_scores,25):.3f} "
                f"p50={np.percentile(new_scores,50):.3f} "
                f"p75={np.percentile(new_scores,75):.3f} "
                f"p90={np.percentile(new_scores,90):.3f}"
            )
            logger.warning(f"  当前 MEDIUM 阈值={THRESH_MED}, HIGH 阈值={THRESH_HIGH}")

        if args.dry_run:
            logger.info("[dry_run] 不写入，仅统计完毕")
            return

        # ---- 4. 写回 HDF5 ----
        logger.info("写入 risk_label 到 HDF5...")
        h5["risk_label"][:] = new_labels
        h5.flush()
        logger.info("✅ 写入完成")

    # ---- 5. 更新 metadata.json ----
    meta_path = os.path.join(os.path.dirname(args.h5_path), "metadata.json")
    if os.path.exists(meta_path):
        import json
        with open(meta_path) as f:
            meta = json.load(f)
        meta["risk_label_dist"] = {
            "low":  int(new_dist[0]),
            "mid":  int(new_dist[1]),
            "high": int(new_dist[2]),
        }
        meta["relabeled_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        meta["labeler_version"] = "v2_fixed"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"metadata.json 已更新: {meta_path}")

    logger.info("=" * 60)
    logger.info("relabel 完成！可以开始训练：")
    logger.info("  python main.py --mode train")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()