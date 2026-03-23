#!/usr/bin/env python3
"""
preprocess.py
一次性预处理：.b3d → HDF5

解决 7.8GB RAM 无法装下 2,776,079 帧的问题：
  - 逐文件流式读取，不累积 Python 对象
  - 写入压缩 HDF5（lzf，无损，读写速度快）
  - 同时写入 metadata.json（序列划分信息）

运行一次即可，之后训练直接读 HDF5：
  python preprocess.py
  # 输出：/mnt/d/Camargo2021_Formatted_No_Arm/processed/dataset.h5
  #       /mnt/d/Camargo2021_Formatted_No_Arm/processed/metadata.json

磁盘占用估算（2.7M 帧，lzf 压缩）：
  joint_angles/vel/acc: 3 × 2.7M × 23 × 4B ≈ 750 MB
  grf:                  2.7M × 12 × 4B       ≈ 130 MB
  com:                  2.7M × 3 × 4B        ≈  33 MB
  markers:              2.7M × 28×3 × 4B     ≈ 900 MB（可选跳过）
  其他字段              ≈ 100 MB
  总计（压缩后）        ≈ 1.5-2 GB
"""
import argparse
import hashlib
import json
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
        logging.FileHandler("preprocess.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("preprocess")


def main():
    parser = argparse.ArgumentParser(description="RehabGuardian 预处理：.b3d → HDF5")
    parser.add_argument("--data_root",  default="/mnt/d/Camargo2021_Formatted_No_Arm/")
    parser.add_argument("--out_dir",    default="/mnt/d/Camargo2021_Formatted_No_Arm/processed/")
    parser.add_argument("--skip_markers", action="store_true",
                        help="跳过 markers（节省 ~900MB 磁盘）")
    parser.add_argument("--chunk",      type=int, default=4096,
                        help="HDF5 chunk 大小（影响随机读性能）")
    args = parser.parse_args()

    try:
        import h5py
    except ImportError:
        logger.error("请安装 h5py: pip install h5py")
        sys.exit(1)

    try:
        from utils.b3d_loader import B3DLoader
        from utils.teacher_labeler import TeacherLabeler
    except ImportError as e:
        logger.error(f"导入失败: {e}")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    h5_path   = os.path.join(args.out_dir, "dataset.h5")
    meta_path = os.path.join(args.out_dir, "metadata.json")
    lock_path = os.path.join(args.out_dir, "data_version_lock.json")

    # ------------------------------------------------------------------
    # 扫描文件
    # ------------------------------------------------------------------
    loader   = B3DLoader(args.data_root)
    n_files  = len(loader.file_list)
    logger.info(f"找到 {n_files} 个 .b3d 文件")

    if n_files == 0:
        logger.error("无 .b3d 文件，退出")
        sys.exit(1)

    # 数据版本锁
    fingerprint = hashlib.sha256(
        "".join(sorted(loader.file_list)).encode()
    ).hexdigest()[:16]

    # ------------------------------------------------------------------
    # 两遍扫描：
    #   Pass 1：统计总帧数（预分配 HDF5）
    #   Pass 2：写入数据
    # ------------------------------------------------------------------
    logger.info("Pass 1：统计总帧数（逐文件计数，不进内存）...")

    seq_info = []    # [{seq_id, start, end, n_frames, filepath}, ...]
    total_frames = 0

    for idx, fp in enumerate(loader.file_list):
        logger.info(f"  计数 [{idx+1}/{n_files}] {os.path.basename(fp)}")
        file_start = total_frames
        for s in loader.load_trial(fp):
            if total_frames == file_start:
                current_seq = s.sequence_id
                seq_start   = total_frames
            if s.sequence_id != current_seq:
                seq_info.append({
                    "seq_id":   current_seq,
                    "start":    seq_start,
                    "end":      total_frames,
                    "filepath": fp,
                })
                current_seq = s.sequence_id
                seq_start   = total_frames
            total_frames += 1

        if total_frames > file_start:
            seq_info.append({
                "seq_id":   current_seq,
                "start":    seq_start,
                "end":      total_frames,
                "filepath": fp,
            })

    logger.info(f"Pass 1 完成：{total_frames} 帧，{len(seq_info)} 个序列")

    if total_frames == 0:
        logger.error("未读取到任何有效帧")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 创建 HDF5 文件（预分配）
    # ------------------------------------------------------------------
    logger.info(f"创建 HDF5 文件: {h5_path}")
    N  = total_frames
    C  = args.chunk
    CM = "lzf"   # lzf: 速度比 gzip 快 3-5x，压缩率略低

    labeler = TeacherLabeler()

    with h5py.File(h5_path, "w") as h5:
        # 必要字段
        ds_ja    = h5.create_dataset("joint_angles",   (N,23),   dtype="f4", chunks=(C,23),   compression=CM)
        ds_vel   = h5.create_dataset("joint_vel",      (N,23),   dtype="f4", chunks=(C,23),   compression=CM)
        ds_acc   = h5.create_dataset("joint_acc",      (N,23),   dtype="f4", chunks=(C,23),   compression=CM)
        ds_grf_l = h5.create_dataset("grf_left",       (N,6),    dtype="f4", chunks=(C,6),    compression=CM)
        ds_grf_r = h5.create_dataset("grf_right",      (N,6),    dtype="f4", chunks=(C,6),    compression=CM)
        ds_com   = h5.create_dataset("com",             (N,3),    dtype="f4", chunks=(C,3),    compression=CM)
        ds_mask  = h5.create_dataset("grf_mask",        (N,),     dtype="f4", chunks=(C,),     compression=CM)
        ds_pw    = h5.create_dataset("physics_weight",  (N,),     dtype="f4", chunks=(C,),     compression=CM)
        ds_risk  = h5.create_dataset("risk_label",      (N,),     dtype="i2", chunks=(C,),     compression=CM)
        ds_seq   = h5.create_dataset("seq_idx",         (N,),     dtype="i4", chunks=(C,),     compression=CM)

        if not args.skip_markers:
            ds_mk = h5.create_dataset("markers", (N,28,3), dtype="f4",
                                      chunks=(C,28,3), compression=CM)

        # ------------------------------------------------------------------
        # Pass 2：写入数据
        # ------------------------------------------------------------------
        logger.info("Pass 2：写入 HDF5...")
        written = 0
        t0      = time.time()

        # 用于批量写（减少 HDF5 写次数）
        BUF = 8192   # 每攒够 BUF 帧写一次
        buf = {
            "ja": [], "vel": [], "acc": [],
            "grf_l": [], "grf_r": [], "com": [],
            "mask": [], "pw": [], "risk": [], "seq_idx": [],
            "mk": [],
        }

        seq_map = {}   # seq_id → 整数索引
        seq_counter = 0

        def _flush(buf, written):
            n = len(buf["ja"])
            if n == 0:
                return written
            sl = slice(written, written + n)
            ds_ja[sl]    = np.stack(buf["ja"])
            ds_vel[sl]   = np.stack(buf["vel"])
            ds_acc[sl]   = np.stack(buf["acc"])
            ds_grf_l[sl] = np.stack(buf["grf_l"])
            ds_grf_r[sl] = np.stack(buf["grf_r"])
            ds_com[sl]   = np.stack(buf["com"])
            ds_mask[sl]  = buf["mask"]
            ds_pw[sl]    = buf["pw"]
            ds_risk[sl]  = buf["risk"]
            ds_seq[sl]   = buf["seq_idx"]
            if not args.skip_markers and buf["mk"]:
                ds_mk[sl] = np.stack(buf["mk"])
            for k in buf:
                buf[k].clear()
            return written + n

        for idx, fp in enumerate(loader.file_list):
            logger.info(f"  写入 [{idx+1}/{n_files}] {os.path.basename(fp)}")
            for s in loader.load_trial(fp):
                sid = s.sequence_id
                if sid not in seq_map:
                    seq_map[sid] = seq_counter
                    seq_counter += 1
                sidx = seq_map[sid]

                # 教师打标（单帧）
                grf12 = s.grf if s.grf is not None else np.zeros(12, np.float32)
                rl, _ = labeler.label_frame(s.joint_angles, grf12)

                buf["ja"].append(s.joint_angles)
                buf["vel"].append(s.joint_vel)
                buf["acc"].append(s.joint_acc)
                buf["grf_l"].append(s.grf_left)
                buf["grf_r"].append(s.grf_right)
                buf["com"].append(s.com)
                buf["mask"].append(s.grf_mask)
                buf["pw"].append(s.physics_weight)
                buf["risk"].append(rl)
                buf["seq_idx"].append(sidx)
                if not args.skip_markers:
                    mk = s.markers if s.markers is not None else np.zeros((28,3), np.float32)
                    buf["mk"].append(mk)

                if len(buf["ja"]) >= BUF:
                    written = _flush(buf, written)

            # 文件结束时刷新
            written = _flush(buf, written)

            elapsed = time.time() - t0
            fps_est = written / elapsed if elapsed > 0 else 0
            eta     = (N - written) / fps_est if fps_est > 0 else 0
            logger.info(
                f"    进度: {written}/{N} 帧 "
                f"({100*written/N:.1f}%) "
                f"速度:{fps_est:.0f}帧/s ETA:{eta/60:.1f}min"
            )

        # 最后刷新
        written = _flush(buf, written)
        logger.info(f"写入完成，共 {written} 帧")

        # 写元信息
        h5.attrs["total_frames"]  = written
        h5.attrs["n_sequences"]   = seq_counter
        h5.attrs["fingerprint"]   = fingerprint
        h5.attrs["created"]       = time.strftime("%Y-%m-%d %H:%M:%S")
        h5.attrs["target_fps"]    = 60
        h5.attrs["skip_markers"]  = args.skip_markers

    # ------------------------------------------------------------------
    # 写 metadata.json
    # ------------------------------------------------------------------
    # 按序列 ID 划分 train/val/test（避免数据泄露）
    all_seq_ids = sorted(seq_map.keys())
    n_seq = len(all_seq_ids)
    tr_end = int(n_seq * 0.70)
    va_end = int(n_seq * 0.85)

    metadata = {
        "total_frames":  written,
        "n_sequences":   n_seq,
        "fingerprint":   fingerprint,
        "created":       time.strftime("%Y-%m-%d %H:%M:%S"),
        "h5_path":       h5_path,
        "seq_id_to_int": seq_map,
        "split": {
            "train": all_seq_ids[:tr_end],
            "val":   all_seq_ids[tr_end:va_end],
            "test":  all_seq_ids[va_end:],
        },
        "split_sizes": {
            "train": tr_end,
            "val":   va_end - tr_end,
            "test":  n_seq - va_end,
        },
        "risk_label_dist": {},   # 下面填充
    }

    # 统计风险标签分布
    with h5py.File(h5_path, "r") as h5:
        labels = h5["risk_label"][:]
    dist = np.bincount(labels.clip(0,2), minlength=3)
    metadata["risk_label_dist"] = {
        "low": int(dist[0]),
        "mid": int(dist[1]),
        "high": int(dist[2]),
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # 数据版本锁
    lock = {
        "b3d_fingerprint":  fingerprint,
        "raw_fingerprint":  "",
        "b3d_file_count":   n_files,
        "raw_file_count":   0,
        "created":          time.strftime("%Y-%m-%d %H:%M:%S"),
        "python_seed":      42,
    }
    with open(lock_path, "w") as f:
        json.dump(lock, f, indent=2)

    # 打印摘要
    h5_size_gb = os.path.getsize(h5_path) / 1e9
    logger.info("=" * 60)
    logger.info("预处理完成！")
    logger.info(f"  HDF5 文件:   {h5_path}  ({h5_size_gb:.2f} GB)")
    logger.info(f"  总帧数:      {written:,}")
    logger.info(f"  序列数:      {n_seq}")
    logger.info(f"  Train/Val/Test 序列: {tr_end}/{va_end-tr_end}/{n_seq-va_end}")
    logger.info(f"  风险标签:    低={dist[0]:,} 中={dist[1]:,} 高={dist[2]:,}")
    logger.info(f"  耗时:        {(time.time()-t0)/60:.1f} 分钟")
    logger.info("=" * 60)
    logger.info("下一步：python main.py --mode train")


if __name__ == "__main__":
    main()