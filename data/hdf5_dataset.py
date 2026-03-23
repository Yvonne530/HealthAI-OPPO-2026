"""
data/hdf5_dataset.py
HDF5 驱动的 PyTorch Dataset，解决 7.8GB RAM 装不下 2.7M 帧的问题。

核心设计：
  - HDF5 文件只在 __getitem__ 时按需读取，不全量加载
  - 每个 worker 独立持有 HDF5 文件句柄（h5py 线程不安全，worker 各自开）
  - 滑窗索引预先计算并缓存在 RAM（每帧仅需 4 字节 int32，2.7M帧 ≈ 11MB）
  - train/val/test 按 seq_idx 切分（不随机打乱帧），杜绝数据泄露

内存占用分析（训练时）：
  - 窗口索引：2.7M × 4B = 11MB
  - 每个 batch(B=16) 读取：16×20×72×4B = 1.5MB（远小于 RAM 上限）
  - HDF5 系统缓存：约 256MB（OS 自动管理）
  - 总额外 RAM：< 512MB ✅
"""
import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

FUTURE_K    = 10
SEQ_LEN     = 20    # FNO 输入时序长度
VIS_LEN     = 5     # ST-GCN 输入帧数
N_VIS_NODES = 33


class HDF5RehabDataset(Dataset):
    """
    从 HDF5 文件流式读取，支持 DataLoader 多进程。

    每个样本返回 dict：
      pose_seq:    (VIS_LEN, 33, 3)    视觉骨骼序列（含增强）
      ja_gt:       (23,)               关节角度 GT（归一化后）
      markers_gt:  (84,)               标记点 GT（若 h5 含 markers）
      bio_seq:     (SEQ_LEN, 72)       [ja|vel|acc|com] 时序（归一化后）
      grf_future:  (FUTURE_K, 12)      未来 K 帧 GRF GT（归一化后）
      grf_mask:    (FUTURE_K,)         GRF 置信度
      physics_weight: scalar           物理残差权重
      risk_label:  int                 0/1/2
    """

    def __init__(
        self,
        h5_path:    str,
        meta_path:  str,
        split:      str = "train",
        normalizer  = None,
        seq_len:    int = SEQ_LEN,
        future_k:   int = FUTURE_K,
        vis_len:    int = VIS_LEN,
        augment:    bool = True,
        noise_std:  float = 0.015,
        occ_prob:   float = 0.30,
    ):
        self.h5_path   = h5_path
        self.seq_len   = seq_len
        self.future_k  = future_k
        self.vis_len   = vis_len
        self.norm      = normalizer
        self.augment   = augment and (split == "train")
        self.noise_std = noise_std
        self.occ_prob  = occ_prob

        # 加载 metadata（很小，放 RAM）
        with open(meta_path) as f:
            meta = json.load(f)

        self.total_frames = meta["total_frames"]
        seq_id_to_int     = meta["seq_id_to_int"]
        split_seqs        = set(meta["split"][split])  # 该 split 的 seq_id 集合

        # 将 seq_id → int 映射反转，过滤出该 split 的 seq_idx 集合
        split_seq_ints = {v for k, v in seq_id_to_int.items() if k in split_seqs}

        # ------------------------------------------------------------------
        # 预读取 seq_idx 数组（仅 int32，2.7M 帧 ≈ 11MB）
        # ------------------------------------------------------------------
        logger.info(f"[HDF5Dataset/{split}] 读取 seq_idx 索引...")
        try:
            import h5py
            with h5py.File(h5_path, "r") as h5:
                seq_idx_all = h5["seq_idx"][:].astype(np.int32)   # (N,)
                has_markers = "markers" in h5
        except Exception as e:
            raise RuntimeError(f"HDF5 读取失败: {e}")

        self.has_markers = has_markers

        # 标记属于本 split 的帧（布尔掩码）
        in_split = np.isin(seq_idx_all, list(split_seq_ints))

        # ------------------------------------------------------------------
        # 构建滑窗索引：
        #   窗口起点 i 合法条件：
        #   1. 帧 [i, i+vis_len+future_k) 全在 split 内
        #   2. 帧 [i, i+vis_len+future_k) 全属于同一序列
        # ------------------------------------------------------------------
        logger.info(f"[HDF5Dataset/{split}] 构建滑窗索引...")
        need = vis_len + future_k   # 窗口需要的总帧数（vis历史 + future目标）
        # 实际上还需要 seq_len 帧的生物力学序列（可能比 vis_len 更长）
        # 取 max(seq_len, vis_len) 作为历史需求
        hist_need = max(seq_len, vis_len)
        need      = hist_need + future_k

        valid_windows = []
        i = 0
        while i <= self.total_frames - need:
            # 快速检查：窗口内所有帧是否在 split 且同一序列
            window_seqs = seq_idx_all[i: i + need]
            if in_split[i] and (window_seqs == window_seqs[0]).all():
                valid_windows.append(i)
                i += 1
            else:
                # 跳到下一个 in_split 帧
                next_valid = np.argmax(in_split[i+1:]) + i + 1
                if not in_split[i+1:].any():
                    break
                i = next_valid

        self.windows    = np.array(valid_windows, dtype=np.int32)
        self.hist_need  = hist_need

        logger.info(
            f"[HDF5Dataset/{split}] {len(self.windows)} 个窗口 "
            f"(frames={in_split.sum()}, hist={hist_need}, K={future_k})"
        )

        # ------------------------------------------------------------------
        # Normalizer 统计（仅 train 需要 fit，val/test 直接用）
        # ------------------------------------------------------------------
        if normalizer is not None and normalizer._fitted:
            logger.info(f"[HDF5Dataset/{split}] 使用已 fit 的 normalizer")
        elif normalizer is not None and split == "train":
            logger.info("[HDF5Dataset/train] 从 HDF5 计算归一化统计量（采样 10 万帧）...")
            self._fit_normalizer_from_hdf5(normalizer, h5_path, in_split)

        # HDF5 文件句柄（每个 worker 延迟初始化）
        self._h5: Optional[object] = None

    def _fit_normalizer_from_hdf5(self, normalizer, h5_path: str, in_split: np.ndarray) -> None:
        """从 HDF5 采样数据 fit normalizer（不全量加载）"""
        import h5py

        n_sample = min(100_000, in_split.sum())
        # 在 split 帧中均匀采样
        split_indices = np.where(in_split)[0]
        chosen = split_indices[
            np.linspace(0, len(split_indices)-1, n_sample, dtype=int)
        ]

        with h5py.File(h5_path, "r") as h5:
            ja   = h5["joint_angles"][chosen]    # (n, 23)
            vel  = h5["joint_vel"][chosen]
            acc  = h5["joint_acc"][chosen]
            grf_l= h5["grf_left"][chosen]
            grf_r= h5["grf_right"][chosen]
            com  = h5["com"][chosen]
            if self.has_markers:
                mk = h5["markers"][chosen].reshape(n_sample, -1)
            else:
                mk = np.zeros((n_sample, 84), np.float32)

        # 直接用 numpy 计算 channel-wise stats
        eps = 1e-8
        stats = {}
        for name, arr in [
            ("joint_angles", ja), ("joint_vel", vel), ("joint_acc", acc),
            ("markers_flat", mk), ("grf_left", grf_l), ("grf_right", grf_r),
            ("com", com),
        ]:
            stats[f"{name}_mean"] = arr.mean(0).astype(np.float32)
            stats[f"{name}_std"]  = (arr.std(0) + eps).astype(np.float32)

        normalizer.stats   = stats
        normalizer._fitted = True
        logger.info(f"  归一化参数计算完成（采样 {n_sample} 帧）")

    # ------------------------------------------------------------------
    # Dataset 接口
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.windows)

    def _get_h5(self):
        """延迟初始化 HDF5 句柄（每个 DataLoader worker 独立持有）"""
        if self._h5 is None:
            import h5py
            self._h5 = h5py.File(self.h5_path, "r", swmr=True)
        return self._h5

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start    = int(self.windows[idx])
        h5       = self._get_h5()
        hist_end = start + self.hist_need
        fut_end  = hist_end + self.future_k

        # --- 按需从 HDF5 读取窗口数据 ---
        ja_hist   = h5["joint_angles"][start:hist_end]    # (hist, 23)
        vel_hist  = h5["joint_vel"][start:hist_end]
        acc_hist  = h5["joint_acc"][start:hist_end]
        com_hist  = h5["com"][start:hist_end]             # (hist, 3)
        grf_l_fut = h5["grf_left"][hist_end:fut_end]      # (K, 6)
        grf_r_fut = h5["grf_right"][hist_end:fut_end]     # (K, 6)
        mask_fut  = h5["grf_mask"][hist_end:fut_end]      # (K,)
        pw        = float(h5["physics_weight"][start])
        risk_lbl  = int(h5["risk_label"][hist_end])

        # 标记点（可能不存在）
        if self.has_markers:
            mk_hist = h5["markers"][start:hist_end]        # (hist, 28, 3)
        else:
            mk_hist = np.zeros((self.hist_need, 28, 3), np.float32)

        # --- 归一化 ---
        norm = self.norm
        if norm and norm._fitted:
            def _n(arr, name, shape):
                flat = arr.reshape(-1, arr.shape[-1] if arr.ndim>1 else 1)
                out  = (flat - norm.stats[f"{name}_mean"]) / norm.stats[f"{name}_std"]
                return out.reshape(arr.shape).astype(np.float32)

            ja_hist  = _n(ja_hist,  "joint_angles", ja_hist.shape)
            vel_hist = _n(vel_hist, "joint_vel",    vel_hist.shape)
            acc_hist = _n(acc_hist, "joint_acc",    acc_hist.shape)
            com_hist = _n(com_hist, "com",          com_hist.shape)
            grf_l_fut_n = _n(grf_l_fut, "grf_left",  grf_l_fut.shape)
            grf_r_fut_n = _n(grf_r_fut, "grf_right", grf_r_fut.shape)
        else:
            grf_l_fut_n = grf_l_fut.astype(np.float32)
            grf_r_fut_n = grf_r_fut.astype(np.float32)

        # --- 构建各任务输入 ---

        # 1) 视觉骨骼序列 (VIS_LEN, 33, 3)
        mk_vis = mk_hist[-self.vis_len:]             # 取最近 vis_len 帧
        pose_seq = np.zeros((self.vis_len, 33, 3), np.float32)
        n = min(mk_vis.shape[0], 33)
        valid = ~np.any(np.isnan(mk_vis[:, :n]), axis=-1)  # (vis_len, n)
        pose_seq[:, :n][valid] = mk_vis[:, :n][valid]

        if self.augment:
            pose_seq = self._augment_pose(pose_seq)

        # 2) ST-GCN 目标 ja_gt + markers_gt（当前帧）
        cur_ja     = ja_hist[-1]                     # (23,)
        cur_mk_raw = mk_hist[-1].flatten()           # (84,)
        if norm and norm._fitted:
            cur_mk = (cur_mk_raw.reshape(1,-1) - norm.stats["markers_flat_mean"]) / \
                      norm.stats["markers_flat_std"]
            cur_mk = cur_mk.flatten().astype(np.float32)
        else:
            cur_mk = cur_mk_raw.astype(np.float32)
        cur_mk = np.nan_to_num(cur_mk, 0.0)

        # 3) FNO 时序特征 (SEQ_LEN, 72)
        # 取最近 seq_len 帧，不足时前面 padding 0
        sl = self.seq_len
        bio_seq = np.zeros((sl, 72), np.float32)
        hist_slice = slice(max(0, self.hist_need - sl), self.hist_need)
        n_avail = ja_hist[hist_slice].shape[0]
        pad_start = sl - n_avail
        bio_seq[pad_start:, :23] = ja_hist[hist_slice]
        bio_seq[pad_start:, 23:46] = vel_hist[hist_slice]
        bio_seq[pad_start:, 46:69] = acc_hist[hist_slice]
        bio_seq[pad_start:, 69:72] = com_hist[hist_slice]

        # 4) 未来 GRF (K, 12)
        grf_future = np.concatenate([grf_l_fut_n, grf_r_fut_n], axis=-1)  # (K,12)

        # 5) 填充 mask/grf_future 到 FUTURE_K
        K = self.future_k
        if grf_future.shape[0] < K:
            pad = K - grf_future.shape[0]
            grf_future = np.concatenate([grf_future, np.zeros((pad,12),np.float32)], 0)
            mask_fut   = np.concatenate([mask_fut,   np.zeros(pad,np.float32)], 0)

        return {
            "pose_seq":       torch.from_numpy(pose_seq),
            "ja_gt":          torch.from_numpy(cur_ja.astype(np.float32)),
            "markers_gt":     torch.from_numpy(cur_mk),
            "bio_seq":        torch.from_numpy(bio_seq),
            "grf_future":     torch.from_numpy(grf_future[:K]),
            "grf_mask":       torch.from_numpy(mask_fut[:K].astype(np.float32)),
            "physics_weight": torch.tensor(pw, dtype=torch.float32),
            "risk_label":     torch.tensor(risk_lbl, dtype=torch.long),
        }

    def _augment_pose(self, pose_seq: np.ndarray) -> np.ndarray:
        out = pose_seq.copy()
        T, N, C = out.shape
        out += np.random.randn(*out.shape).astype(np.float32) * self.noise_std
        if np.random.rand() < self.occ_prob:
            n_occ = max(1, int(N * 0.20))
            occ   = np.random.choice(N, n_occ, replace=False)
            out[:, occ, :] = 0.0
        # 小角度旋转
        ang = np.random.uniform(-0.17, 0.17)   # ±10°
        c, s = np.cos(ang), np.sin(ang)
        R = np.array([[c,0,s],[0,1,0],[-s,0,c]], np.float32)
        return (out @ R.T).astype(np.float32)

    def __del__(self):
        if self._h5 is not None:
            try:
                self._h5.close()
            except Exception:
                pass


def build_datasets(
    h5_path:   str,
    meta_path: str,
    normalizer = None,
    seq_len:   int = SEQ_LEN,
    future_k:  int = FUTURE_K,
) -> Tuple[HDF5RehabDataset, HDF5RehabDataset, HDF5RehabDataset]:
    """一次性创建 train/val/test 三个 Dataset"""
    ds_train = HDF5RehabDataset(h5_path, meta_path, "train", normalizer,
                                seq_len=seq_len, future_k=future_k, augment=True)
    ds_val   = HDF5RehabDataset(h5_path, meta_path, "val",   normalizer,
                                seq_len=seq_len, future_k=future_k, augment=False)
    ds_test  = HDF5RehabDataset(h5_path, meta_path, "test",  normalizer,
                                seq_len=seq_len, future_k=future_k, augment=False)
    return ds_train, ds_val, ds_test


if __name__ == "__main__":
    import tempfile, json
    logging.basicConfig(level=logging.INFO)

    # 合成 HDF5 测试
    try:
        import h5py
    except ImportError:
        print("h5py 未安装，跳过测试")
        exit(0)

    N    = 5000
    rng  = np.random.default_rng(0)
    tmpd = tempfile.mkdtemp()
    h5p  = os.path.join(tmpd, "test.h5")
    mp   = os.path.join(tmpd, "meta.json")

    with h5py.File(h5p, "w") as h5:
        h5.create_dataset("joint_angles",  data=rng.normal(0,.3,(N,23)).astype(np.float32))
        h5.create_dataset("joint_vel",     data=rng.normal(0,1.,(N,23)).astype(np.float32))
        h5.create_dataset("joint_acc",     data=rng.normal(0,5.,(N,23)).astype(np.float32))
        h5.create_dataset("grf_left",      data=rng.uniform(0,600,(N,6)).astype(np.float32))
        h5.create_dataset("grf_right",     data=rng.uniform(0,600,(N,6)).astype(np.float32))
        h5.create_dataset("com",           data=rng.normal(0,1.,(N,3)).astype(np.float32))
        h5.create_dataset("grf_mask",      data=np.ones(N,np.float32))
        h5.create_dataset("physics_weight",data=np.ones(N,np.float32))
        h5.create_dataset("risk_label",    data=rng.integers(0,3,N).astype(np.int16))
        seq_idx = np.zeros(N, np.int32)
        seq_idx[2000:] = 1; seq_idx[4000:] = 2
        h5.create_dataset("seq_idx", data=seq_idx)
        h5.attrs["total_frames"] = N

    meta = {
        "total_frames": N,
        "n_sequences":  3,
        "fingerprint":  "test",
        "created":      "2026-03-23",
        "h5_path":      h5p,
        "seq_id_to_int":{"seq_a":0,"seq_b":1,"seq_c":2},
        "split": {
            "train": ["seq_a"],
            "val":   ["seq_b"],
            "test":  ["seq_c"],
        },
    }
    with open(mp,"w") as f: json.dump(meta, f)

    ds = HDF5RehabDataset(h5p, mp, "train", augment=True)
    print(f"训练集窗口数: {len(ds)}")

    if len(ds) > 0:
        item = ds[0]
        for k,v in item.items():
            print(f"  {k}: {v.shape} {v.dtype}")
        assert item["pose_seq"].shape   == (5, 33, 3)
        assert item["bio_seq"].shape    == (20, 72)
        assert item["grf_future"].shape == (10, 12)
        assert item["grf_mask"].shape   == (10,)
        print("HDF5Dataset 验证 ✅")

    import shutil; shutil.rmtree(tmpd)