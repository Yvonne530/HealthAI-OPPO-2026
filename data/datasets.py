"""
data/datasets.py  (v2 - 统一 E2E Dataset)
核心改进：
  - E2EDataset：单一数据集返回所有任务所需数据
    (pose_seq, ja_gt, markers_gt, grf_future, risk_label, grf_mask)
  - 支持数据增强（高斯噪声、随机遮挡、随机旋转）
  - 按 trial 划分，防止数据泄露
"""
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# 维度常量
N_JOINTS       = 23
N_MARKERS_FLAT = 84    # 28×3
GRF_DIM        = 12
RISK_CLASSES   = 3
NUM_VISUAL     = 33
FUTURE_K       = 10    # 与 FNO 对齐


def _split_by_sequence(
    samples,
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
    split: str         = "train",
) -> list:
    """按 sequence_id 分组后划分，避免同 trial 跨 split"""
    seq_map: Dict[str, list] = {}
    for s in samples:
        key = s.sequence_id or "__default__"
        seq_map.setdefault(key, []).append(s)

    all_ids = sorted(seq_map.keys())
    n = len(all_ids)
    tr_end = int(n * train_ratio)
    va_end = int(n * (train_ratio + val_ratio))

    chosen = {"train": all_ids[:tr_end],
              "val":   all_ids[tr_end:va_end],
              "test":  all_ids[va_end:]}[split]

    out = []
    for sid in chosen:
        out.extend(seq_map[sid])
    return out


# ======================================================================
# 数据增强（视觉骨骼）
# ======================================================================

class PoseAugmentor:
    """
    对视觉骨骼序列 (T, 33, 3) 进行数据增强
    模拟手机摄像头的真实误差：
      1. 高斯噪声（传感器噪声）
      2. 随机遮挡（视角遮挡，约 30% 概率）
      3. 小角度随机旋转（手机持握角度变化）
    """

    def __init__(
        self,
        noise_std:      float = 0.015,    # 1.5cm 噪声标准差
        occ_prob:       float = 0.30,     # 遮挡概率
        occ_frac:       float = 0.20,     # 遮挡节点比例
        rot_max_deg:    float = 10.0,     # 最大旋转角（度）
        enabled:        bool  = True,
    ):
        self.noise_std   = noise_std
        self.occ_prob    = occ_prob
        self.occ_frac    = occ_frac
        self.rot_max_rad = np.deg2rad(rot_max_deg)
        self.enabled     = enabled

    def __call__(self, pose_seq: np.ndarray) -> np.ndarray:
        """pose_seq: (T, 33, 3)"""
        if not self.enabled:
            return pose_seq

        out = pose_seq.copy()
        T, N, C = out.shape

        # 1. 高斯噪声
        out += np.random.randn(*out.shape).astype(np.float32) * self.noise_std

        # 2. 随机遮挡（整帧统一，模拟视角遮挡）
        if np.random.rand() < self.occ_prob:
            n_occ = max(1, int(N * self.occ_frac))
            occ_idx = np.random.choice(N, size=n_occ, replace=False)
            out[:, occ_idx, :] = 0.0

        # 3. 小角度旋转（绕 y 轴，矢状面内旋转）
        angle = np.random.uniform(-self.rot_max_rad, self.rot_max_rad)
        c, s  = np.cos(angle), np.sin(angle)
        R     = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
        out   = out @ R.T    # (T, N, 3) @ (3, 3)

        return out.astype(np.float32)


# ======================================================================
# 统一 E2E Dataset
# ======================================================================

class E2EDataset(Dataset):
    """
    端到端联合训练数据集

    每个样本包含：
      - pose_seq:    (T_vis=5, 33, 3)   视觉骨骼（含增强）
      - ja_gt:       (23,)               关节角度 GT
      - markers_gt:  (84,)               标记点 GT
      - grf_future:  (K=10, 12)          未来 K 帧 GRF GT
      - risk_label:  int                 风险等级 (0/1/2)
      - grf_mask:    (K,)                GRF 置信度 mask

    训练时：
      - STGCN 预测 pred_ja 来替代 pose_seq
      - FNO 接收 build_fno_features(pred_ja) 作为输入
      - 三模型联合 backward
    """

    def __init__(
        self,
        samples,
        split:       str   = "train",
        num_vis:     int   = 5,       # ST-GCN 时序窗口
        future_k:    int   = FUTURE_K,
        normalizer         = None,
        augmentor:   Optional[PoseAugmentor] = None,
        train_ratio: float = 0.70,
        val_ratio:   float = 0.15,
    ):
        self.num_vis  = num_vis
        self.future_k = future_k
        self.norm     = normalizer
        self.augment  = augmentor if augmentor is not None else PoseAugmentor(enabled=(split=="train"))

        raw = _split_by_sequence(samples, train_ratio, val_ratio, split)
        self._samples = raw

        # 滑窗索引：需要 num_vis 帧历史 + future_k 帧未来
        need = num_vis + future_k
        self.windows: List[int] = []
        for i in range(len(raw) - need):
            ids = set(raw[j].sequence_id for j in range(i, i + need))
            if len(ids) == 1:
                self.windows.append(i)

        logger.info(
            f"[E2EDataset/{split}] {len(raw)} 帧, "
            f"{len(self.windows)} 窗口 (vis={num_vis}, K={future_k})"
        )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start = self.windows[idx]
        hist  = self._samples[start: start + self.num_vis]   # 历史帧
        ref   = hist[0]                                        # 归一化参考

        # 归一化
        if self.norm and self.norm._fitted:
            hist_n = [self.norm.normalize(s, ref) for s in hist]
            future = [self.norm.normalize(self._samples[start + self.num_vis + k], ref)
                      for k in range(self.future_k)]
        else:
            hist_n = hist
            future = [self._samples[start + self.num_vis + k] for k in range(self.future_k)]

        # ---- 视觉骨骼序列 (T_vis, 33, 3) ----
        pose_frames = []
        for s in hist_n:
            mk = s.markers if s.markers is not None else np.zeros((28,3), np.float32)
            vis = self._markers_to_visual(mk)  # (33, 3)
            pose_frames.append(vis)
        pose_seq = np.stack(pose_frames, axis=0)          # (num_vis, 33, 3)
        pose_seq = self.augment(pose_seq)                 # 数据增强

        # ---- 当前帧 GT ----
        cur = hist_n[-1]   # 用最后一历史帧作为当前帧 GT
        ja_gt      = cur.joint_angles if cur.joint_angles is not None else np.zeros(N_JOINTS, np.float32)
        markers_gt = cur.markers.flatten() if cur.markers is not None else np.zeros(N_MARKERS_FLAT, np.float32)

        # ---- 未来 K 帧 GRF GT ----
        grf_future  = np.zeros((self.future_k, GRF_DIM), np.float32)
        grf_mask    = np.ones(self.future_k, np.float32)
        for k, fs in enumerate(future):
            g = fs.grf
            if g is not None:
                grf_future[k] = g.astype(np.float32)
            grf_mask[k] = float(getattr(fs, "grf_mask", 1.0))

        # ---- 风险标签 ----
        risk_label = int(getattr(cur, "risk_label", 0))
        assert 0 <= risk_label < 3

        # ---- shape 断言 ----
        assert pose_seq.shape    == (self.num_vis, 33, 3),      f"{pose_seq.shape}"
        assert ja_gt.shape       == (N_JOINTS,),                f"{ja_gt.shape}"
        assert markers_gt.shape  == (N_MARKERS_FLAT,),          f"{markers_gt.shape}"
        assert grf_future.shape  == (self.future_k, GRF_DIM),   f"{grf_future.shape}"

        return {
            "pose_seq":    torch.from_numpy(pose_seq).float(),
            "ja_gt":       torch.from_numpy(ja_gt.astype(np.float32)),
            "markers_gt":  torch.from_numpy(markers_gt.astype(np.float32)),
            "grf_future":  torch.from_numpy(grf_future),
            "grf_mask":    torch.from_numpy(grf_mask),
            "risk_label":  torch.tensor(risk_label, dtype=torch.long),
        }

    @staticmethod
    def _markers_to_visual(markers: np.ndarray) -> np.ndarray:
        """
        将 28 个标记点映射到 33 个 MediaPipe 骨架点。
        直接用标记点坐标，不足的用零填充。
        """
        visual = np.zeros((33, 3), dtype=np.float32)
        n = min(markers.shape[0], 33)
        valid = ~np.any(np.isnan(markers[:n]), axis=-1)
        visual[:n][valid] = markers[:n][valid]
        return visual


# ======================================================================
# 保留单任务 Dataset（兼容性）
# ======================================================================

class FNODataset(Dataset):
    """单独训练 FNO 时使用（无 pose 输入）"""

    def __init__(self, samples, split="train", seq_len=20, future_k=FUTURE_K,
                 normalizer=None, train_ratio=0.70, val_ratio=0.15):
        self.seq_len   = seq_len
        self.future_k  = future_k
        self.norm      = normalizer
        raw = _split_by_sequence(samples, train_ratio, val_ratio, split)
        self._samples  = raw
        need = seq_len + future_k
        self.windows = [i for i in range(len(raw) - need)
                        if len(set(raw[j].sequence_id for j in range(i, i+need))) == 1]
        logger.info(f"[FNODataset/{split}] {len(self.windows)} 窗口")

    def __len__(self): return len(self.windows)

    def __getitem__(self, idx):
        start  = self.windows[idx]
        seq    = self._samples[start: start + self.seq_len]
        future = self._samples[start + self.seq_len: start + self.seq_len + self.future_k]
        ref    = seq[0]

        feats = np.zeros((self.seq_len, 72), np.float32)
        for i, s in enumerate(seq):
            sn = self.norm.normalize(s, ref) if self.norm and self.norm._fitted else s
            feats[i] = sn.to_fno_feature()

        grf_gt = np.zeros((self.future_k, GRF_DIM), np.float32)
        for k, fs in enumerate(future):
            fn = self.norm.normalize(fs, ref) if self.norm and self.norm._fitted else fs
            g  = fn.grf
            if g is not None:
                grf_gt[k] = g.astype(np.float32)

        return torch.from_numpy(feats), torch.from_numpy(grf_gt)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from utils.rg_sample import RGSample

    rng = np.random.default_rng(0)
    samples = []
    for i in range(400):
        s = RGSample()
        s.t = float(i)/60; s.sequence_id = "s_a" if i < 250 else "s_b"
        s.joint_angles = rng.normal(0,.3,23).astype(np.float32)
        s.joint_vel    = rng.normal(0,1.,23).astype(np.float32)
        s.joint_acc    = rng.normal(0,5.,23).astype(np.float32)
        s.markers      = rng.normal(0,.5,(28,3)).astype(np.float32)
        s.grf_left     = np.abs(rng.normal(300,100,6)).astype(np.float32)
        s.grf_right    = np.abs(rng.normal(300,100,6)).astype(np.float32)
        s.com          = rng.normal(0,1.,3).astype(np.float32)
        s.contact_left = True; s.contact_right = False; s.valid = True
        s.risk_label   = int(rng.integers(0,3))
        s.grf_mask     = 1.0
        samples.append(s)

    aug = PoseAugmentor(enabled=True)
    ds  = E2EDataset(samples, split="train", augmentor=aug)
    print(f"E2EDataset 大小: {len(ds)}")
    if len(ds) > 0:
        item = ds[0]
        for k, v in item.items():
            print(f"  {k}: {v.shape} {v.dtype}")
        assert item["pose_seq"].shape  == (5, 33, 3)
        assert item["ja_gt"].shape     == (23,)
        assert item["markers_gt"].shape== (84,)
        assert item["grf_future"].shape== (10, 12)
        assert item["grf_mask"].shape  == (10,)
        print("E2EDataset 验证 ✅")

    # DataLoader
    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=8, shuffle=True)
    batch = next(iter(dl))
    print(f"Batch: pose_seq={batch['pose_seq'].shape}")
    print("datasets.py v2 OK ✅")