"""
utils/normalizer.py
Channel-wise z-score 归一化 + 空间去漂移
- 只从训练集 fit
- 空间去漂移：以序列第一帧 com(x,z) 为原点
"""
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from .rg_sample import RGSample

logger = logging.getLogger(__name__)
_EPS = 1e-8


class Normalizer:
    """Channel-wise z-score 归一化器"""

    FIELDS = [
        ("joint_angles", 23), ("joint_vel", 23), ("joint_acc", 23),
        ("markers_flat", 84), ("grf_left", 6),  ("grf_right", 6), ("com", 3),
    ]

    def __init__(self, remove_global_drift: bool = True):
        self.remove_global_drift = remove_global_drift
        self.stats: Dict[str, np.ndarray] = {}
        self._fitted = False

    # ------------------------------------------------------------------

    def _remove_drift(self, s: RGSample, ref: Optional[RGSample]) -> RGSample:
        r = ref if ref is not None else s
        dx = float(r.com[0]) if r.com is not None else 0.0
        dz = float(r.com[2]) if r.com is not None else 0.0
        if s.markers is not None:
            s.markers[:, 0] -= dx
            s.markers[:, 2] -= dz
        if s.com is not None:
            s.com[0] -= dx
            s.com[2] -= dz
        return s

    # ------------------------------------------------------------------

    def fit(self, samples: List[RGSample]) -> "Normalizer":
        buckets: Dict[str, List[np.ndarray]] = {n: [] for n, _ in self.FIELDS}

        for s in samples:
            if not s.valid:
                continue
            sc = s.copy()
            if self.remove_global_drift:
                self._remove_drift(sc, sc)

            def push(name, val, dim):
                if val is None:
                    return
                v = np.asarray(val, dtype=np.float32).flatten()
                if v.shape[0] == dim:
                    buckets[name].append(v)

            push("joint_angles", sc.joint_angles, 23)
            push("joint_vel",    sc.joint_vel,    23)
            push("joint_acc",    sc.joint_acc,    23)
            push("markers_flat", sc.markers,      84)
            push("grf_left",     sc.grf_left,      6)
            push("grf_right",    sc.grf_right,     6)
            push("com",          sc.com,           3)

        for name, dim in self.FIELDS:
            data = buckets[name]
            if len(data) == 0:
                logger.warning(f"Normalizer: 无 {name} 数据，使用默认 0/1")
                self.stats[f"{name}_mean"] = np.zeros(dim, np.float32)
                self.stats[f"{name}_std"]  = np.ones(dim, np.float32)
            else:
                arr = np.stack(data)
                if name == "markers_flat":
                    # Markers may have NaN for missing channels; use nanmean/nanstd
                    m = np.nanmean(arr, axis=0).astype(np.float32)
                    s = (np.nanstd(arr, axis=0) + _EPS).astype(np.float32)
                    bad = ~np.isfinite(m)
                    m[bad] = 0.0
                    s[bad] = 1.0
                    self.stats[f"{name}_mean"] = m
                    self.stats[f"{name}_std"]  = s
                else:
                    self.stats[f"{name}_mean"] = arr.mean(0).astype(np.float32)
                    self.stats[f"{name}_std"]  = (arr.std(0) + _EPS).astype(np.float32)

        self._fitted = True
        logger.info(f"Normalizer.fit 完成，样本数={len(samples)}")
        return self

    # ------------------------------------------------------------------

    def normalize(self, s: RGSample, ref: Optional[RGSample] = None) -> RGSample:
        assert self._fitted, "请先调用 fit()"
        sc = s.copy()
        if self.remove_global_drift and ref is not None:
            self._remove_drift(sc, ref)

        def norm(val, name, shape):
            if val is None:
                return None
            v = np.asarray(val, np.float32).flatten()
            v = (v - self.stats[f"{name}_mean"]) / self.stats[f"{name}_std"]
            return v.reshape(shape)

        sc.joint_angles = norm(sc.joint_angles, "joint_angles", (23,))
        sc.joint_vel    = norm(sc.joint_vel,    "joint_vel",    (23,))
        sc.joint_acc    = norm(sc.joint_acc,    "joint_acc",    (23,))
        sc.markers      = norm(sc.markers,      "markers_flat", (28, 3))
        sc.grf_left     = norm(sc.grf_left,     "grf_left",     (6,))
        sc.grf_right    = norm(sc.grf_right,    "grf_right",    (6,))
        sc.com          = norm(sc.com,           "com",          (3,))
        return sc

    def denormalize_grf(self, grf: np.ndarray) -> np.ndarray:
        """将归一化 GRF (12,) 还原为物理单位"""
        left  = grf[:6]  * self.stats["grf_left_std"]  + self.stats["grf_left_mean"]
        right = grf[6:]  * self.stats["grf_right_std"] + self.stats["grf_right_mean"]
        return np.concatenate([left, right]).astype(np.float32)

    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez(path, **self.stats)
        logger.info(f"归一化参数保存到 {path}")

    def load(self, path: str) -> "Normalizer":
        data = np.load(path)
        self.stats = {k: data[k] for k in data.files}
        self._fitted = True
        logger.info(f"归一化参数从 {path} 加载")
        return self

    def __repr__(self) -> str:
        return f"Normalizer(drift={self.remove_global_drift}, fitted={self._fitted})"