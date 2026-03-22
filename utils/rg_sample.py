"""
utils/rg_sample.py  (v3)
统一数据结构 - 字段名与 teacher_labeler 对齐（risk_label 而非 risk_level）
新增：grf_mask, physics_weight（物理残差权重）
"""
from __future__ import annotations
import numpy as np
from typing import Optional


class RGSample:
    """RehabGuardian 统一数据样本 v3"""

    def __init__(self):
        self.t:              Optional[float]      = None
        self.sequence_id:    Optional[str]        = None
        self.visual_2d:      Optional[np.ndarray] = None   # (33, 3)
        self.visual_seq:     Optional[np.ndarray] = None   # (5, 33, 3)
        # 生物力学层
        self.joint_angles:   Optional[np.ndarray] = None   # (23,) rad
        self.joint_vel:      Optional[np.ndarray] = None   # (23,) rad/s
        self.joint_acc:      Optional[np.ndarray] = None   # (23,) rad/s²
        self.markers:        Optional[np.ndarray] = None   # (28, 3) m
        # 物理层
        self.grf_left:       Optional[np.ndarray] = None   # (6,) N/N·m
        self.grf_right:      Optional[np.ndarray] = None   # (6,)
        self.grf_mask:       float                = 1.0    # GRF 置信度 0~1
        self.physics_weight: float                = 1.0    # 物理残差倒数权重
        self.com:            Optional[np.ndarray] = None   # (3,) m
        self.contact_left:   Optional[bool]       = None
        self.contact_right:  Optional[bool]       = None
        # 风险层（字段名与 teacher_labeler 一致）
        self.risk_label:     Optional[int]        = None   # 0/1/2
        self.risk_score:     Optional[float]      = None   # 0.0~1.0
        self.description:    Optional[str]        = None
        # 辅助
        self.valid:          bool                 = True
        self.original_t:     Optional[float]      = None

    @property
    def grf(self) -> Optional[np.ndarray]:
        """合并左右脚 GRF → (12,)"""
        if self.grf_left is not None and self.grf_right is not None:
            return np.concatenate([self.grf_left, self.grf_right])
        return None

    def is_complete(self) -> bool:
        return all(x is not None for x in [
            self.joint_angles, self.joint_vel, self.joint_acc,
            self.markers, self.grf_left, self.grf_right, self.com
        ]) and self.valid

    def copy(self) -> RGSample:
        new = RGSample()
        for attr in ['t', 'sequence_id', 'risk_label', 'risk_score', 'description',
                     'valid', 'original_t', 'contact_left', 'contact_right',
                     'grf_mask', 'physics_weight']:
            setattr(new, attr, getattr(self, attr))
        for attr in ['visual_2d', 'visual_seq', 'joint_angles', 'joint_vel',
                     'joint_acc', 'markers', 'grf_left', 'grf_right', 'com']:
            v = getattr(self, attr)
            setattr(new, attr, v.copy() if v is not None else None)
        return new

    def to_fno_feature(self) -> np.ndarray:
        """构建 72 维 FNO 特征 [ja|vel|acc|com]"""
        assert self.joint_angles is not None
        assert self.joint_vel    is not None
        assert self.joint_acc    is not None
        assert self.com          is not None
        feat = np.concatenate([
            self.joint_angles.astype(np.float32),
            self.joint_vel.astype(np.float32),
            self.joint_acc.astype(np.float32),
            self.com.astype(np.float32),
        ])
        assert feat.shape == (72,), f"FNO feature shape: {feat.shape}"
        return feat

    def to_risk_feature(self) -> np.ndarray:
        """构建 35 维 Risk 特征 [ja|grf]"""
        assert self.joint_angles is not None
        assert self.grf          is not None
        feat = np.concatenate([
            self.joint_angles.astype(np.float32),
            self.grf.astype(np.float32),
        ])
        assert feat.shape == (35,), f"Risk feature shape: {feat.shape}"
        return feat

    def __repr__(self) -> str:
        t_s = f"{self.t:.3f}s" if self.t is not None else "None"
        return (f"RGSample(t={t_s}, valid={self.valid}, seq={self.sequence_id}, "
                f"ja={self.joint_angles.shape if self.joint_angles is not None else None}, "
                f"grf_mask={self.grf_mask:.2f}, pw={self.physics_weight:.2f})")


if __name__ == "__main__":
    s = RGSample()
    s.t = 0.0; s.valid = True; s.sequence_id = "test"
    s.joint_angles = np.zeros(23, np.float32)
    s.joint_vel    = np.zeros(23, np.float32)
    s.joint_acc    = np.zeros(23, np.float32)
    s.markers      = np.zeros((28, 3), np.float32)
    s.grf_left     = np.array([0,0,350,0,0,0], np.float32)
    s.grf_right    = np.zeros(6, np.float32)
    s.com          = np.array([0., 1., 0.], np.float32)
    s.grf_mask     = 0.5
    s.physics_weight = 2.0
    s.risk_label   = 1

    assert s.is_complete()
    assert s.grf.shape == (12,)
    assert s.to_fno_feature().shape == (72,)
    assert s.to_risk_feature().shape == (35,)

    s2 = s.copy()
    s2.joint_angles[0] = 99
    assert s.joint_angles[0] == 0
    assert s2.risk_label == 1
    assert s2.grf_mask == 0.5
    print("rg_sample v3 OK ✅")
    print(s)