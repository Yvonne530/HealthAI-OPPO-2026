"""
utils/teacher_labeler.py  (v2 - 修复 medium=0 bug)

根本原因：
  规则3（膝关节不对称 > 15°）原设置 score=0.70 → 直接触发 HIGH 阈值(≥0.7)
  步行时左右膝交替运动导致不对称极为普遍 → 几乎所有帧被标为 HIGH

修复方案：
  1. 规则3：增加"站立期门控"（GRF > 100N），摆动期不触发
  2. 规则3：score 从 0.70 → 0.55（MEDIUM 范围）
  3. GRF 评分改为连续线性模型：
     0~0.5BW  → score 0~0.35    （摆动/轻接触 → LOW）
     0.5~1.5BW → score 0.35~0.60 （正常步行站立 → MEDIUM）
     1.5~2.5BW → score 0.60~0.85 （高冲击 → MEDIUM→HIGH）
     >2.5BW   → score 0.85       （极端冲击 → HIGH）
  4. MEDIUM 阈值从 0.40 降至 0.35（确保正常步行站立被标为 MEDIUM）

期望标签分布（Camargo2021 步行数据集）：
  LOW:    ~35%（摆动期 + 轻接触）
  MEDIUM: ~50%（正常步行站立期）
  HIGH:   ~15%（冲击峰值 + 膝关节异常）

文献依据：
  - Hewett 2005: 膝关节过伸为 ACL 损伤独立预测因子
  - Myer 2010:   双侧落地不对称与 ACL 损伤相关（仅在负重时适用）
  - Winter 2009: 正常步行 GRF 约 1.0-1.2 BW
"""
import math
import logging
from typing import List, Tuple

import numpy as np
from .rg_sample import RGSample

logger = logging.getLogger(__name__)

# ---- 关节索引 ----
KNEE_L_IDX  = 6   # knee_angle_l (OpenSim gait2392)
KNEE_R_IDX  = 13  # knee_angle_r
GRF_LEFT_Z  = 2   # grf_left[2]  = Fz 左脚垂直力
GRF_RIGHT_Z = 8   # grf_right[2] = Fz 右脚垂直力 (index 6+2)

# ---- 膝关节阈值 ----
KNEE_OVEREXT_RAD = math.radians(-5.0)    # 过伸：< -5° → HIGH
KNEE_MAX_RAD     = math.radians(120.0)   # 过度屈曲：> 120° → MEDIUM-HIGH
KNEE_ASYMM_RAD   = math.radians(20.0)   # 不对称阈值提高到 20°（15°在步行中太常见）

# ---- GRF 阈值（连续线性模型）----
BODY_WEIGHT_N = 70.0 * 9.81   # 参考体重（N）= 686.7N
GRF_STANCE_GATE = 100.0        # > 100N 才认为处于站立期
GRF_SCALE_LOW   = 0.5          # < 0.5 BW = 摆动/轻接触 → LOW
GRF_SCALE_MED   = 1.5          # 0.5~1.5 BW = 正常步行 → MEDIUM
GRF_SCALE_HIGH  = 2.5          # > 2.5 BW = 高冲击 → HIGH

# ---- 标签阈值 ----
THRESH_HIGH = 0.70   # score >= 0.70 → HIGH
THRESH_MED  = 0.35   # score >= 0.35 → MEDIUM（降低，确保正常步行站立被标为MEDIUM）
SMOOTH_WINDOW = 7    # 平滑窗口（增大到 7 帧，步态标签更稳定）


class TeacherLabeler:
    """
    基于生物力学规则的教师打标器（固定阈值版，用于训练时打伪标签）
    """

    def __init__(
        self,
        knee_overext_rad:  float = KNEE_OVEREXT_RAD,
        knee_max_rad:      float = KNEE_MAX_RAD,
        knee_asymm_rad:    float = KNEE_ASYMM_RAD,
        body_weight_n:     float = BODY_WEIGHT_N,
        smooth_window:     int   = SMOOTH_WINDOW,
    ):
        self.knee_overext_rad = knee_overext_rad
        self.knee_max_rad     = knee_max_rad
        self.knee_asymm_rad   = knee_asymm_rad
        self.body_weight_n    = body_weight_n
        self.smooth_window    = smooth_window

    def label_frame(
        self, joint_angles: np.ndarray, grf: np.ndarray
    ) -> Tuple[int, float]:
        """
        单帧打标（在 preprocess.py 中逐帧调用）。

        Returns:
            label: 0=low, 1=medium, 2=high
            score: 0.0~1.0 连续风险分数
        """
        assert joint_angles.shape == (23,), f"joint_angles shape: {joint_angles.shape}"
        assert grf.shape[0] == 12, f"grf shape: {grf.shape}"

        knee_l = joint_angles[KNEE_L_IDX]
        knee_r = joint_angles[KNEE_R_IDX]

        # 垂直 GRF 合力（左 + 右）
        grf_z = abs(grf[GRF_LEFT_Z]) + abs(grf[GRF_RIGHT_Z])

        score = 0.0

        # ----------------------------------------------------------------
        # 规则1：膝关节过伸（< -5°）← 最高风险
        # 依据：Hewett 2005 — 过伸是 ACL 损伤最强独立预测因子
        # ----------------------------------------------------------------
        if knee_l < self.knee_overext_rad or knee_r < self.knee_overext_rad:
            score = max(score, 0.92)

        # ----------------------------------------------------------------
        # 规则2：膝关节过度屈曲（> 120°）
        # ----------------------------------------------------------------
        elif knee_l > self.knee_max_rad or knee_r > self.knee_max_rad:
            score = max(score, 0.78)

        # ----------------------------------------------------------------
        # 规则3：左右膝不对称 + 站立期门控
        # FIX v2: score 从 0.70 → 0.55，且只在负重时触发（GRF > 100N）
        # 摆动期左右膝固然不对称，但此时不承重，无 ACL 损伤风险
        # ----------------------------------------------------------------
        asymm = abs(knee_l - knee_r)
        if asymm > self.knee_asymm_rad and grf_z > GRF_STANCE_GATE:
            score = max(score, 0.55)

        # ----------------------------------------------------------------
        # 规则4：GRF 连续线性评分（FIX v2: 不再二值化）
        # 物理依据：Winter 2009 正常步行 GRF ~1.0-1.2 BW
        # 归一化到参考体重（70kg）
        # ----------------------------------------------------------------
        grf_bw = grf_z / self.body_weight_n   # 体重倍数

        if grf_bw > GRF_SCALE_HIGH:           # > 2.5 BW: 极端冲击
            score = max(score, 0.85)
        elif grf_bw > GRF_SCALE_MED:          # 1.5~2.5 BW: 高冲击
            frac  = (grf_bw - GRF_SCALE_MED) / (GRF_SCALE_HIGH - GRF_SCALE_MED)
            score = max(score, 0.60 + 0.25 * float(frac))   # 0.60~0.85
        elif grf_bw > GRF_SCALE_LOW:          # 0.5~1.5 BW: 正常站立
            frac  = (grf_bw - GRF_SCALE_LOW) / (GRF_SCALE_MED - GRF_SCALE_LOW)
            score = max(score, 0.35 + 0.25 * float(frac))   # 0.35~0.60
        # else: < 0.5 BW（摆动期），score 保持接近 0

        # ----------------------------------------------------------------
        # 规则5：膝关节接近过伸软阈值
        # ----------------------------------------------------------------
        knee_risk = max(
            float(np.clip((self.knee_overext_rad - knee_l) / math.radians(8), 0, 1)),
            float(np.clip((self.knee_overext_rad - knee_r) / math.radians(8), 0, 1)),
        )
        score = max(score, knee_risk * 0.55)

        # ---- 三分类映射 ----
        if score >= THRESH_HIGH:
            label = 2
        elif score >= THRESH_MED:
            label = 1
        else:
            label = 0

        return label, float(score)

    def label_sequence(
        self, samples: List[RGSample]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """对样本序列打标 + 平滑（仅在非 preprocess 路径使用）"""
        labels = np.zeros(len(samples), dtype=np.int64)
        scores = np.zeros(len(samples), dtype=np.float32)

        for i, s in enumerate(samples):
            if s.joint_angles is None or s.grf is None:
                continue
            labels[i], scores[i] = self.label_frame(s.joint_angles, s.grf)

        # 移动平均平滑
        kernel         = np.ones(self.smooth_window) / self.smooth_window
        scores_smooth  = np.convolve(scores, kernel, mode='same').astype(np.float32)
        labels_smooth  = np.where(
            scores_smooth >= THRESH_HIGH, 2,
            np.where(scores_smooth >= THRESH_MED, 1, 0)
        ).astype(np.int64)

        for i, s in enumerate(samples):
            s.risk_label = int(labels_smooth[i])
            s.risk_score = float(scores_smooth[i])

        dist = np.bincount(labels_smooth, minlength=3)
        logger.info(
            f"TeacherLabeler: {len(samples)} 帧 | "
            f"低={dist[0]} 中={dist[1]} 高={dist[2]}"
        )
        return labels_smooth, scores_smooth

    def dynamic_threshold_from_heart_rate(
        self, heart_rate: float, base_grf_high: float = None
    ) -> float:
        """根据心率动态调整 GRF 风险阈值（推理阶段使用）"""
        if base_grf_high is None:
            base_grf_high = GRF_SCALE_HIGH * self.body_weight_n
        hr_factor = 1.0 - 0.2 * np.clip((heart_rate - 80) / 40, -1, 1)
        return float(base_grf_high * hr_factor)


if __name__ == "__main__":
    import math
    labeler = TeacherLabeler()

    print("=== TeacherLabeler v2 验证 ===")
    print(f"MEDIUM 阈值: {THRESH_MED}  HIGH 阈值: {THRESH_HIGH}")
    print()

    scenarios = [
        ("摆动期",         0, math.radians(15), math.radians(50)),
        ("站立中期",      750, math.radians(15), math.radians(10)),
        ("站立峰值",      850, math.radians(20), math.radians(15)),
        ("快步行走",     1000, math.radians(15), math.radians(45)),
        ("上楼梯峰值",   1300, math.radians(25), math.radians(60)),
        ("高冲击",       1800, math.radians(25), math.radians(50)),
        ("膝关节过伸",    700, math.radians(-8), math.radians(30)),
    ]

    print(f"{'场景':<14} {'GRF':>6}N {'knee_L':>8} {'knee_R':>8} {'score':>6} {'标签'}")
    print("-" * 65)
    for name, grf_z, kl, kr in scenarios:
        ja  = np.zeros(23, np.float32)
        ja[KNEE_L_IDX] = kl; ja[KNEE_R_IDX] = kr
        grf = np.zeros(12, np.float32)
        grf[GRF_LEFT_Z] = grf_z * 0.6   # 单足支撑约 60%
        grf[GRF_RIGHT_Z] = grf_z * 0.4
        lbl, sc = labeler.label_frame(ja, grf)
        lbls = ['LOW', 'MED', 'HIGH'][lbl]
        print(f"{name:<14} {grf_z:>6.0f}N "
              f"{math.degrees(kl):>6.0f}°  {math.degrees(kr):>6.0f}° "
              f"{sc:>6.3f} {lbls}")

    # 验证关键场景
    ja_norm = np.zeros(23, np.float32)
    ja_norm[KNEE_L_IDX] = math.radians(20); ja_norm[KNEE_R_IDX] = math.radians(22)
    grf_norm = np.zeros(12, np.float32); grf_norm[GRF_LEFT_Z] = 450; grf_norm[GRF_RIGHT_Z] = 300
    lbl_n, sc_n = labeler.label_frame(ja_norm, grf_norm)
    assert lbl_n == 1, f"正常步行应为 MEDIUM，得 {lbl_n}"
    print(f"\n正常步行站立期: label={lbl_n} (MEDIUM) score={sc_n:.3f} ✅")

    ja_bad = np.zeros(23, np.float32)
    ja_bad[KNEE_L_IDX] = math.radians(-8)
    grf_bad = np.zeros(12, np.float32); grf_bad[GRF_LEFT_Z] = 400
    lbl_b, sc_b = labeler.label_frame(ja_bad, grf_bad)
    assert lbl_b == 2, f"膝过伸应为 HIGH，得 {lbl_b}"
    print(f"膝关节过伸:     label={lbl_b} (HIGH)   score={sc_b:.3f} ✅")

    print("\nteacher_labeler v2 OK ✅")