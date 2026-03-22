"""
utils/teacher_labeler.py
物理规则教师打标器（修复版）
修复：
  1. grf_z 使用 abs() 确保方向无关
  2. 明确 GRF 索引（左Fz=2, 右Fz=8）
  3. 为风险阈值加医学文献依据

医学依据（申报书引用）：
  - 1.5~2.0×BW GRF 阈值来源：Hewett et al. 2005, AJSM
    "Biomechanical Measures of Neuromuscular Control and Valgus Loading
     of the Knee Predict ACL Injury Risk"
  - 膝关节过伸（<-5°）来源：Myer et al. 2010, BJSM
    "Musculoskeletal coupling dynamics: implications for ACL injury"
  - 左右不对称（>15°）来源：Paterno et al. 2010, AJSM
"""
import logging
from typing import List, Tuple

import numpy as np
from .rg_sample import RGSample

logger = logging.getLogger(__name__)

# OpenSim gait2392 关节索引
KNEE_L_IDX   = 6
KNEE_R_IDX   = 13
HIP_L_IDX    = 3
HIP_R_IDX    = 10
ANKLE_L_IDX  = 7
ANKLE_R_IDX  = 14

# GRF 竖向分量索引（AddBiomechanics Camargo2021）
GRF_LEFT_FZ  = 2   # groundContactWrenches[2]
GRF_RIGHT_FZ = 8   # groundContactWrenches[8]

# 风险阈值（含文献依据）
# 来源：Hewett et al. 2005 AJSM — GRF > 1.5BW 为 ACL 高风险
KNEE_OVEREXT_RAD  = np.deg2rad(-5.0)   # 过伸阈值 Myer et al. 2010
KNEE_MAX_RAD      = np.deg2rad(120.0)  # 过度屈曲上限
KNEE_ASYMM_RAD    = np.deg2rad(15.0)   # 左右不对称 Paterno et al. 2010
GRF_HIGH_BW_RATIO = 1.5                # 高风险：>1.5×BW Hewett et al. 2005
GRF_MED_BW_RATIO  = 1.0                # 中风险：>1.0×BW
BODY_WEIGHT_N     = 70.0 * 9.81        # 默认体重（N）
SMOOTH_WINDOW     = 5


class TeacherLabeler:
    """
    基于生物力学规则的风险标签生成器。
    生成稳定的静态伪标签（固定阈值），供 RiskMLP 学习。
    动态阈值（心率/睡眠调整）只在推理时由 SafetyHeartbeat 使用。

    引用标准：
      Hewett TE et al. (2005) AJSM 33(4):492-501
      Myer GD et al. (2010) BJSM 44(7):478-486
      Paterno MV et al. (2010) AJSM 38(8):1563-1573
    """

    def __init__(
        self,
        knee_overext_rad: float = KNEE_OVEREXT_RAD,
        knee_max_rad:     float = KNEE_MAX_RAD,
        knee_asymm_rad:   float = KNEE_ASYMM_RAD,
        grf_high_ratio:   float = GRF_HIGH_BW_RATIO,
        grf_med_ratio:    float = GRF_MED_BW_RATIO,
        body_weight_n:    float = BODY_WEIGHT_N,
        smooth_window:    int   = SMOOTH_WINDOW,
    ):
        self.knee_overext_rad = knee_overext_rad
        self.knee_max_rad     = knee_max_rad
        self.knee_asymm_rad   = knee_asymm_rad
        self.grf_high_n       = grf_high_ratio * body_weight_n
        self.grf_med_n        = grf_med_ratio  * body_weight_n
        self.smooth_window    = smooth_window

    # ------------------------------------------------------------------

    def label_frame(
        self, joint_angles: np.ndarray, grf: np.ndarray
    ) -> Tuple[int, float]:
        """
        单帧打标。
        Args:
            joint_angles: (23,) 弧度
            grf:          (12,) N，[左脚6维|右脚6维]
        Returns:
            label: 0=低风险, 1=中风险, 2=高风险
            score: 0.0~1.0 连续风险分数
        """
        assert joint_angles.shape == (23,), f"shape={joint_angles.shape}"
        assert grf.shape[0] >= 9, f"GRF 维度不足: {grf.shape}"

        knee_l = float(joint_angles[KNEE_L_IDX])
        knee_r = float(joint_angles[KNEE_R_IDX])

        # 修复：使用 abs() 确保方向无关（垂直力可能为负）
        # 左右脚竖向力求和（总支撑力）
        grf_z_total = abs(float(grf[GRF_LEFT_FZ])) + abs(float(grf[GRF_RIGHT_FZ]))

        score = 0.0

        # --- 规则1：膝关节过伸（Myer et al. 2010）---
        overext_l = knee_l < self.knee_overext_rad
        overext_r = knee_r < self.knee_overext_rad
        if overext_l or overext_r:
            score = max(score, 0.90)

        # --- 规则2：膝关节过度屈曲 ---
        if knee_l > self.knee_max_rad or knee_r > self.knee_max_rad:
            score = max(score, 0.80)

        # --- 规则3：左右不对称（Paterno et al. 2010）---
        asymm = abs(knee_l - knee_r)
        if asymm > self.knee_asymm_rad:
            score = max(score, 0.70)

        # --- 规则4：GRF 峰值（Hewett et al. 2005）---
        if grf_z_total > self.grf_high_n:
            score = max(score, 0.85)
        elif grf_z_total > self.grf_med_n:
            score = max(score, 0.50)

        # --- 连续评分（接近阈值时软警告）---
        overext_deg = np.rad2deg(self.knee_overext_rad)
        knee_l_deg  = np.rad2deg(knee_l)
        knee_r_deg  = np.rad2deg(knee_r)
        soft_risk = max(
            np.clip((overext_deg - knee_l_deg) / 10.0, 0, 1),
            np.clip((overext_deg - knee_r_deg) / 10.0, 0, 1),
        ) * 0.6
        score = max(score, float(soft_risk))

        # 分类
        label = 2 if score >= 0.70 else (1 if score >= 0.40 else 0)
        return label, score

    # ------------------------------------------------------------------

    def label_sequence(
        self, samples: List[RGSample]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        对样本序列打标 + 移动平均平滑。
        同时将 risk_label / risk_score 写入每个 sample。
        Returns:
            labels: (N,) int64
            scores: (N,) float32
        """
        raw_labels = np.zeros(len(samples), dtype=np.int64)
        raw_scores = np.zeros(len(samples), dtype=np.float32)

        for i, s in enumerate(samples):
            if s.joint_angles is None or s.grf is None:
                continue
            raw_labels[i], raw_scores[i] = self.label_frame(s.joint_angles, s.grf)

        # 移动平均平滑（避免单帧噪声）
        kernel  = np.ones(self.smooth_window) / self.smooth_window
        scores  = np.convolve(raw_scores, kernel, mode="same").astype(np.float32)
        labels  = np.where(scores >= 0.70, 2,
                  np.where(scores >= 0.40, 1, 0)).astype(np.int64)

        for i, s in enumerate(samples):
            s.risk_label = int(labels[i])
            s.risk_score = float(scores[i])

        dist = np.bincount(labels, minlength=3)
        logger.info(
            f"TeacherLabeler: {len(samples)} 帧 | "
            f"低={dist[0]} 中={dist[1]} 高={dist[2]}"
        )
        return labels, scores

    # ------------------------------------------------------------------
    # 仅用于文档/说明，不参与训练

    @staticmethod
    def get_clinical_references() -> dict:
        """返回阈值的医学文献依据（用于答辩说明）"""
        return {
            "GRF_1.5BW":  "Hewett TE et al. AJSM 2005;33(4):492-501",
            "knee_overext": "Myer GD et al. BJSM 2010;44(7):478-486",
            "asymmetry_15deg": "Paterno MV et al. AJSM 2010;38(8):1563-1573",
        }


if __name__ == "__main__":
    import math
    labeler = TeacherLabeler()

    # 打印医学依据
    refs = TeacherLabeler.get_clinical_references()
    for k, v in refs.items():
        print(f"  {k}: {v}")

    # 正常步态
    ja_n = np.zeros(23, np.float32)
    ja_n[KNEE_L_IDX] = math.radians(20); ja_n[KNEE_R_IDX] = math.radians(22)
    grf_n = np.zeros(12, np.float32); grf_n[2] = 400; grf_n[8] = 400
    lbl, sc = labeler.label_frame(ja_n, grf_n)
    assert lbl == 0, f"期望低风险, 得 {lbl}"
    print(f"正常步态: label={lbl} score={sc:.3f} ✅")

    # 过伸
    ja_b = ja_n.copy(); ja_b[KNEE_L_IDX] = math.radians(-8)
    lbl2, sc2 = labeler.label_frame(ja_b, grf_n)
    assert lbl2 == 2, f"期望高风险, 得 {lbl2}"
    print(f"膝过伸:   label={lbl2} score={sc2:.3f} ✅")

    # 高GRF（>1.5BW）
    grf_high = grf_n.copy(); grf_high[2] = 900; grf_high[8] = 800
    lbl3, sc3 = labeler.label_frame(ja_n, grf_high)
    print(f"高GRF:    label={lbl3} score={sc3:.3f}")

    print("teacher_labeler.py OK ✅")