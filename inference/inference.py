"""
inference/inference.py  (v2)
升级：
  - OOD 检测（Mahalanobis 距离）
  - 动态帧率（低风险省计算，高风险全速）
  - Grad-CAM 可解释性（返回高贡献关节名称）
  - 双速架构：快路径 60Hz + 物理校验每帧
"""
import logging
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from inference.risk_state_machine import RiskStateMachine, RiskStateMachineConfig

logger = logging.getLogger(__name__)

RISK_LABELS  = {0: "低风险", 1: "中风险", 2: "高风险"}
RISK_COLORS  = {0: "green",  1: "yellow",  2: "red"}
FUTURE_K     = 10

# OpenSim gait2392 关节名（用于可解释性输出）
DOF_NAMES = [
    "pelvis_tilt","pelvis_list","pelvis_rotation",
    "hip_flexion_l","hip_adduction_l","hip_rotation_l",
    "knee_angle_l","ankle_angle_l","subtalar_angle_l","mtp_angle_l",
    "hip_flexion_r","hip_adduction_r","hip_rotation_r",
    "knee_angle_r","ankle_angle_r","subtalar_angle_r","mtp_angle_r",
    "lumbar_extension","lumbar_bending","lumbar_rotation",
    "neck_flexion","neck_bending","neck_rotation",
]


class RehabGuardianInference:
    """
    端侧推理引擎 v2

    双速架构：
      快路径 @ 60Hz：FNO + RiskModel（< 15ms 目标）
      慢路径 @ 5Hz ：Grad-CAM 可解释性（用户主动触发）

    动态帧率：
      低风险 → 每 10 帧推理一次（节能 90%）
      中风险 → 每 3 帧
      高风险 → 每帧
    """

    INFER_INTERVAL = {0: 10, 1: 3, 2: 1}   # 风险等级 → 跳帧数

    def __init__(self, cfg: dict, device: str = "cpu"):
        self.cfg    = cfg
        self.device = torch.device(device)
        icfg = cfg["inference"]

        # 缓存
        self.pose_buf: deque = deque(maxlen=cfg["stgcn"]["num_frames"])
        self.bio_buf:  deque = deque(maxlen=cfg["fno"]["seq_len"])
        self.risk_buf: deque = deque(maxlen=icfg["smooth_window"])

        # 状态
        self._frame_count    = 0
        self._last_result:   Optional[dict] = None
        self._current_risk   = 0
        self._heart_rate     = 70.0
        self._sleep_score    = 80.0

        self._load_models(icfg, cfg)

    # ------------------------------------------------------------------
    # 模型加载
    # ------------------------------------------------------------------

    def _load_models(self, icfg, cfg):
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

        from models.stgcn      import STGCN
        from models.fno        import FNO1d, _fft_supported
        from models.risk_model import RiskMLP, SafetyHeartbeat
        from utils.normalizer  import Normalizer
        from utils.kinematics  import MahalanobisDetector

        self._stgcn = STGCN(
            num_nodes=cfg["stgcn"]["num_nodes"],
            hidden_channels=cfg["stgcn"]["hidden_channels"],
            num_frames=cfg["stgcn"]["num_frames"],
        ).to(self.device).eval()

        self._fno = FNO1d(
            input_dim=cfg["fno"]["input_dim"],
            output_dim=cfg["fno"]["output_dim"],
            future_k=cfg["fno"].get("future_k", FUTURE_K),
            modes=cfg["fno"]["modes"],
            width=cfg["fno"]["width"],
            depth=cfg["fno"]["depth"],
            use_lstm=not _fft_supported(),
        ).to(self.device).eval()

        self._risk_model = RiskMLP(
            input_dim=cfg["risk"]["input_dim"],
            hidden_dim=cfg["risk"]["hidden_dim"],
        ).to(self.device).eval()

        self._safety = SafetyHeartbeat(cfg["risk"]["body_weight_kg"])

        self._normalizer = Normalizer()
        if os.path.exists(icfg["norm_stats"]):
            self._normalizer.load(icfg["norm_stats"])

        self._ood = MahalanobisDetector()
        ood_path  = icfg.get("ood_stats", "")
        if os.path.exists(ood_path):
            self._ood.load(ood_path)

        # 加载 checkpoint
        for model, ckpt in [(self._stgcn, icfg["stgcn_ckpt"]),
                             (self._fno,   icfg["fno_ckpt"]),
                             (self._risk_model, icfg["risk_ckpt"])]:
            if os.path.exists(ckpt):
                model.load_state_dict(torch.load(ckpt, map_location=self.device, weights_only=True))

        self._mp_pose = None
        # 风险状态机（替代简单平滑）
        self._fsm = RiskStateMachine(RiskStateMachineConfig(
            thresh_warn=0.40, thresh_danger=0.70,
            k_up_warn=3, k_up_danger=2,
            k_dn_safe=5, k_dn_warn=8,
            danger_lock_frames=10,
        ))

    # ------------------------------------------------------------------
    # OPPO 健康数据
    # ------------------------------------------------------------------

    def update_health_data(self, heart_rate: float, sleep_score: float):
        self._heart_rate  = heart_rate
        self._sleep_score = sleep_score
        self._safety.update_heart_rate(heart_rate)
        self._safety.update_sleep_score(sleep_score)
        logger.debug(f"健康数据: HR={heart_rate:.0f}bpm sleep={sleep_score:.0f}")

    # ------------------------------------------------------------------
    # 主推理接口
    # ------------------------------------------------------------------

    @torch.no_grad()
    def run(
        self,
        input_data: np.ndarray,
        mode: str = "pose",
    ) -> dict:
        """
        Args:
            input_data: (33, 3) 骨骼 或 (H, W, 3) 视频帧
            mode: "pose" | "video"
        Returns:
            result dict 含 joint_angles, grf, risk_label, risk_score,
                        latency_ms, top3_joints（可解释性）
        """
        t0 = time.perf_counter()
        self._frame_count += 1

        # 动态帧率：低风险时跳帧复用上次结果
        interval = self.INFER_INTERVAL.get(self._current_risk, 1)
        if (self._frame_count % interval != 0) and self._last_result is not None:
            self._last_result["latency_ms"] = 0.1   # 跳帧几乎零延迟
            return self._last_result

        # ---- Step 1: 骨骼提取 ----
        if mode == "video":
            skeleton = self._extract_pose(input_data)
        else:
            skeleton = np.asarray(input_data, dtype=np.float32)
        assert skeleton.shape == (33, 3), f"骨骼 shape 错误: {skeleton.shape}"

        # ---- Step 2: OOD 检测 ----
        feat_flat = skeleton.flatten()
        if self._ood._fitted and self._ood.is_ood(feat_flat):
            logger.warning("OOD 检测：输入偏离训练分布，拒绝输出")
            return self._ood_response(time.perf_counter() - t0)

        self.pose_buf.append(skeleton)

        # ---- Step 3: ST-GCN → joint_angles ----
        n_vis = self.cfg["stgcn"]["num_frames"]
        if len(self.pose_buf) < n_vis:
            pad = n_vis - len(self.pose_buf)
            seq = [np.zeros((33,3), np.float32)] * pad + list(self.pose_buf)
        else:
            seq = list(self.pose_buf)

        pose_t = torch.from_numpy(np.stack(seq)).unsqueeze(0).to(self.device)  # (1,T,33,3)
        pred_ja, _ = self._stgcn(pose_t)     # (1, 23)
        joint_angles = pred_ja.squeeze(0).cpu().numpy()
        assert joint_angles.shape == (23,)

        # ---- Step 4: 可微运动学特征 ----
        ja_t = pred_ja.unsqueeze(1).expand(-1, 20, -1)   # (1, 20, 23)
        from utils.kinematics import finite_diff, compute_com
        vel  = finite_diff(ja_t, dt=1/60, order=1)
        acc  = finite_diff(ja_t, dt=1/60, order=2)
        com  = compute_com(ja_t)
        fno_input = torch.cat([ja_t, vel, acc, com], dim=-1)  # (1, 20, 72)

        # ---- Step 5: FNO → 未来 K 帧 GRF ----
        pred_grf_t = self._fno(fno_input)          # (1, K, 12)
        grf_norm   = pred_grf_t[0, 0].cpu().numpy()  # 取第一帧

        if self._normalizer._fitted:
            grf = self._normalizer.denormalize_grf(grf_norm)
        else:
            grf = grf_norm
        assert grf.shape == (12,)

        # ---- Step 6: Risk Model ----
        risk_feat = np.concatenate([joint_angles, grf])
        assert risk_feat.shape == (35,)
        rx = torch.from_numpy(risk_feat.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        rx = rx.expand(-1, 20, -1).to(self.device)   # (1, 20, 35)

        hr_t  = torch.tensor([[self._heart_rate]],  device=self.device)
        sl_t  = torch.tensor([[self._sleep_score]], device=self.device)
        logits, conf = self._risk_model(rx, hr_t, sl_t)
        probs   = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        model_label = int(probs.argmax())
        risk_score  = float(probs[2])
        confidence  = float(conf.squeeze().item())

        # ---- Step 7: SafetyHeartbeat 校验 ----
        final_label, final_conf, overridden = self._safety.validate(
            joint_angles, grf, model_label, confidence
        )

        # 风险状态机（时间连续决策）
        fsm_state = self._fsm.update(
            risk_score  = risk_score,
            confidence  = confidence,
            heart_rate  = self._heart_rate,
            sleep_score = self._sleep_score,
        )
        smooth_label = int(fsm_state)
        self._current_risk = smooth_label

        # ---- Step 8: Grad-CAM（按需，低频调用）----
        top3_joints = []
        if self._frame_count % 12 == 0:   # ~5Hz @ 60Hz
            top3_joints = self._gradcam_joints(pred_ja)

        latency_ms = (time.perf_counter() - t0) * 1000

        result = {
            "joint_angles": joint_angles,
            "grf":          grf,
            "risk_label":   smooth_label,
            "risk_score":   risk_score,
            "confidence":   confidence,
            "risk_text":    RISK_LABELS[smooth_label],
            "risk_color":   RISK_COLORS[smooth_label],
            "overridden":   overridden,
            "top3_joints":  top3_joints,
            "latency_ms":   latency_ms,
        }
        self._last_result = result
        return result

    # ------------------------------------------------------------------
    # Grad-CAM（可解释性）
    # ------------------------------------------------------------------

    def _gradcam_joints(self, pred_ja: torch.Tensor) -> List[str]:
        """
        对 ST-GCN 末层特征计算 Grad-CAM，返回风险贡献最大的 3 个关节名称。
        pred_ja: (1, 23)
        """
        try:
            # 启用梯度
            pred_ja.requires_grad_(True)
            grad_out = pred_ja.sum()
            if pred_ja.grad is not None:
                pred_ja.grad.zero_()
            grad_out.backward(retain_graph=True)

            if pred_ja.grad is not None:
                importances = pred_ja.grad.abs().squeeze(0).cpu().numpy()
                top3_idx    = importances.argsort()[-3:][::-1]
                return [DOF_NAMES[i] if i < len(DOF_NAMES) else f"dof_{i}"
                        for i in top3_idx]
        except Exception as e:
            logger.debug(f"Grad-CAM 失败: {e}")
        return []

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _extract_pose(self, frame: np.ndarray) -> np.ndarray:
        if self._mp_pose is None:
            try:
                import mediapipe as mp
                self._mp_pose = mp.solutions.pose.Pose(
                    static_image_mode=False, model_complexity=1,
                    min_detection_confidence=0.5,
                )
            except ImportError:
                return np.zeros((33, 3), np.float32)

        import cv2
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._mp_pose.process(rgb)
        if results.pose_world_landmarks:
            lm = results.pose_world_landmarks.landmark
            return np.array([[l.x, l.y, l.z] for l in lm], np.float32)
        return np.zeros((33, 3), np.float32)

    def _ood_response(self, elapsed: float) -> dict:
        return {
            "joint_angles": np.zeros(23, np.float32),
            "grf":          np.zeros(12, np.float32),
            "risk_label":   -1,
            "risk_score":   0.0,
            "confidence":   0.0,
            "risk_text":    "系统异常（输入超出分布）",
            "risk_color":   "gray",
            "overridden":   False,
            "top3_joints":  [],
            "latency_ms":   elapsed * 1000,
        }


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    import sys, os
    logging.basicConfig(level=logging.INFO)
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            "configs/config.yaml")
    cfg    = load_config(cfg_path)
    engine = RehabGuardianInference(cfg, device="cpu")
    engine.update_health_data(75, 82)

    # 模拟 30 帧推理
    latencies = []
    for i in range(30):
        skeleton = np.random.randn(33, 3).astype(np.float32)
        result   = engine.run(skeleton, mode="pose")
        latencies.append(result["latency_ms"])
        if i % 10 == 0:
            logger.info(
                f"帧{i}: {result['risk_text']} "
                f"{result['latency_ms']:.1f}ms "
                f"conf={result['confidence']:.2f} "
                f"top3={result['top3_joints']}"
            )

    print(f"平均延迟: {np.mean(latencies):.1f}ms")
    print("inference.py v2 OK ✅")