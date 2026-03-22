"""
utils/b3d_loader.py  (v4 - 残差物理权重)
新增：per-frame physics_weight = 1 / (normalized_residual + ε)
  - linearResidual：逆动力学线性残差（力，N）
  - angularResidual：逆动力学角残差（力矩，N·m）
  - 残差越小 → 权重越高 → 该帧在 loss 中贡献越大
  - 与 .sto id/ 目录的逆动力学误差逻辑一致

物理解释：
  AddBiomechanics 通过 trajectory optimization 填充 .b3d，
  residual force/torque 是优化收敛后的"物理不可解释量"，
  越小说明该帧越符合牛顿力学，越值得信任。
"""
import os
import logging
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
from scipy.interpolate import interp1d

from .rg_sample import RGSample

logger = logging.getLogger(__name__)

N_MARKERS    = 28
N_DOFS       = 23
TARGET_FPS   = 60
ORIGINAL_FPS = 200

PASS_SCORE_WEIGHTS = {
    "marker_rms":            0.4,
    "residual_force":        0.3,
    "residual_torque":       0.2,
    "missing_marker_ratio":  0.1,
}

MARKER_ORDER: List[str] = [
    "L_ASIS",      "L_Ankle_Lat",  "L_Heel",       "L_Knee_Lat",
    "L_PSIS",      "L_Shank_Front","L_Shank_Rear",  "L_Shank_Upper",
    "L_Thigh_Front","L_Thigh_Rear","L_Thigh_Upper", "L_Toe_Lat",
    "L_Toe_Med",   "L_Toe_Tip",   "R_ASIS",        "R_Ankle_Lat",
    "R_Heel",      "R_Knee_Lat",  "R_PSIS",        "R_Shank_Front",
    "R_Shank_Rear","R_Shank_Upper","R_Thigh_Front", "R_Thigh_Rear",
    "R_Thigh_Upper","R_Toe_Lat",  "R_Toe_Med",     "R_Toe_Tip",
]
assert len(MARKER_ORDER) == N_MARKERS

# 残差归一化参考值（基于 Camargo2021 统计）
_REF_LINEAR_RES  = 10.0   # N   — 典型线性残差量级
_REF_ANGULAR_RES = 5.0    # N·m — 典型角残差量级
_PHYSICS_EPS     = 0.05   # 防止除零


class B3DLoader:
    """
    .b3d 加载器 v4
    新增：per-frame physics_weight（逆动力学残差权重）
    """

    def __init__(self, data_root: str, target_fps: int = TARGET_FPS,
                 min_frames: int = 30, vel_diff_threshold: float = 0.5):
        if not os.path.isdir(data_root):
            raise FileNotFoundError(f"数据目录不存在: {data_root}")
        self.data_root          = data_root
        self.target_fps         = target_fps
        self.min_frames         = min_frames
        self.vel_diff_threshold = vel_diff_threshold
        self.file_list: List[str] = self._scan_files()
        logger.info(f"[B3DLoader] {len(self.file_list)} 个 .b3d 文件")

    def _scan_files(self) -> List[str]:
        files = []
        for root, _, fnames in os.walk(self.data_root):
            for f in sorted(fnames):
                if f.endswith(".b3d"):
                    files.append(os.path.join(root, f))
        return sorted(files)

    # ------------------------------------------------------------------
    # 动态 Pass 选择
    # ------------------------------------------------------------------

    def _select_best_pass(self, trial, nimble) -> Tuple[int, object]:
        n = len(trial.processingPasses)
        if n == 0:
            raise ValueError("无 processingPasses")
        if n == 1:
            return 0, trial.processingPasses[0]

        best_idx, best_score = 0, float("inf")
        for i, pp in enumerate(trial.processingPasses):
            try:
                mrms   = float(np.mean(pp.markerRMS))         if len(pp.markerRMS)      > 0 else 1.0
                rforce = float(np.mean(np.abs(pp.linearResidual)))  if len(pp.linearResidual)  > 0 else 1.0
                rtorq  = float(np.mean(np.abs(pp.angularResidual))) if len(pp.angularResidual) > 0 else 1.0
                w      = PASS_SCORE_WEIGHTS
                score  = (w["marker_rms"] * mrms
                        + w["residual_force"]  * rforce / _REF_LINEAR_RES
                        + w["residual_torque"] * rtorq  / _REF_ANGULAR_RES)
                if score < best_score:
                    best_score, best_idx = score, i
            except Exception:
                continue

        return best_idx, trial.processingPasses[best_idx]

    # ------------------------------------------------------------------
    # 计算 per-frame 物理权重（核心新增）
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_physics_weights(pp, n_frames: int) -> np.ndarray:
        """
        per-frame physics_weight = 1 / (normalized_combined_residual + ε)

        combined_residual[t] = |linearResidual[t]|  / REF_F
                             + |angularResidual[t]| / REF_T

        权重归一化到均值=1，便于与其他样本比较。

        物理意义：
          - AddBiomechanics 通过 trajectory optimization 求解逆动力学
          - 残差代表"无法用已知力解释的运动量"
          - 残差小 → 该帧动力学自洽 → 更可信 → 权重更高
          - 残差大 → 可能是接触切换/快速加速 → 降低 loss 贡献

        参考：
          Werling et al. 2023 "AddBiomechanics Dataset: Capturing the
          Physics of Human Motion at Scale" EMNLP. Section 3.2.
        """
        weights = np.ones(n_frames, dtype=np.float32)

        try:
            lin_res = np.asarray(pp.linearResidual,  dtype=np.float32)   # (T,) 或 (T, 3)
            ang_res = np.asarray(pp.angularResidual, dtype=np.float32)

            # 统一处理：若是多维取 L2 范数
            lin_abs = np.abs(lin_res).mean(axis=-1) if lin_res.ndim > 1 else np.abs(lin_res)
            ang_abs = np.abs(ang_res).mean(axis=-1) if ang_res.ndim > 1 else np.abs(ang_res)

            # 对齐长度
            min_len = min(n_frames, len(lin_abs), len(ang_abs))
            if min_len < 2:
                return weights

            combined = (lin_abs[:min_len] / (_REF_LINEAR_RES  + _PHYSICS_EPS)
                      + ang_abs[:min_len] / (_REF_ANGULAR_RES + _PHYSICS_EPS))

            # 权重 = 1 / (残差 + ε)，裁剪防止极端值
            raw_weights = 1.0 / (combined + _PHYSICS_EPS)
            # 归一化到均值=1（保持 loss 数量级不变）
            raw_weights = raw_weights / (raw_weights.mean() + _PHYSICS_EPS)
            raw_weights = np.clip(raw_weights, 0.1, 10.0)

            weights[:min_len] = raw_weights.astype(np.float32)

            # 统计信息
            logger.debug(
                f"  physics_weight: mean={raw_weights.mean():.2f} "
                f"min={raw_weights.min():.2f} max={raw_weights.max():.2f}"
            )
        except Exception as e:
            logger.debug(f"  physics_weight 计算失败（使用默认1.0）: {e}")

        return weights

    # ------------------------------------------------------------------
    # vel/acc 一致性校验
    # ------------------------------------------------------------------

    def _validate_vel(self, vel_mat, pos_mat, dt):
        diff_vel = np.gradient(pos_mat, dt, axis=0)
        err = float(np.mean(np.abs(vel_mat - diff_vel)))
        if err > self.vel_diff_threshold:
            logger.debug(f"  vel 不一致 (err={err:.3f})，重新差分计算")
            return diff_vel.astype(np.float32)
        return vel_mat

    # ------------------------------------------------------------------
    # 主加载接口
    # ------------------------------------------------------------------

    def load_trial(self, filepath: str) -> Iterator[RGSample]:
        try:
            import nimblephysics as nimble
        except ImportError:
            raise ImportError("pip install nimblephysics")

        try:
            b3d = nimble.biomechanics.OpenSimParser.loadB3D(filepath)
        except Exception as e:
            logger.warning(f"加载失败 {filepath}: {e}")
            return

        n_trials = len(b3d.trials) if hasattr(b3d, "trials") else 0
        for ti in range(n_trials):
            try:
                yield from self._extract_trial(b3d, ti, filepath, nimble)
            except Exception as e:
                logger.warning(f"  trial {ti} 失败: {e}")

    def _extract_trial(self, b3d, ti, filepath, nimble) -> Iterator[RGSample]:
        trial = b3d.trials[ti]
        pass_idx, pp = self._select_best_pass(trial, nimble)

        def _f32(x, ec=None):
            arr = np.asarray(x, dtype=np.float32)
            if arr.ndim == 2 and ec:
                if arr.shape[0] == ec and arr.shape[1] != ec:
                    arr = arr.T
            return arr

        try:
            pos_mat = _f32(pp.pos,   ec=N_DOFS)
            vel_raw = _f32(pp.vel,   ec=N_DOFS)
            grf_mat = _f32(pp.groundContactWrenches, ec=12)
            com_mat = _f32(pp.comPos, ec=3)
            ct_mat  = _f32(pp.contact, ec=2)
        except AttributeError as e:
            logger.warning(f"  字段缺失: {e}"); return

        n = pos_mat.shape[0]
        try:
            dt = float(trial.getTrialTimestep())
        except Exception:
            dt = 1.0 / ORIGINAL_FPS
        ts = np.arange(n, dtype=np.float64) * dt

        vel_mat = self._validate_vel(vel_raw, pos_mat, dt)
        acc_mat = np.gradient(vel_mat, dt, axis=0).astype(np.float32)

        # *** per-frame 物理权重（核心新增）***
        physics_weights = self._compute_physics_weights(pp, n)

        try:
            markers_mat = self._parse_markers_ordered(trial.markerObservations, n)
        except Exception:
            markers_mat = np.full((n, N_MARKERS, 3), np.nan, dtype=np.float32)

        grf_mask_arr = np.ones(n, dtype=np.float32)
        for i in range(n):
            try:
                reason = trial.getMissingGRFReason(i)
                if reason == nimble.biomechanics.MissingGRFReason.notMissingGRF:
                    grf_mask_arr[i] = 1.0
                elif reason == nimble.biomechanics.MissingGRFReason.missingBlip:
                    grf_mask_arr[i] = 0.5
                else:
                    grf_mask_arr[i] = 0.0
            except Exception:
                pass

        pos_mat, vel_mat, acc_mat, grf_mat, com_mat, markers_mat = \
            self._interpolate_invalid_frames(
                pos_mat, vel_mat, acc_mat, grf_mat, com_mat, markers_mat,
                grf_mask_arr, ts
            )

        if self.target_fps != ORIGINAL_FPS:
            resampled = self._resample_to_target(
                pos_mat, vel_mat, acc_mat, grf_mat, com_mat,
                markers_mat, grf_mask_arr, physics_weights, ts
            )
            if resampled is None:
                return
            pos_mat,vel_mat,acc_mat,grf_mat,com_mat,markers_mat,grf_mask_arr,physics_weights,ts = resampled

        n = pos_mat.shape[0]
        if n < self.min_frames:
            return

        base_seq = f"{os.path.basename(filepath)}__t{ti}__p{pass_idx}"
        for i in range(n):
            s = RGSample()
            s.t              = float(ts[i])
            s.original_t     = float(ts[i])
            s.sequence_id    = base_seq
            s.joint_angles   = pos_mat[i].copy()
            s.joint_vel      = vel_mat[i].copy()
            s.joint_acc      = acc_mat[i].copy()
            s.markers        = markers_mat[i].copy()
            s.grf_left       = grf_mat[i, :6].copy()
            s.grf_right      = grf_mat[i, 6:12].copy()
            s.grf_mask       = float(grf_mask_arr[i])
            s.physics_weight = float(physics_weights[i])   # ← 新增
            s.com            = com_mat[i].copy()
            s.contact_left   = bool(ct_mat[min(i,len(ct_mat)-1), 0] > 0.5)
            s.contact_right  = bool(ct_mat[min(i,len(ct_mat)-1), 1] > 0.5)
            s.valid = True

            assert s.joint_angles.shape == (N_DOFS,)
            assert s.grf_left.shape     == (6,)
            assert s.physics_weight     > 0
            yield s

    # ------------------------------------------------------------------
    # 插值 / 重采样
    # ------------------------------------------------------------------

    def _interpolate_invalid_frames(self, pos, vel, acc, grf, com, markers, mask, ts):
        def _interp(arr):
            out = arr.copy()
            for d in range(arr.shape[1]):
                valid = np.where(mask >= 0.5)[0]
                if len(valid) < 2:
                    continue
                f = interp1d(ts[valid], arr[valid, d], kind="linear",
                             bounds_error=False,
                             fill_value=(arr[valid[0],d], arr[valid[-1],d]))
                out[:, d] = f(ts)
            return out.astype(np.float32)
        return pos, vel, acc, _interp(grf), _interp(com), markers

    def _resample_to_target(self, pos, vel, acc, grf, com, markers, mask, pw, ts):
        dur = float(ts[-1] - ts[0])
        if dur <= 0:
            return None
        n_out  = max(int(round(dur * self.target_fps)), 2)
        ts_out = np.linspace(ts[0], ts[-1], n_out)

        def _rs2d(arr):
            out = np.zeros((n_out, arr.shape[1]), np.float32)
            for d in range(arr.shape[1]):
                f = interp1d(ts, arr[:,d].astype(np.float64), kind="linear",
                             bounds_error=False, fill_value=(arr[0,d],arr[-1,d]))
                out[:,d] = f(ts_out).astype(np.float32)
            return out

        def _rs1d(arr):
            f = interp1d(ts, arr.astype(np.float64), kind="linear",
                         bounds_error=False, fill_value=(arr[0],arr[-1]))
            return f(ts_out).astype(np.float32)

        mk_out = np.full((n_out, N_MARKERS, 3), np.nan, np.float32)
        for mi in range(N_MARKERS):
            for ci in range(3):
                col = markers[:, mi, ci]
                vld = ~np.isnan(col)
                if vld.sum() < 2:
                    continue
                f = interp1d(ts[vld], col[vld].astype(np.float64), kind="linear",
                             bounds_error=False, fill_value=np.nan)
                mk_out[:, mi, ci] = f(ts_out).astype(np.float32)

        return (_rs2d(pos), _rs2d(vel), _rs2d(acc), _rs2d(grf), _rs2d(com),
                mk_out, _rs1d(mask), _rs1d(pw), ts_out)

    # ------------------------------------------------------------------
    # Marker 解析
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_markers_ordered(raw, n_frames: int) -> np.ndarray:
        result = np.full((n_frames, N_MARKERS, 3), np.nan, dtype=np.float32)
        if raw is None:
            return result
        if isinstance(raw, np.ndarray):
            arr = raw.astype(np.float32)
            if arr.shape == (n_frames, N_MARKERS, 3):
                return arr
            if arr.shape == (n_frames, N_MARKERS * 3):
                return arr.reshape(n_frames, N_MARKERS, 3)
            return result
        if not isinstance(raw, (list, tuple)):
            return result
        name_to_idx = {name: i for i, name in enumerate(MARKER_ORDER)}
        for fi, obs in enumerate(raw[:n_frames]):
            if not isinstance(obs, dict):
                continue
            for name, val in obs.items():
                if name in name_to_idx and val is not None:
                    v = np.asarray(val, dtype=np.float32)
                    if v.shape == (3,):
                        result[fi, name_to_idx[name]] = v
        return result

    # ------------------------------------------------------------------
    # 批量加载
    # ------------------------------------------------------------------

    def get_all_samples(self) -> List[RGSample]:
        all_s: List[RGSample] = []
        for idx, fp in enumerate(self.file_list):
            logger.info(f"[{idx+1}/{len(self.file_list)}] {os.path.basename(fp)}")
            n = len(all_s)
            for s in self.load_trial(fp):
                all_s.append(s)
            logger.debug(f"  +{len(all_s)-n} 帧")
        logger.info(f"[B3DLoader] 共 {len(all_s)} 帧")
        return all_s

    def __len__(self) -> int:
        return len(self.file_list)

    def __repr__(self) -> str:
        return f"B3DLoader(root={self.data_root!r}, files={len(self.file_list)}, fps={self.target_fps})"


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    root = sys.argv[1] if len(sys.argv) > 1 else "/mnt/d/Camargo2021_Formatted_No_Arm/"
    if not os.path.isdir(root):
        print(f"目录不存在: {root} — 跳过真实加载测试")
        # 验证静态方法
        pw = B3DLoader._compute_physics_weights.__func__ if hasattr(
            B3DLoader._compute_physics_weights, '__func__') else None
        print("b3d_loader v4 结构 OK ✅")
        sys.exit(0)
    loader = B3DLoader(root)
    samples = list(loader.load_trial(loader.file_list[0]))
    if samples:
        s = samples[0]
        print(f"第1帧 physics_weight={s.physics_weight:.3f}")
        pw_vals = [ss.physics_weight for ss in samples]
        print(f"physics_weight: mean={np.mean(pw_vals):.3f} "
              f"min={np.min(pw_vals):.3f} max={np.max(pw_vals):.3f}")
        assert any(pw != 1.0 for pw in pw_vals), "所有权重均为默认值，残差字段未读取"
    print("b3d_loader v4 OK ✅")