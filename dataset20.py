"""
b3d_to_chat_summary.py
把所有 .b3d 文件转换成可以直接粘贴到聊天框的紧凑摘要

已知条件（来自你的扫描结果）：
- num_dofs = 23
- 帧结构: processingPasses, markerObservations, rawForcePlateForces 等
- 无 EMG / IMU
- 帧数不固定（111 ~ 2000帧不等）

运行方式：
    python b3d_to_chat_summary.py

输出：
    chat_summary.json   （完整版，所有受试者）
    chat_summary.txt    （极简文本版，可直接粘贴）
"""

import os
import json
import numpy as np
import nimblephysics as nimble
from pathlib import Path
from typing import Optional

# ========== 配置 ==========
DATA_ROOT = "/mnt/d/Camargo2021_Formatted_No_Arm"
OUTPUT_JSON = "chat_summary.json"
OUTPUT_TXT  = "chat_summary.txt"
# 每个文件只深度解析前 N 帧（够用，省时间）
DEEP_FRAMES = 3
# ==========================


def arr(x):
    try:
        return np.array(x)
    except Exception:
        return None


def stat(x, label="") -> Optional[dict]:
    """对任意数组型数据求统计摘要"""
    a = arr(x)
    if a is None or a.size == 0:
        return None
    return {
        "shape": list(a.shape),
        "min":   round(float(np.min(a)), 6),
        "max":   round(float(np.max(a)), 6),
        "mean":  round(float(np.mean(a)), 6),
        "std":   round(float(np.std(a)), 6),
    }


def probe_frame(f) -> dict:
    """
    对单帧做全面探查：
    - processingPasses 里的每个属性
    - rawForcePlate 系列
    - markerObservations
    """
    info = {}

    # ---- rawForcePlate 系列 ----
    for attr in ["rawForcePlateForces", "rawForcePlateCenterOfPressures",
                 "rawForcePlateTorques"]:
        try:
            val = getattr(f, attr)
            s = stat(val)
            if s:
                info[attr] = s
        except Exception:
            pass

    # ---- missingGRFReason ----
    try:
        info["missingGRFReason"] = str(f.missingGRFReason)
    except Exception:
        pass

    # ---- t (时间戳) ----
    try:
        info["t"] = float(f.t)
    except Exception:
        pass

    # ---- markerObservations ----
    try:
        obs = f.markerObservations
        if obs and len(obs) > 0:
            coords = []
            names  = []
            for m in obs:
                # 尝试拿坐标
                a = arr(m)
                if a is not None:
                    coords.append(a.flatten().tolist())
                # 尝试拿名称
                for na in ["name", "markerName", "label"]:
                    try:
                        names.append(getattr(m, na))
                        break
                    except Exception:
                        pass
                else:
                    names.append(None)

            info["markerObservations"] = {
                "count": len(obs),
                "names_sample": names[:5],
                "coord_stat":   stat(coords) if coords else None,
                "coord_frame0": coords[0] if coords else None,  # 第一个marker的xyz
            }
    except Exception as e:
        info["markerObservations"] = {"error": str(e)}

    # ---- processingPasses ----
    try:
        passes = f.processingPasses
        if passes and len(passes) > 0:
            info["processingPasses"] = {}
            for pi, p in enumerate(passes):
                pass_info = {}
                p_attrs = [a for a in dir(p) if not a.startswith("_")]
                pass_info["attrs"] = p_attrs

                for attr in p_attrs:
                    try:
                        val = getattr(p, attr)
                        if callable(val):
                            continue
                        s = stat(val)
                        if s:
                            pass_info[attr] = s
                        elif isinstance(val, (int, float, str, bool)):
                            pass_info[attr] = val
                        elif isinstance(val, (list, tuple)) and len(val) < 50:
                            pass_info[attr] = list(val)
                    except Exception:
                        pass

                info["processingPasses"][f"pass_{pi}"] = pass_info
    except Exception as e:
        info["processingPasses"] = {"error": str(e)}

    return info


def summarize_b3d(file_path: str) -> dict:
    """对单个 .b3d 文件生成完整摘要"""
    subject = nimble.biomechanics.SubjectOnDisk(file_path)
    summary = {"file": os.path.basename(file_path)}

    # ---- Subject 级别 ----
    subj = {}
    subj["num_trials"] = subject.getNumTrials()
    subj["num_dofs"]   = subject.getNumDofs()

    # DOF 名称（最关键！）
    try:
        subj["dof_names"] = [subject.getDofName(i) for i in range(subject.getNumDofs())]
    except Exception as e:
        subj["dof_names_error"] = str(e)

    # 生物学信息
    for attr in ["getMass", "getHeightM", "getBiologicalSex",
                 "getAgeYears", "getSubjectName", "getNumProcessingPasses"]:
        try:
            subj[attr] = getattr(subject, attr)()
        except Exception:
            pass

    summary["subject"] = subj

    # ---- Trial 级别 ----
    trials = []
    for ti in range(subject.getNumTrials()):
        t_info = {"trial_id": ti}
        t_info["length"] = subject.getTrialLength(ti)

        try:
            t_info["timestep"] = subject.getTrialTimestep(ti)
            t_info["fps"]      = round(1.0 / t_info["timestep"]) if t_info["timestep"] > 0 else None
        except Exception:
            pass

        try:
            t_info["name"] = subject.getTrialName(ti)
        except Exception:
            pass

        try:
            t_info["tags"] = list(subject.getTrialTags(ti))
        except Exception:
            pass

        trials.append(t_info)

    summary["trials"] = trials

    # ---- Frame 级别深度探查（只用第一个trial的前 DEEP_FRAMES 帧）----
    summary["frame_probe"] = {}
    if subject.getNumTrials() > 0:
        trial_len = subject.getTrialLength(0)
        n = min(DEEP_FRAMES, trial_len)
        frames = subject.readFrames(0, 0, n)

        for fi, frame in enumerate(frames):
            summary["frame_probe"][f"frame_{fi}"] = probe_frame(frame)

    return summary


def find_b3d_files(root: str) -> list:
    """递归查找所有 .b3d 文件"""
    files = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname.endswith(".b3d"):
                files.append(os.path.join(dirpath, fname))
    return sorted(files)


def build_chat_txt(all_summaries: list) -> str:
    """
    生成极简文本版摘要（最适合粘贴到聊天框）
    只保留代码生成必须的信息
    """
    lines = ["=" * 60,
             "Camargo2021 数据集探查报告（供代码生成用）",
             "=" * 60, ""]

    for s in all_summaries:
        lines.append(f"📁 {s['file']}")
        subj = s.get("subject", {})
        lines.append(f"  DOFs: {subj.get('num_dofs')} | Trials: {subj.get('num_trials')}")

        dof_names = subj.get("dof_names")
        if dof_names:
            lines.append(f"  DOF 名称: {dof_names}")

        for attr in ["getMass", "getHeightM", "getBiologicalSex", "getAgeYears"]:
            if attr in subj:
                lines.append(f"  {attr}: {subj[attr]}")

        # 帧率（从第一个trial取）
        if s.get("trials"):
            t0 = s["trials"][0]
            lines.append(f"  Trial[0] 帧率: {t0.get('fps')} Hz | timestep: {t0.get('timestep')} s")
            lines.append(f"  Trial[0] 名称: {t0.get('name')} | tags: {t0.get('tags')}")

        # Frame 探查
        fp = s.get("frame_probe", {})
        if "frame_0" in fp:
            f0 = fp["frame_0"]
            lines.append(f"  --- Frame[0] 探查 ---")

            # t
            if "t" in f0:
                lines.append(f"    t (时间戳): {f0['t']}")

            # missingGRFReason
            if "missingGRFReason" in f0:
                lines.append(f"    missingGRFReason: {f0['missingGRFReason']}")

            # rawForcePlate
            for k in ["rawForcePlateForces", "rawForcePlateCenterOfPressures", "rawForcePlateTorques"]:
                if k in f0:
                    lines.append(f"    {k}: {f0[k]}")

            # markerObservations
            if "markerObservations" in f0:
                mo = f0["markerObservations"]
                lines.append(f"    markerObservations: count={mo.get('count')} "
                             f"names_sample={mo.get('names_sample')} "
                             f"coord_frame0={mo.get('coord_frame0')}")
                if mo.get("coord_stat"):
                    lines.append(f"    marker坐标统计: {mo['coord_stat']}")

            # processingPasses
            if "processingPasses" in f0 and isinstance(f0["processingPasses"], dict):
                pp = f0["processingPasses"]
                lines.append(f"    processingPasses 数量: {len(pp)}")
                for pname, pdata in pp.items():
                    if isinstance(pdata, dict):
                        lines.append(f"    {pname} 属性: {pdata.get('attrs', [])}")
                        for attr, val in pdata.items():
                            if attr == "attrs":
                                continue
                            lines.append(f"      .{attr}: {val}")

        lines.append("")

    return "\n".join(lines)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):  return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray):     return obj.tolist()
        return super().default(obj)


if __name__ == "__main__":
    print(f"🔍 扫描目录: {DATA_ROOT}")
    b3d_files = find_b3d_files(DATA_ROOT)
    print(f"   找到 {len(b3d_files)} 个 .b3d 文件\n")

    if not b3d_files:
        print("❌ 未找到任何 .b3d 文件，请检查 DATA_ROOT 路径")
        exit(1)

    all_summaries = []

    for i, fpath in enumerate(b3d_files):
        print(f"[{i+1}/{len(b3d_files)}] 处理: {os.path.basename(fpath)} ...")
        try:
            s = summarize_b3d(fpath)
            all_summaries.append(s)
            print(f"  ✅ 完成")
        except Exception as e:
            import traceback
            print(f"  ❌ 失败: {e}")
            traceback.print_exc()
            all_summaries.append({"file": os.path.basename(fpath), "error": str(e)})

    # 保存 JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False, cls=NpEncoder)
    print(f"\n✅ 完整 JSON 已保存: {OUTPUT_JSON}")

    # 保存纯文本
    txt = build_chat_txt(all_summaries)
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write(txt)
    print(f"✅ 纯文本摘要已保存: {OUTPUT_TXT}")

    # 打印文本摘要到终端
    print("\n" + "=" * 60)
    print("📋 以下内容可直接粘贴到聊天框：")
    print("=" * 60)
    print(txt)