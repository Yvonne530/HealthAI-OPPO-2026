#!/usr/bin/env python3
"""
fix_markers_stats.py
修复 norm_stats 中 markers_flat 通道的 NaN 值。

策略：
  - 有真实数据的 marker 通道：保留 mean/std
  - 全 NaN 的通道（数据缺失）：mean=0, std=1
    （归一化后 value = (0 - 0) / 1 = 0，与 Android 端缺失 marker 填充 0 一致）

输出：
  - 覆盖写入 assets/norm_stats.json（Android App）
  - 同时生成 assets/norm_stats.npz（训练侧复用）
"""
import json
import numpy as np
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

def fix_markers_stats(json_path: str) -> str:
    with open(json_path, "r") as f:
        stats = json.load(f)

    mean_arr = np.array(stats["markers_flat_mean"], dtype=np.float64)
    std_arr  = np.array(stats["markers_flat_std"], dtype=np.float64)

    # Identify all-NaN channels
    bad = ~np.isfinite(mean_arr)
    n_bad = bad.sum()
    n_good = (~bad).sum()

    print(f"markers_flat: {n_good} good channels, {n_bad} all-NaN channels (total {len(mean_arr)})")

    # Fix: mean=0, std=1 for all-NaN channels
    mean_arr[bad] = 0.0
    std_arr[bad]  = 1.0

    # Ensure no remaining NaN/Inf
    assert np.all(np.isfinite(mean_arr)), "Still have NaN in mean after fix!"
    assert np.all(np.isfinite(std_arr)),  "Still have NaN in std after fix!"

    stats["markers_flat_mean"] = mean_arr.tolist()
    stats["markers_flat_std"]  = std_arr.tolist()

    # Write JSON
    with open(json_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"✅ Written: {json_path}")

    # Also save as npz for training side
    npz_path = json_path.replace(".json", ".npz")
    np.savez(npz_path, **{k: np.array(v, dtype=np.float32) for k, v in stats.items()})
    print(f"✅ Written: {npz_path}")

    return json_path


if __name__ == "__main__":
    # Try v1 source first, then current app assets
    v1_path = os.path.join(ROOT, "projects", "RehabGuardian_v1", "app", "src", "main", "assets", "norm_stats_PhaseA.json")
    app_path = os.path.join(ROOT, "projects", "RehabGuardian", "app", "src", "main", "assets", "norm_stats.json")

    if os.path.exists(v1_path):
        src = v1_path
        print(f"Using v1 source: {src}")
        # Generate fixed version into app assets
        fix_markers_stats(src)
        # Copy fixed JSON to app assets
        import shutil
        dst = app_path
        fix_markers_stats(src)
        # Read fixed content and write to app path
        with open(src, "r") as f:
            content = json.load(f)
        with open(dst, "w") as f:
            json.dump(content, f, indent=2)
        # Also save npz alongside app assets
        npz_dst = os.path.join(os.path.dirname(dst), "norm_stats.npz")
        np.savez(npz_dst, **{k: np.array(v, dtype=np.float32) for k, v in content.items()})
        print(f"✅ Copied fixed JSON to: {dst}")
        print(f"✅ Copied fixed npz to:  {npz_dst}")
    elif os.path.exists(app_path):
        fix_markers_stats(app_path)
    else:
        print("ERROR: No norm_stats JSON found")
        exit(1)

    # Validate
    with open(dst if os.path.exists(dst) else app_path, "r") as f:
        final = json.load(f)
    assert not any(np.any(np.isnan(np.array(final[k]))) for k in ["markers_flat_mean", "markers_flat_std"]), "Validation failed!"
    print("✅ Validation passed: no NaN/Inf in markers_flat")
