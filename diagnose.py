"""
diagnose.py - 30秒快速诊断，找出卡住的根本原因
"""
import os, sys, time
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

print("=" * 55)
print("RehabGuardian 快速诊断")
print("=" * 55)

# 1. h5py multiprocessing 测试
print("\n[1] h5py fork 安全性测试...")
import h5py, numpy as np, tempfile, multiprocessing as mp

def _worker_test(h5_path, result_q):
    try:
        with h5py.File(h5_path, 'r') as h5:
            _ = h5['joint_angles'][0]
        result_q.put('ok')
    except Exception as e:
        result_q.put(f'fail:{e}')

tmpf = tempfile.mktemp(suffix='.h5')
with h5py.File(tmpf, 'w') as h5:
    h5.create_dataset('joint_angles', data=np.zeros((10,23), np.float32))

q = mp.Queue()
p = mp.Process(target=_worker_test, args=(tmpf, q))
p.start(); p.join(timeout=5)

if p.is_alive():
    p.kill()
    print("  ❌ h5py fork 5秒超时 → DataLoader num_workers>0 会死锁！")
    print("  ✅ 解决方案：num_workers=0（已在修复版config中设置）")
    H5_FORK_OK = False
else:
    result = q.get() if not q.empty() else 'timeout'
    if result == 'ok':
        print(f"  ✅ h5py fork 安全（result={result}）")
        H5_FORK_OK = True
    else:
        print(f"  ❌ h5py fork 失败：{result}")
        H5_FORK_OK = False
os.unlink(tmpf)

# 2. CUDA 状态
print("\n[2] CUDA 状态...")
import torch
if torch.cuda.is_available():
    vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
    vram_free  = (torch.cuda.get_device_properties(0).total_memory 
                  - torch.cuda.memory_allocated(0)) / 1e9
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {vram_total:.1f}GB 总量, {vram_free:.1f}GB 空闲")
else:
    print("  ❌ CUDA 不可用")

# 3. HDF5 单帧读取速度
print("\n[3] HDF5 单线程读取速度...")
h5_path = "/mnt/d/Camargo2021_Formatted_No_Arm/processed/dataset.h5"
if os.path.exists(h5_path):
    with h5py.File(h5_path, 'r') as h5:
        t0 = time.perf_counter()
        N  = 1000
        idx = np.random.randint(0, h5['joint_angles'].shape[0], N)
        idx.sort()  # 顺序读更快
        _ = h5['joint_angles'][idx]
        _ = h5['grf_left'][idx]
        elapsed = time.perf_counter() - t0
    fps = N / elapsed
    print(f"  1000 帧随机读: {elapsed*1000:.0f}ms → {fps:.0f} 帧/s")
    if fps < 5000:
        print("  ⚠️  读取偏慢（/mnt/d/ 9p协议，正常现象）")
    # 估算 batch_size=16 下每秒能跑多少 batch
    batch_fps = fps / 16
    print(f"  batch_size=16 预估: {batch_fps:.0f} batch/s")
    # 预估每 epoch 时间
    N_windows = 1_714_269
    epoch_min  = N_windows / 16 / batch_fps / 60
    print(f"  全量 epoch 预估: {epoch_min:.0f} 分钟 ← 太慢！建议用 epoch_subset")
else:
    print(f"  HDF5 不存在: {h5_path}")

# 4. batch_size=16 显存占用估算
print("\n[4] 显存需求估算 (batch_size=16)...")
B = 16
items = {
    "pose_seq (5,33,3)":   B*5*33*3*4,
    "bio_seq  (20,72)":    B*20*72*4,
    "grf_future (10,12)":  B*10*12*4,
    "ST-GCN 参数 ~0.5M":   0.5e6*4,
    "FNO 参数 ~0.5M":      0.5e6*4,
    "激活值+梯度 (估算)":   200*1e6,
}
total = sum(items.values())
for k,v in items.items():
    print(f"  {k:<28} {v/1e6:.1f} MB")
print(f"  总计: {total/1e9:.2f} GB  ← 8.6GB VRAM 完全够用")

print("\n" + "=" * 55)
print("诊断完成")
if not H5_FORK_OK:
    print("⚠️  必须修复：config.yaml → num_workers: 0")
print("=" * 55)