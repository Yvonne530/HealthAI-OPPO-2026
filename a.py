import h5py
import numpy as np

with h5py.File('/mnt/d/Camargo2021_Formatted_No_Arm/processed/dataset.h5', 'r') as f:
    labels = f['labels'][:]
    unique, counts = np.unique(labels, return_counts=True)
    
    print("=" * 50)
    print("重新打标后的风险分布")
    print("=" * 50)
    
    total = len(labels)
    for u, c in zip(unique, counts):
        risk_name = {0: "低风险", 1: "中风险", 2: "高风险"}.get(u, f"未知{u}")
        percentage = c / total * 100
        bar = "█" * int(percentage / 2)  # 简单进度条
        print(f"{risk_name} (标签 {u}): {c:>10,} 帧 ({percentage:>5.2f}%) {bar}")
    
    print(f"\n总帧数: {total:,}")
    print(f"类别数: {len(unique)}")
    print("=" * 50)
    
    # 合理性检查
    print("\n【合理性检查】")
    
    # 检查1：是否有3个类别
    if len(unique) == 3:
        print("✓ 通过: 有3个风险等级（低/中/高）")
    else:
        print(f"✗ 警告: 只有 {len(unique)} 个类别，应该是3个")
    
    # 检查2：中风险不为0
    if 1 in unique:
        mid_count = counts[list(unique).index(1)]
        if mid_count > 0:
            print(f"✓ 通过: 中风险有 {mid_count:,} 帧 ({mid_count/total*100:.2f}%)")
        else:
            print("✗ 警告: 中风险为0")
    else:
        print("✗ 警告: 没有中风险标签")
    
    # 检查3：分布是否合理（不应该极端不平衡）
    if len(unique) == 3:
        low_ratio = counts[list(unique).index(0)] / total if 0 in unique else 0
        mid_ratio = counts[list(unique).index(1)] / total if 1 in unique else 0
        high_ratio = counts[list(unique).index(2)] / total if 2 in unique else 0
        
        print(f"\n分布比例: 低={low_ratio:.1%}, 中={mid_ratio:.1%}, 高={high_ratio:.1%}")
        
        # 判断是否合理（通常不应该出现某个类别占比超过80%或低于5%）
        if low_ratio < 0.05:
            print("⚠️ 注意: 低风险占比过小 (<5%)")
        if mid_ratio < 0.05:
            print("⚠️ 注意: 中风险占比过小 (<5%)")
        if high_ratio < 0.05:
            print("⚠️ 注意: 高风险占比过小 (<5%)")
        
        if low_ratio > 0.8 or high_ratio > 0.8:
            print("⚠️ 注意: 某个类别占比过大 (>80%)，可能存在类别不平衡")
        else:
            print("✓ 分布相对平衡")
    
    # 检查4：与之前的对比
    print("\n【与之前对比】")
    print("之前: 低=827,320 (29.8%), 中=0 (0%), 高=1,948,759 (70.2%)")
    
    if 1 in unique and counts[list(unique).index(1)] > 0:
        print("✓ 改进: 中风险从 0 变为有数据")
    else:
        print("⚠️ 未改进: 中风险仍然为 0")