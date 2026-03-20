import os
import json
import numpy as np
import nimblephysics as nimble

# ========== 配置路径 ==========
FILE_PATH = "/mnt/d/Camargo2021_Formatted_No_Arm/AB06_split0/AB06_split0.b3d"
# =================================

def inspect_b3d_simple(file_path):
    """简化版B3D检查器 - 只提取可靠数据"""
    
    print(f"\n{'='*60}")
    print(f"📁 分析文件: {os.path.basename(file_path)}")
    print(f"{'='*60}\n")
    
    # 加载数据
    subject = nimble.biomechanics.SubjectOnDisk(file_path)
    
    report = {
        "file": file_path,
        "基本信息": {},
        "运动学数据": {},
        "动力学数据": {},
        "标记点数据": {},
        "问题诊断": {}
    }
    
    # ========== 1. 基本信息 ==========
    print("📊 读取基本信息...")
    trial_idx = 0
    trial_length = subject.getTrialLength(trial_idx)
    timestep = subject.getTrialTimestep(trial_idx)
    
    # 尝试获取质量（如果失败就用默认值）
    try:
        mass = subject.getMass()
    except:
        mass = 75.0  # 默认75kg
    
    report["基本信息"] = {
        "试验名称": subject.getTrialName(trial_idx),
        "总帧数": trial_length,
        "采样间隔(秒)": timestep,
        "帧率(Hz)": round(1.0 / timestep),
        "持续时间(秒)": round(trial_length * timestep, 3),
        "受试者身高(米)": subject.getHeightM(),
        "性别": subject.getBiologicalSex(),
        "年龄": subject.getAgeYears(),
        "估算体重(kg)": mass,
        "处理通道数": subject.getNumProcessingPasses()
    }
    
    # ========== 2. 读取所有帧 ==========
    print(f"📖 读取全部 {trial_length} 帧数据...")
    all_frames = subject.readFrames(trial_idx, 0, trial_length)
    print(f"   ✅ 成功读取 {len(all_frames)} 帧")
    
    # ========== 3. 提取运动学数据（位置） ==========
    print("\n🏃 提取运动学数据...")
    
    positions = []  # 关节位置
    com_positions = []  # 质心位置
    
    for frame in all_frames:
        if hasattr(frame, 'processingPasses') and frame.processingPasses:
            # 使用最后一个处理通道
            last_pass = frame.processingPasses[-1]
            
            # 关节位置
            if hasattr(last_pass, 'pos'):
                pos_val = last_pass.pos
                if pos_val is not None:
                    positions.append(np.array(pos_val))
            
            # 质心位置
            if hasattr(last_pass, 'comPos'):
                com_val = last_pass.comPos
                if com_val is not None:
                    com_positions.append(np.array(com_val))
    
    if positions:
        positions = np.array(positions)
        report["运动学数据"]["关节位置"] = {
            "形状": list(positions.shape),
            "范围": [float(np.min(positions)), float(np.max(positions))],
            "均值": float(np.mean(positions)),
            "标准差": float(np.std(positions))
        }
        
        # 计算速度（通过差分）
        if len(positions) > 1:
            velocities = np.diff(positions, axis=0) / timestep
            report["运动学数据"]["关节速度"] = {
                "形状": list(velocities.shape),
                "范围": [float(np.min(velocities)), float(np.max(velocities))],
                "均值": float(np.mean(velocities)),
                "标准差": float(np.std(velocities))
            }
    
    if com_positions:
        com_positions = np.array(com_positions)
        report["运动学数据"]["质心位置"] = {
            "形状": list(com_positions.shape),
            "范围": [float(np.min(com_positions)), float(np.max(com_positions))],
            "均值": float(np.mean(com_positions)),
            "标准差": float(np.std(com_positions))
        }
        
        # 计算质心速度和加速度
        if len(com_positions) > 1:
            com_vel = np.diff(com_positions, axis=0) / timestep
            report["运动学数据"]["质心速度"] = {
                "形状": list(com_vel.shape),
                "范围": [float(np.min(com_vel)), float(np.max(com_vel))],
                "均值": float(np.mean(com_vel)),
                "标准差": float(np.std(com_vel)),
                "合理性": bool(np.max(np.abs(com_vel)) < 10)  # 人体速度<10m/s
            }
            
            if len(com_vel) > 1:
                com_acc = np.diff(com_vel, axis=0) / timestep
                report["运动学数据"]["质心加速度"] = {
                    "形状": list(com_acc.shape),
                    "范围": [float(np.min(com_acc)), float(np.max(com_acc))],
                    "均值": float(np.mean(com_acc)),
                    "标准差": float(np.std(com_acc)),
                    "合理性": bool(np.max(np.abs(com_acc)) < 50)  # 人体加速度<50m/s²
                }
    
    # ========== 4. 提取动力学数据（原始力台数据） ==========
    print("\n⚖️  提取力台数据...")
    
    raw_forces = []  # 原始力台力数据
    raw_cop = []     # 原始压力中心
    processed_grf = []  # 处理后的GRF
    
    for frame_idx, frame in enumerate(all_frames):
        # 原始力台数据（最可靠）
        if hasattr(frame, 'rawForcePlateForces'):
            forces = frame.rawForcePlateForces
            if forces is not None and len(forces) > 0:
                try:
                    # 转换为numpy数组以便索引
                    forces_np = np.array(forces)
                    # 计算总垂直力（假设Z轴是第3列）
                    if len(forces_np.shape) == 2 and forces_np.shape[1] >= 3:
                        total_vertical = np.sum(forces_np[:, 2])
                        raw_forces.append(total_vertical)
                    elif len(forces_np.shape) == 1 and len(forces_np) >= 3:
                        total_vertical = forces_np[2]
                        raw_forces.append(total_vertical)
                    else:
                        # 其他情况，直接求和
                        total_vertical = np.sum(forces_np)
                        raw_forces.append(total_vertical)
                except Exception as e:
                    print(f"   警告: 处理力台数据时出错 {e}")
                    raw_forces.append(0)
        
        # 压力中心
        if hasattr(frame, 'rawForcePlateCenterOfPressures'):
            cop = frame.rawForcePlateCenterOfPressures
            if cop is not None and len(cop) > 0:
                try:
                    raw_cop.append(np.array(cop))
                except:
                    pass
        
        # 处理后的GRF（可能有问题）
        if hasattr(frame, 'processingPasses') and frame.processingPasses:
            last_pass = frame.processingPasses[-1]
            if hasattr(last_pass, 'groundContactWrenches'):
                grf = last_pass.groundContactWrenches
                if grf is not None and len(grf) >= 3:
                    try:
                        grf_np = np.array(grf)
                        # 提取左右脚的垂直力（假设顺序是左脚力、左脚力矩、右脚力、右脚力矩）
                        if len(grf_np) >= 9:
                            left_vertical = grf_np[2]  # 左脚垂直力
                            right_vertical = grf_np[8]  # 右脚垂直力
                            processed_grf.append(left_vertical + right_vertical)
                        elif len(grf_np) >= 3:
                            processed_grf.append(grf_np[2])  # 只有一只脚？
                    except Exception as e:
                        pass
    
    # 转换为numpy数组
    raw_forces = np.array(raw_forces) if raw_forces else None
    processed_grf = np.array(processed_grf) if processed_grf else None
    
    # 分析原始力台数据
    if raw_forces is not None and len(raw_forces) > 0:
        report["动力学数据"]["原始力台_垂直力"] = {
            "形状": list(raw_forces.shape),
            "范围": [float(np.min(raw_forces)), float(np.max(raw_forces))],
            "均值": float(np.mean(raw_forces)),
            "标准差": float(np.std(raw_forces)),
            "合理性": bool(np.max(raw_forces) < 2000)  # 正常<2000N
        }
        
        # 检查是否合理（假设体重70kg，站立应~686N）
        if len(raw_forces) > 10 and np.mean(raw_forces[:10]) > 0:
            estimated_weight = np.mean(raw_forces[:10]) / 9.8
            report["动力学数据"]["原始力台_估算体重"] = {
                "值_kg": round(estimated_weight, 1),
                "合理性": bool(40 < estimated_weight < 120)
            }
    
    # 分析处理后的GRF（对比用）
    if processed_grf is not None and len(processed_grf) > 0:
        report["动力学数据"]["处理后GRF_垂直力"] = {
            "形状": list(processed_grf.shape),
            "范围": [float(np.min(processed_grf)), float(np.max(processed_grf))],
            "均值": float(np.mean(processed_grf)),
            "标准差": float(np.std(processed_grf))
        }
        
        # 对比原始和处理后的数据
        if raw_forces is not None and len(raw_forces) == len(processed_grf):
            # 避免除零
            valid_idx = raw_forces > 1
            if np.any(valid_idx):
                ratio = processed_grf[valid_idx] / raw_forces[valid_idx]
                report["动力学数据"]["处理前后比值"] = {
                    "均值": float(np.mean(ratio)),
                    "标准差": float(np.std(ratio)),
                    "中位数": float(np.median(ratio)),
                    "说明": "如果比值>2，说明处理后数据被放大了"
                }
    
    # ========== 5. 标记点数据 ==========
    print("\n📍 提取标记点数据...")
    
    if all_frames and hasattr(all_frames[0], 'markerObservations'):
        markers = all_frames[0].markerObservations
        report["标记点数据"]["标记点数量"] = len(markers)
        
        # 获取标记点名称
        marker_names = []
        marker_coords = []
        
        for m in markers:
            # 获取名称
            if hasattr(m, 'name'):
                marker_names.append(m.name)
            elif isinstance(m, tuple) and len(m) > 0:
                marker_names.append(str(m[0]))
            else:
                marker_names.append(f"marker_{len(marker_names)}")
            
            # 获取坐标
            coord = None
            if hasattr(m, 'position'):
                coord = m.position
            elif isinstance(m, tuple) and len(m) >= 4:
                coord = m[1:4]
            elif isinstance(m, (list, tuple)) and len(m) >= 3:
                coord = m[:3]
            
            if coord is not None:
                try:
                    marker_coords.append(np.array(coord))
                except:
                    pass
        
        report["标记点数据"]["标记点名称_前10个"] = marker_names[:10]
        
        if marker_coords:
            marker_coords = np.array(marker_coords)
            report["标记点数据"]["坐标范围_第一帧"] = {
                "min": float(np.min(marker_coords)),
                "max": float(np.max(marker_coords)),
                "单位": "米" if np.max(np.abs(marker_coords)) < 5 else "毫米"
            }
    
    # ========== 6. 问题诊断 ==========
    print("\n🔍 进行问题诊断...")
    
    diagnosis = []
    suggestions = []
    
    # 检查运动学
    if com_positions is not None and len(com_positions) > 1:
        com_vel_max = np.max(np.abs(np.diff(com_positions, axis=0) / timestep))
        if com_vel_max < 10:
            diagnosis.append(f"✅ 运动学数据合理（最大速度{com_vel_max:.2f}m/s）")
        else:
            diagnosis.append(f"⚠️ 运动学速度异常：{com_vel_max:.1f}m/s（正常<10m/s）")
            suggestions.append("检查运动学数据是否被正确提取")
    
    # 检查GRF
    if raw_forces is not None and len(raw_forces) > 0:
        grf_max = np.max(raw_forces)
        grf_min = np.min(raw_forces)
        
        if grf_max < 2000 and grf_min >= 0:
            diagnosis.append(f"✅ 原始力台数据合理（峰值{grf_max:.0f}N，最小值{grf_min:.0f}N）")
        elif grf_max < 2000 and grf_min < 0:
            diagnosis.append(f"⚠️ 原始力台有负值（{grf_min:.0f}N），可能有拉力")
            suggestions.append("检查力台是否标定正确")
        elif grf_max > 2000:
            diagnosis.append(f"⚠️ 原始力台数据偏大（峰值{grf_max:.0f}N）")
            suggestions.append("可能需要除以标定系数")
    else:
        diagnosis.append("⚠️ 未找到原始力台数据")
    
    # 检查处理后GRF是否被放大
    if processed_grf is not None and len(processed_grf) > 0:
        if raw_forces is not None and len(raw_forces) > 0:
            if np.mean(processed_grf) > 2 * np.mean(raw_forces):
                ratio = np.mean(processed_grf) / np.mean(raw_forces)
                diagnosis.append(f"⚠️ 处理后GRF是原始数据的{ratio:.1f}倍，可能被错误累加")
                suggestions.append("建议使用原始力台数据 rawForcePlateForces 替代处理后GRF")
    
    # 检查接触状态
    contact_frames = 0
    total_frames = 0
    for frame in all_frames[:50]:
        if hasattr(frame, 'processingPasses') and frame.processingPasses:
            last_pass = frame.processingPasses[-1]
            if hasattr(last_pass, 'contact'):
                contact = last_pass.contact
                if contact is not None:
                    total_frames += 1
                    try:
                        contact_np = np.array(contact)
                        if np.sum(contact_np) > 0:
                            contact_frames += 1
                    except:
                        pass
    
    if total_frames > 0:
        contact_ratio = contact_frames / total_frames
        diagnosis.append(f"接触状态：{contact_ratio:.0%}的帧有接触")
        if contact_ratio < 0.1:
            diagnosis.append("   ⚠️ 接触比例过低，可能定义有问题（0可能表示接触）")
            suggestions.append("尝试反转contact的含义（如果当前是0=接触，1=无接触）")
        elif contact_ratio > 0.9:
            diagnosis.append("   ⚠️ 接触比例过高，可能一直检测到接触")
    
    report["问题诊断"] = {
        "诊断结果": diagnosis,
        "建议": suggestions
    }
    
    # ========== 7. 总结 ==========
    print("\n" + "="*60)
    print("📋 总结报告")
    print("="*60)
    
    for diag in diagnosis:
        print(f"  {diag}")
    
    if suggestions:
        print("\n💡 建议：")
        for sugg in suggestions:
            print(f"  • {sugg}")
    
    # ========== 8. 保存结果 ==========
    output_file = "b3d_analysis_report.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, cls=NpEncoder)
    
    print(f"\n✅ 完整报告已保存到: {output_file}")
    
    return report

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, (list, tuple)):
            return [self.default(x) for x in obj]
        return str(obj)

if __name__ == "__main__":
    if not os.path.exists(FILE_PATH):
        print(f"❌ 文件不存在: {FILE_PATH}")
        exit(1)
    
    try:
        report = inspect_b3d_simple(FILE_PATH)
        
        # 打印关键结论
        print("\n" + "="*60)
        print("🎯 关键结论")
        print("="*60)
        
        if "动力学数据" in report:
            if "原始力台_垂直力" in report["动力学数据"]:
                grf = report["动力学数据"]["原始力台_垂直力"]
                print(f"\n📊 原始力台垂直力:")
                print(f"   范围: {grf['范围'][0]:.0f} ~ {grf['范围'][1]:.0f} N")
                print(f"   均值: {grf['均值']:.0f} N")
                print(f"   合理性: {'✅ 合理' if grf['合理性'] else '⚠️ 需要检查'}")
            
            if "处理后GRF_垂直力" in report["动力学数据"]:
                grf_proc = report["动力学数据"]["处理后GRF_垂直力"]
                print(f"\n📊 处理后GRF垂直力:")
                print(f"   范围: {grf_proc['范围'][0]:.0f} ~ {grf_proc['范围'][1]:.0f} N")
                print(f"   均值: {grf_proc['均值']:.0f} N")
                
                if "处理前后比值" in report["动力学数据"]:
                    ratio = report["动力学数据"]["处理前后比值"]["中位数"]
                    print(f"\n📈 处理后/原始比值: {ratio:.1f}x")
                    if ratio > 2:
                        print("\n⚠️ 重要发现：处理后GRF被放大了！")
                        print("   → 建议直接使用原始力台数据 rawForcePlateForces")
        
        if "运动学数据" in report:
            if "质心速度" in report["运动学数据"]:
                vel = report["运动学数据"]["质心速度"]
                print(f"\n🏃 质心运动:")
                print(f"   速度范围: {vel['范围'][0]:.2f} ~ {vel['范围'][1]:.2f} m/s")
                print(f"   合理性: {'✅ 合理' if vel['合理性'] else '⚠️ 异常'}")
            
            if "质心加速度" in report["运动学数据"]:
                acc = report["运动学数据"]["质心加速度"]
                print(f"   加速度范围: {acc['范围'][0]:.1f} ~ {acc['范围'][1]:.1f} m/s²")
                print(f"   合理性: {'✅ 合理' if acc['合理性'] else '⚠️ 异常'}")
        
        print("\n" + "="*60)
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()