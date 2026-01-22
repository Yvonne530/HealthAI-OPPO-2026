"""
训练监控脚本
实时显示训练进度、Loss、显存占用等
"""
import json
import time
import os
from datetime import datetime
import subprocess

def get_gpu_memory():
    """获取 GPU 显存使用情况"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True
        )
        used, total = map(int, result.stdout.strip().split(','))
        return used, total
    except:
        return None, None

def parse_log_file(log_path):
    """解析训练日志文件"""
    if not os.path.exists(log_path):
        return []
    
    logs = []
    with open(log_path, 'r') as f:
        for line in f:
            try:
                log = json.loads(line.strip())
                logs.append(log)
            except:
                pass
    return logs

def print_training_status(logs):
    """打印训练状态"""
    if not logs:
        print("⏳ 等待训练开始...")
        return
    
    latest = logs[-1]
    
    print("\n" + "="*70)
    print(f"  训练监控 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # 基本信息
    if 'current_steps' in latest:
        print(f"\n当前步数: {latest['current_steps']}")
    
    if 'epoch' in latest:
        print(f"当前 Epoch: {latest['epoch']:.2f}")
    
    # Loss 信息
    if 'loss' in latest:
        print(f"训练 Loss: {latest['loss']:.4f}")
    
    if 'eval_loss' in latest:
        print(f"验证 Loss: {latest['eval_loss']:.4f}")
    
    # 学习率
    if 'learning_rate' in latest:
        print(f"学习率: {latest['learning_rate']:.2e}")
    
    # 显存信息
    used_mem, total_mem = get_gpu_memory()
    if used_mem and total_mem:
        usage_percent = (used_mem / total_mem) * 100
        print(f"\nGPU 显存: {used_mem} MB / {total_mem} MB ({usage_percent:.1f}%)")
        
        # 显存预警
        if usage_percent > 95:
            print("⚠️  警告: 显存使用率超过 95%，可能发生 OOM")
    
    # 训练速度
    if 'train_samples_per_second' in latest:
        print(f"训练速度: {latest['train_samples_per_second']:.2f} samples/s")
    
    # 最近 10 步的 Loss 趋势
    if len(logs) >= 10:
        recent_losses = [log.get('loss', 0) for log in logs[-10:] if 'loss' in log]
        if recent_losses:
            avg_loss = sum(recent_losses) / len(recent_losses)
            print(f"\n最近 10 步平均 Loss: {avg_loss:.4f}")
            
            # 判断是否收敛
            if len(recent_losses) >= 2:
                trend = recent_losses[-1] - recent_losses[0]
                if trend < 0:
                    print("📉 Loss 趋势: 下降（良好）")
                elif trend > 0.1:
                    print("📈 Loss 趋势: 上升（需关注）")
                else:
                    print("➡️  Loss 趋势: 稳定")
    
    print("="*70)

def monitor_training(log_path="/kaggle/working/rehab_lora/trainer_log.jsonl", interval=30):
    """
    持续监控训练过程
    
    Args:
        log_path: 训练日志文件路径
        interval: 刷新间隔（秒）
    """
    print("\n开始监控训练...")
    print(f"日志文件: {log_path}")
    print(f"刷新间隔: {interval} 秒")
    print("按 Ctrl+C 停止监控\n")
    
    try:
        while True:
            # 清屏（在 Jupyter 中可能不生效）
            os.system('clear' if os.name != 'nt' else 'cls')
            
            # 解析日志
            logs = parse_log_file(log_path)
            
            # 显示状态
            print_training_status(logs)
            
            # 等待
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\n监控已停止")

def show_training_summary(log_path="/kaggle/working/rehab_lora/trainer_log.jsonl"):
    """显示训练总结"""
    logs = parse_log_file(log_path)
    
    if not logs:
        print("没有找到训练日志")
        return
    
    print("\n" + "="*70)
    print("  训练总结")
    print("="*70)
    
    # 收集所有 loss 值
    train_losses = [log.get('loss') for log in logs if 'loss' in log]
    eval_losses = [log.get('eval_loss') for log in logs if 'eval_loss' in log]
    
    if train_losses:
        print(f"\n训练 Loss:")
        print(f"  初始: {train_losses[0]:.4f}")
        print(f"  最终: {train_losses[-1]:.4f}")
        print(f"  最小: {min(train_losses):.4f}")
        print(f"  下降: {train_losses[0] - train_losses[-1]:.4f}")
    
    if eval_losses:
        print(f"\n验证 Loss:")
        print(f"  初始: {eval_losses[0]:.4f}")
        print(f"  最终: {eval_losses[-1]:.4f}")
        print(f"  最小: {min(eval_losses):.4f}")
    
    # 训练时长
    if len(logs) >= 2:
        first_log = logs[0]
        last_log = logs[-1]
        
        if 'current_steps' in first_log and 'current_steps' in last_log:
            total_steps = last_log['current_steps'] - first_log['current_steps']
            print(f"\n总训练步数: {total_steps}")
    
    print("="*70 + "\n")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--summary":
            # 显示训练总结
            show_training_summary()
        elif sys.argv[1] == "--once":
            # 显示一次状态
            logs = parse_log_file("/kaggle/working/rehab_lora/trainer_log.jsonl")
            print_training_status(logs)
        else:
            print("用法:")
            print("  python monitor_training.py           # 持续监控")
            print("  python monitor_training.py --once    # 显示一次")
            print("  python monitor_training.py --summary # 显示总结")
    else:
        # 默认: 持续监控
        monitor_training()