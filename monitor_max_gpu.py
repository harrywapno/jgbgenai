#!/usr/bin/env python3
"""
Monitor B200 GPU Maximum Utilization
"""

import subprocess
import time
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_gpu_stats():
    """Get comprehensive GPU statistics"""
    try:
        # Query all GPU metrics
        result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,clocks.gr,clocks.mem',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            parts = result.stdout.strip().split(', ')
            return {
                'name': parts[0],
                'gpu_util': float(parts[1]),
                'memory_used_mb': float(parts[2]),
                'memory_total_mb': float(parts[3]),
                'temperature': float(parts[4]),
                'power': float(parts[5]),
                'graphics_clock': float(parts[6]),
                'memory_clock': float(parts[7])
            }
    except Exception as e:
        logger.error(f"Error getting GPU stats: {e}")
    return None

def monitor_loop():
    """Main monitoring loop"""
    target_memory_gb = 150
    target_power_w = 800
    target_temp_c = 50
    
    print("\033[2J\033[H")  # Clear screen
    
    while True:
        stats = get_gpu_stats()
        
        if stats:
            # Clear and display
            print("\033[2J\033[H")
            print("=" * 80)
            print("B200 GPU MAXIMUM UTILIZATION MONITOR".center(80))
            print("=" * 80)
            print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print()
            
            # GPU info
            print(f"GPU: {stats['name']}")
            print()
            
            # Memory status
            memory_gb = stats['memory_used_mb'] / 1024
            memory_total_gb = stats['memory_total_mb'] / 1024
            memory_pct = (memory_gb / memory_total_gb) * 100
            memory_target_pct = (memory_gb / target_memory_gb) * 100
            
            print("MEMORY STATUS")
            print("-" * 40)
            print(f"Used:     {memory_gb:.1f}GB / {memory_total_gb:.1f}GB ({memory_pct:.1f}%)")
            print(f"Target:   {target_memory_gb}GB ({memory_target_pct:.1f}% achieved)")
            
            # Visual bar
            bar_length = 50
            filled = int(memory_pct * bar_length / 100)
            bar = "█" * filled + "░" * (bar_length - filled)
            print(f"Progress: [{bar}]")
            
            if memory_gb < target_memory_gb:
                print(f"Status:   ⚠️  Need {target_memory_gb - memory_gb:.1f}GB more")
            else:
                print(f"Status:   ✅ Target achieved! ({memory_gb:.1f}GB)")
            print()
            
            # Power status
            print("POWER STATUS")
            print("-" * 40)
            print(f"Current:  {stats['power']:.0f}W / 1000W")
            print(f"Target:   {target_power_w}W ({stats['power']/target_power_w*100:.1f}% achieved)")
            
            # Visual bar
            power_pct = stats['power'] / 10  # Out of 1000W
            filled = int(power_pct * bar_length / 100)
            bar = "█" * filled + "░" * (bar_length - filled)
            print(f"Progress: [{bar}]")
            
            if stats['power'] < target_power_w:
                print(f"Status:   ⚠️  Need {target_power_w - stats['power']:.0f}W more")
            else:
                print(f"Status:   ✅ Target achieved! ({stats['power']:.0f}W)")
            print()
            
            # Temperature status
            print("TEMPERATURE STATUS")
            print("-" * 40)
            print(f"Current:  {stats['temperature']:.0f}°C")
            print(f"Target:   {target_temp_c}°C")
            
            # Visual thermometer
            temp_pct = stats['temperature'] / 100
            filled = int(temp_pct * bar_length)
            bar = "█" * filled + "░" * (bar_length - filled)
            print(f"Progress: [{bar}]")
            
            if stats['temperature'] < target_temp_c - 5:
                print(f"Status:   ⚠️  Too cool, increase workload")
            elif stats['temperature'] > target_temp_c + 5:
                print(f"Status:   ⚠️  Too hot, reduce workload")
            else:
                print(f"Status:   ✅ Optimal temperature range")
            print()
            
            # Performance metrics
            print("PERFORMANCE METRICS")
            print("-" * 40)
            print(f"GPU Utilization:  {stats['gpu_util']:.1f}%")
            print(f"Graphics Clock:   {stats['graphics_clock']:.0f} MHz")
            print(f"Memory Clock:     {stats['memory_clock']:.0f} MHz")
            print()
            
            # Recommendations
            print("OPTIMIZATION STATUS")
            print("-" * 40)
            
            if memory_gb >= target_memory_gb and stats['power'] >= target_power_w:
                print("✅ MAXIMUM PERFORMANCE ACHIEVED!")
                print("   - Memory target exceeded")
                print("   - Power target exceeded")
                print("   - B200 running at full capacity")
            else:
                print("🚀 OPTIMIZING...")
                if memory_gb < target_memory_gb:
                    print(f"   - Loading more data to GPU ({target_memory_gb - memory_gb:.1f}GB needed)")
                if stats['power'] < target_power_w:
                    print(f"   - Increasing computational load ({target_power_w - stats['power']:.0f}W needed)")
                if stats['gpu_util'] < 90:
                    print(f"   - GPU utilization low, adding more parallel operations")
            
            print()
            print("=" * 80)
            print("Press Ctrl+C to exit monitor")
        
        time.sleep(2)

if __name__ == "__main__":
    try:
        monitor_loop()
    except KeyboardInterrupt:
        print("\n\nMonitor closed. GPU processes continue running.")