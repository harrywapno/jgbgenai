import subprocess
import time
import sys

print("\n" + "="*80)
print("ULTIMATE GPU REMIX SYSTEM MONITOR")
print("="*80)

while True:
    try:
        # Get GPU stats
        result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            parts = result.stdout.strip().split(', ')
            gpu_util = float(parts[0])
            mem_used = float(parts[1]) / 1024
            mem_total = float(parts[2]) / 1024
            temp = float(parts[3])
            power = float(parts[4])
            
            # Clear line and print
            sys.stdout.write('\r')
            sys.stdout.write(
                f"GPU: {gpu_util:3.0f}% | "
                f"Memory: {mem_used:5.1f}/{mem_total:5.1f}GB | "
                f"Temp: {temp:2.0f}°C | "
                f"Power: {power:3.0f}W | "
                f"Target: 800W"
            )
            sys.stdout.flush()
            
            # Check processes
            proc_check = subprocess.run(['pgrep', '-f', 'ultimate_gpu_remix_system'], capture_output=True)
            if proc_check.returncode != 0:
                print("\n⚠️  Ultimate system not running! Restarting...")
                subprocess.run(['python3', 'launch_ultimate_system.sh'])
        
        time.sleep(2)
        
    except KeyboardInterrupt:
        print("\n\nMonitor stopped.")
        break
    except Exception as e:
        print(f"\nError: {e}")
        time.sleep(5)
