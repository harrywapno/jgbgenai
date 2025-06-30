#!/bin/bash

echo "Launching Ultimate GPU-Accelerated JGB Remix System"
echo "=================================================="

# Kill existing GPU processes (except scraper)
echo "Stopping old processes..."
pkill -f "stable_max_gpu.py"
pkill -f "push_to_800w.py"
pkill -f "max_gpu_ai_remix.py"
pkill -f "ai_remix_runner.py"
sleep 2

# Check current GPU status
echo -e "\nCurrent GPU Status:"
nvidia-smi --query-gpu=memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader

# Set environment variables for maximum performance
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,expandable_segments:True"
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDA_ARCH_LIST="9.0"

# Enable maximum GPU performance
echo -e "\nSetting GPU to maximum performance mode..."
sudo nvidia-smi -pm 1 2>/dev/null || echo "Persistence mode already set"
sudo nvidia-smi -ac 2619,1980 2>/dev/null || echo "Clocks already set"
sudo nvidia-smi -pl 1000 2>/dev/null || echo "Power limit already set"

# Launch the ultimate system
echo -e "\nStarting Ultimate GPU Remix System..."
nohup python3 -u ultimate_gpu_remix_system.py > ultimate_system.log 2>&1 &
ULTIMATE_PID=$!

echo "✓ Ultimate system started (PID: $ULTIMATE_PID)"

# Monitor script
cat > monitor_ultimate.py << 'EOF'
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
EOF

# Start monitor
echo -e "\nStarting monitor..."
python3 monitor_ultimate.py