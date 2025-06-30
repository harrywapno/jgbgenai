
import time
import subprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GPUMonitor")

while True:
    try:
        # Get GPU stats
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            stats = result.stdout.strip().split(', ')
            gpu_util = float(stats[0])
            memory_used = float(stats[1]) / 1024  # Convert to GB
            memory_total = float(stats[2]) / 1024
            temperature = float(stats[3])
            power = float(stats[4])
            
            logger.info(f"GPU Stats - Util: {gpu_util}%, Memory: {memory_used:.1f}/{memory_total:.1f}GB, Temp: {temperature}°C, Power: {power}W")
            
            # Alert if underutilized
            if gpu_util < 50:
                logger.warning("⚠ GPU utilization below 50% - consider increasing workload")
        
    except Exception as e:
        logger.error(f"Error monitoring GPU: {e}")
    
    time.sleep(60)  # Check every minute
