#!/usr/bin/env python3
"""
Push B200 to 800W Power Consumption
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import torch.nn as nn
import numpy as np
import threading
import time
import logging

torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def intensive_matmul_loop():
    """Continuous large matrix multiplications"""
    device = torch.device('cuda')
    
    while True:
        try:
            # Use float16 for more operations per second
            size = 16384  # Very large matrices
            A = torch.randn(size, size, device=device, dtype=torch.float16)
            B = torch.randn(size, size, device=device, dtype=torch.float16)
            
            # Chain multiplications
            C = torch.matmul(A, B)
            D = torch.matmul(C, A)
            E = torch.matmul(D, B)
            F = torch.matmul(E, C)
            
            # Force sync
            torch.cuda.synchronize()
            
        except Exception as e:
            # If OOM, use smaller size
            size = 8192
            continue


def intensive_conv_loop():
    """Continuous convolution operations"""
    device = torch.device('cuda')
    
    # Large CNN
    model = nn.Sequential(
        nn.Conv2d(3, 256, 7),
        nn.ReLU(),
        nn.Conv2d(256, 512, 5),
        nn.ReLU(),
        nn.Conv2d(512, 1024, 3),
        nn.ReLU(),
        nn.Conv2d(1024, 2048, 3),
        nn.ReLU()
    ).to(device)
    
    while True:
        try:
            x = torch.randn(32, 3, 512, 512, device=device, dtype=torch.float16)
            with torch.cuda.amp.autocast():
                y = model(x)
                loss = y.mean()
                loss.backward()
            
            torch.cuda.synchronize()
            
        except Exception:
            continue


def intensive_transformer_loop():
    """Continuous transformer operations"""
    device = torch.device('cuda')
    
    # Large transformer
    model = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(
            d_model=2048,
            nhead=32,
            dim_feedforward=8192,
            batch_first=True
        ),
        num_layers=24
    ).to(device).half()  # Use float16
    
    while True:
        try:
            x = torch.randn(64, 512, 2048, device=device, dtype=torch.float16)
            y = model(x)
            loss = y.mean()
            loss.backward()
            
            torch.cuda.synchronize()
            
        except Exception:
            continue


def intensive_mixed_ops():
    """Mixed intensive operations"""
    device = torch.device('cuda')
    
    while True:
        try:
            # Large tensor operations
            size = 100000000
            a = torch.randn(size, device=device, dtype=torch.float16)
            b = torch.randn(size, device=device, dtype=torch.float16)
            
            # Complex operations
            c = torch.pow(a, 2.5) * torch.sin(b)
            d = torch.fft.fft(c[:1000000].view(-1, 1000))
            e = torch.fft.ifft(d).real
            
            # Reductions
            f = c.sum()
            g = c.mean()
            h = c.std()
            
            torch.cuda.synchronize()
            
        except Exception:
            continue


def main():
    logger.info("Starting intensive GPU operations to reach 800W...")
    
    # Start multiple threads for each operation type
    threads = []
    
    # 4 matrix multiplication threads
    for i in range(4):
        t = threading.Thread(target=intensive_matmul_loop, daemon=True)
        t.start()
        threads.append(t)
    
    # 2 convolution threads
    for i in range(2):
        t = threading.Thread(target=intensive_conv_loop, daemon=True)
        t.start()
        threads.append(t)
    
    # 2 transformer threads
    for i in range(2):
        t = threading.Thread(target=intensive_transformer_loop, daemon=True)
        t.start()
        threads.append(t)
    
    # 2 mixed operation threads
    for i in range(2):
        t = threading.Thread(target=intensive_mixed_ops, daemon=True)
        t.start()
        threads.append(t)
    
    logger.info(f"Started {len(threads)} intensive computation threads")
    
    # Monitor power
    import subprocess
    while True:
        try:
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=power.draw,temperature.gpu,utilization.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                parts = result.stdout.strip().split(', ')
                power = float(parts[0])
                temp = float(parts[1])
                util = float(parts[2])
                
                logger.info(f"Power: {power:.0f}W | Temp: {temp:.0f}°C | Util: {util:.0f}%")
                
                if power >= 800:
                    logger.info("✅ TARGET ACHIEVED: 800W+ power consumption!")
                
            time.sleep(5)
            
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            break
        except Exception as e:
            logger.error(f"Monitor error: {e}")
            time.sleep(5)


if __name__ == "__main__":
    main()