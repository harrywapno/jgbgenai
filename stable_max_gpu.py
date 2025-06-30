#!/usr/bin/env python3
"""
Stable Maximum GPU Utilization for B200
======================================

Pushes B200 to 150GB memory and 800W power consumption stably.
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

import torch
import torch.nn as nn
import torch.cuda as cuda
import numpy as np
from pathlib import Path
import logging
import time
import threading
import gc

torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StableB200Maximizer:
    """Stable maximum utilization of B200 GPU"""
    
    def __init__(self):
        self.device = torch.device('cuda')
        self.stop_flag = False
        
        # GPU properties
        props = torch.cuda.get_device_properties(0)
        self.total_memory_gb = props.total_memory / 1024**3
        logger.info(f"B200 GPU: {props.name} with {self.total_memory_gb:.1f}GB memory")
        
        # Storage for allocated tensors
        self.memory_blocks = []
        self.models = {}
        self.threads = []
        
    def allocate_memory_blocks(self, target_gb=150):
        """Allocate memory in stable chunks"""
        logger.info(f"Allocating {target_gb}GB of GPU memory...")
        
        # Allocate in 5GB chunks for stability
        chunk_size_gb = 5
        num_chunks = int(target_gb / chunk_size_gb)
        
        for i in range(num_chunks):
            try:
                # Calculate tensor size for chunk (float32 = 4 bytes)
                elements = int(chunk_size_gb * 1024 * 1024 * 1024 / 4)
                
                # Allocate tensor
                tensor = torch.randn(elements, device=self.device, dtype=torch.float32)
                self.memory_blocks.append(tensor)
                
                # Log progress
                allocated = (i + 1) * chunk_size_gb
                logger.info(f"Allocated {allocated}GB / {target_gb}GB")
                
                # Small computation to ensure allocation
                tensor *= 1.0001
                
            except RuntimeError as e:
                logger.warning(f"Could not allocate chunk {i+1}: {e}")
                break
        
        # Check actual allocation
        allocated_gb = torch.cuda.memory_allocated() / 1024**3
        logger.info(f"✓ Successfully allocated {allocated_gb:.1f}GB")
        
    def create_compute_models(self):
        """Create models optimized for high power consumption"""
        logger.info("Creating compute-intensive models...")
        
        # 1. Medium-sized transformer (stable)
        self.models['transformer'] = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=1024,
                nhead=16,
                dim_feedforward=4096,
                batch_first=True
            ),
            num_layers=8
        ).to(self.device)
        
        # 2. CNN stack
        self.models['cnn'] = nn.Sequential(
            nn.Conv2d(3, 64, 7, padding=3),
            nn.ReLU(),
            nn.Conv2d(64, 128, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1)
        ).to(self.device)
        
        # 3. MLP network
        self.models['mlp'] = nn.Sequential(
            nn.Linear(10000, 5000),
            nn.ReLU(),
            nn.Linear(5000, 5000),
            nn.ReLU(),
            nn.Linear(5000, 5000),
            nn.ReLU(),
            nn.Linear(5000, 10000)
        ).to(self.device)
        
        logger.info(f"✓ Created {len(self.models)} compute models")
        
    def compute_thread_matmul(self):
        """Thread for continuous matrix multiplication"""
        logger.info("Starting matrix multiplication thread...")
        
        sizes = [2048, 3072, 4096, 5120]
        
        while not self.stop_flag:
            try:
                size = np.random.choice(sizes)
                batch = 32
                
                # Allocate matrices
                A = torch.randn(batch, size, size, device=self.device, dtype=torch.float32)
                B = torch.randn(batch, size, size, device=self.device, dtype=torch.float32)
                
                # Chain operations
                C = torch.bmm(A, B)
                D = torch.bmm(C, A.transpose(1, 2))
                E = torch.bmm(D, B.transpose(1, 2))
                
                # Ensure computation completes
                _ = E.sum().item()
                
                # Update memory blocks periodically
                if np.random.random() < 0.1:
                    idx = np.random.randint(0, len(self.memory_blocks))
                    self.memory_blocks[idx] *= 1.0001
                
            except Exception as e:
                logger.error(f"MatMul error: {e}")
                time.sleep(0.1)
    
    def compute_thread_conv(self):
        """Thread for continuous convolution operations"""
        logger.info("Starting convolution thread...")
        
        if 'cnn' not in self.models:
            return
        
        while not self.stop_flag:
            try:
                batch = 16
                size = 256
                
                # Generate input
                x = torch.randn(batch, 3, size, size, device=self.device)
                
                # Run through CNN multiple times
                y = self.models['cnn'](x)
                z = self.models['cnn'](y)
                w = self.models['cnn'](z)
                
                # Force sync
                _ = w.mean().item()
                
            except Exception as e:
                logger.error(f"Conv error: {e}")
                time.sleep(0.1)
    
    def compute_thread_transformer(self):
        """Thread for transformer operations"""
        logger.info("Starting transformer thread...")
        
        if 'transformer' not in self.models:
            return
        
        while not self.stop_flag:
            try:
                batch = 64
                seq_len = 256
                d_model = 1024
                
                # Generate sequences
                x = torch.randn(batch, seq_len, d_model, device=self.device)
                
                # Run transformer
                y = self.models['transformer'](x)
                
                # Self-attention computation
                scores = torch.bmm(y, y.transpose(1, 2)) / np.sqrt(d_model)
                attn = torch.softmax(scores, dim=-1)
                output = torch.bmm(attn, y)
                
                # Force sync
                _ = output.sum().item()
                
            except Exception as e:
                logger.error(f"Transformer error: {e}")
                time.sleep(0.1)
    
    def compute_thread_mixed(self):
        """Thread for mixed operations"""
        logger.info("Starting mixed operations thread...")
        
        while not self.stop_flag:
            try:
                # FFT on memory blocks
                if self.memory_blocks:
                    idx = np.random.randint(0, len(self.memory_blocks))
                    chunk = self.memory_blocks[idx][:1024*1024].view(-1, 1024)
                    
                    fft = torch.fft.fft(chunk)
                    ifft = torch.fft.ifft(fft).real
                    
                    # Update chunk
                    self.memory_blocks[idx][:1024*1024] = ifft.view(-1)
                
                # MLP operations
                if 'mlp' in self.models:
                    x = torch.randn(128, 10000, device=self.device)
                    y = self.models['mlp'](x)
                    _ = y.mean().item()
                
                # Element-wise operations on large tensors
                if len(self.memory_blocks) >= 2:
                    size = min(100000000, self.memory_blocks[0].shape[0])
                    a = self.memory_blocks[0][:size]
                    b = self.memory_blocks[1][:size]
                    
                    c = torch.pow(a, 2) + torch.pow(b, 2)
                    d = torch.sqrt(torch.abs(c) + 1e-6)
                    e = torch.log(d + 1)
                    
                    self.memory_blocks[0][:size] = e
                
            except Exception as e:
                logger.error(f"Mixed ops error: {e}")
                time.sleep(0.1)
    
    def start_compute_threads(self):
        """Start all computation threads"""
        logger.info("Starting computation threads...")
        
        # Create threads
        threads = [
            threading.Thread(target=self.compute_thread_matmul, daemon=True),
            threading.Thread(target=self.compute_thread_matmul, daemon=True),  # 2x for more load
            threading.Thread(target=self.compute_thread_conv, daemon=True),
            threading.Thread(target=self.compute_thread_transformer, daemon=True),
            threading.Thread(target=self.compute_thread_mixed, daemon=True),
        ]
        
        # Start all threads
        for thread in threads:
            thread.start()
            self.threads.append(thread)
        
        logger.info(f"✓ Started {len(threads)} computation threads")
    
    def monitor_and_log(self):
        """Monitor GPU status"""
        import subprocess
        
        while not self.stop_flag:
            try:
                # Get GPU stats
                result = subprocess.run([
                    'nvidia-smi', '--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw',
                    '--format=csv,noheader,nounits'
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    parts = result.stdout.strip().split(', ')
                    gpu_util = float(parts[0])
                    memory_gb = float(parts[1]) / 1024
                    temp = float(parts[2])
                    power = float(parts[3])
                    
                    logger.info(
                        f"GPU Status - Util: {gpu_util:.0f}% | "
                        f"Memory: {memory_gb:.1f}GB | "
                        f"Temp: {temp:.0f}°C | "
                        f"Power: {power:.0f}W"
                    )
                    
                    # Adjust workload based on metrics
                    if power < 700 and gpu_util < 80:
                        logger.info("Increasing workload...")
                        # Trigger more operations
                        self.compute_thread_matmul()
                
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"Monitor error: {e}")
                time.sleep(5)
    
    def run(self):
        """Main execution"""
        logger.info("=" * 80)
        logger.info("STABLE B200 MAXIMUM UTILIZATION SYSTEM")
        logger.info("=" * 80)
        
        try:
            # Step 1: Allocate memory
            self.allocate_memory_blocks(target_gb=150)
            
            # Step 2: Create models
            self.create_compute_models()
            
            # Step 3: Start computation threads
            self.start_compute_threads()
            
            # Step 4: Monitor
            logger.info("\nSystem running. Press Ctrl+C to stop.")
            self.monitor_and_log()
            
        except KeyboardInterrupt:
            logger.info("\nShutting down...")
            self.stop_flag = True
            
            # Wait for threads
            for thread in self.threads:
                thread.join(timeout=1)
            
            # Clear memory
            self.memory_blocks.clear()
            torch.cuda.empty_cache()
            gc.collect()
            
            logger.info("✓ Shutdown complete")


def main():
    maximizer = StableB200Maximizer()
    maximizer.run()


if __name__ == "__main__":
    main()