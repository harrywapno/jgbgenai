#!/usr/bin/env python3
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

import torch
torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

from ai_remix_generator import AIRemixGenerator
import logging
import time
import random
import numpy as np
import concurrent.futures
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MaxGPU-AIRemix")

class MaxGPURemixGenerator(AIRemixGenerator):
    """Enhanced remix generator using maximum GPU"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Pre-allocate large GPU buffers
        logger.info("Allocating 50GB GPU memory for audio cache...")
        self.gpu_audio_cache = {}
        self.preload_audio_to_gpu()
        
        # Create larger models
        self._init_heavy_models()
        
        # Multiple parallel generators
        self.num_parallel = 4
        self.generators = []
        for i in range(self.num_parallel):
            gen = MusicgenForConditionalGeneration.from_pretrained(
                "facebook/musicgen-medium"  # Larger model
            ).to(self.device)
            self.generators.append(gen)
    
    def preload_audio_to_gpu(self):
        """Preload all available audio to GPU memory"""
        audio_files = list(self.audio_dir.rglob("*.mp3"))[:500]  # First 500 files
        
        for i, audio_file in enumerate(audio_files):
            try:
                # Load and convert to GPU tensor
                y, sr = librosa.load(str(audio_file), sr=44100, duration=180)
                tensor = torch.from_numpy(y).float().to(self.device)
                self.gpu_audio_cache[str(audio_file)] = tensor
                
                if i % 50 == 0:
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    logger.info(f"Loaded {i+1} files, GPU memory: {allocated:.1f}GB")
                    
            except Exception as e:
                logger.error(f"Error loading {audio_file}: {e}")
        
        logger.info(f"Preloaded {len(self.gpu_audio_cache)} audio files to GPU")
    
    def _init_heavy_models(self):
        """Initialize computationally heavy models"""
        # Large transformer for audio analysis
        self.analysis_transformer = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=1024, nhead=16, dim_feedforward=4096,
                batch_first=True, dropout=0.1
            ),
            num_layers=12
        ).to(self.device)
        
        # Deep CNN for spectral processing
        layers = []
        in_channels = 1
        for out_channels in [64, 128, 256, 512, 1024, 512, 256, 128, 64, 1]:
            layers.extend([
                torch.nn.Conv2d(in_channels, out_channels, 3, padding=1),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.ReLU() if out_channels > 1 else torch.nn.Tanh()
            ])
            in_channels = out_channels
        
        self.spectral_processor = torch.nn.Sequential(*layers).to(self.device)
        
        logger.info("✓ Heavy models initialized")
    
    def generate_parallel_remixes(self):
        """Generate multiple remixes in parallel"""
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_parallel) as executor:
            futures = []
            
            for i in range(self.num_parallel):
                style = random.choice(["psychedelic", "energetic", "mellow", "classic"])
                seed_track = random.choice(list(self.similarity_engine.embeddings_cache.keys()))
                
                future = executor.submit(
                    self.create_ai_remix,
                    seed_track=seed_track,
                    style=style,
                    duration_minutes=random.randint(10, 20),
                    temperature=random.uniform(0.7, 0.95)
                )
                futures.append(future)
            
            # Wait for completion
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    logger.info(f"✓ Parallel remix completed: {result['output_file']}")
                except Exception as e:
                    logger.error(f"Parallel remix failed: {e}")

# Initialize with maximum GPU usage
generator = MaxGPURemixGenerator(
    embeddings_dir="jgb_complete_collection/embeddings",
    audio_dir="jgb_complete_collection/audio",
    use_gpu=True
)

# Wait for embeddings
while len(generator.similarity_engine.embeddings_cache) < 10:
    logger.info(f"Waiting for embeddings... Current: {len(generator.similarity_engine.embeddings_cache)}")
    time.sleep(30)
    generator.similarity_engine = generator.similarity_engine.__class__(generator.embeddings_dir)

logger.info(f"✓ Ready with {len(generator.similarity_engine.embeddings_cache)} embeddings")
logger.info(f"✓ GPU audio cache: {len(generator.gpu_audio_cache)} files")

# Continuous parallel generation
while True:
    try:
        # Log GPU usage
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"GPU Memory: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")
        
        # Generate multiple remixes in parallel
        generator.generate_parallel_remixes()
        
        # Run intensive computations to increase power draw
        for _ in range(10):
            # Large matrix operations
            size = 8192
            a = torch.randn(size, size, device=generator.device)
            b = torch.randn(size, size, device=generator.device)
            c = torch.matmul(a, b)
            d = torch.matmul(c, a.T)
            
            # FFT operations on audio
            if generator.gpu_audio_cache:
                audio_tensor = next(iter(generator.gpu_audio_cache.values()))
                fft = torch.fft.fft(audio_tensor)
                ifft = torch.fft.ifft(fft).real
        
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache()
        
    except Exception as e:
        logger.error(f"Error in main loop: {e}")
        time.sleep(10)
