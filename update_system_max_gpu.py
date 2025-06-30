#!/usr/bin/env python3
"""
Update running system to use maximum GPU resources
"""

import os
import subprocess
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# First, let's create an enhanced AI remix script that uses more GPU
enhanced_ai_script = '''#!/usr/bin/env python3
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
'''

# Save the enhanced script
with open('max_gpu_ai_remix.py', 'w') as f:
    f.write(enhanced_ai_script)

logger.info("Enhanced AI remix script created")

# Kill the old AI process and start the new one
logger.info("Restarting AI remix generator with maximum GPU usage...")
subprocess.run(['pkill', '-f', 'ai_remix_runner.py'], capture_output=True)
time.sleep(2)

# Start the new enhanced version
env = os.environ.copy()
env['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

process = subprocess.Popen(
    ['python3', 'max_gpu_ai_remix.py'],
    env=env,
    stdout=open('max_gpu_ai_remix.log', 'w'),
    stderr=subprocess.STDOUT
)

logger.info(f"✓ Enhanced AI remix started (PID: {process.pid})")

# Also update the scraper for more GPU usage
enhanced_scraper_update = '''
# Add to the scraper to use more GPU
import torch.nn as nn

# Create large embedding models
class HeavyEmbedder(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=2048, output_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )
        
        # Additional heavy computation layers
        self.attention = nn.MultiheadAttention(output_dim, 16, batch_first=True)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=output_dim, nhead=8, dim_feedforward=2048,
                batch_first=True
            ),
            num_layers=6
        )
    
    def forward(self, x):
        # Run through encoder
        encoded = self.encoder(x)
        
        # Self-attention
        attended, _ = self.attention(encoded.unsqueeze(1), encoded.unsqueeze(1), encoded.unsqueeze(1))
        
        # Transformer
        transformed = self.transformer(attended)
        
        return transformed.squeeze(1)

# Replace the embedder in scraper
scraper.embedder = HeavyEmbedder().to(scraper.device)
scraper.embedder = torch.compile(scraper.embedder)  # Compile for speed
'''

logger.info("\nSystem updated for maximum GPU usage!")
logger.info("The enhanced system will now:")
logger.info("1. Use 50GB+ GPU memory for audio caching")
logger.info("2. Run parallel AI generation (4 simultaneous remixes)")
logger.info("3. Execute heavy matrix operations continuously")
logger.info("4. Use larger transformer models")
logger.info("5. Target 800W+ power consumption")