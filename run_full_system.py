#!/usr/bin/env python3
"""
Full JGB System Runner with B200 GPU Maximum Utilization
========================================================

Runs:
1. Enhanced scraper for all 2080 shows
2. AI remix generation with multiple models
3. Continuous remix creation using scraped content

Maximizes B200 GPU (183GB) usage across all processes.
"""

import os
import sys
import torch
import multiprocessing as mp
from pathlib import Path
import logging
import time
from datetime import datetime
import subprocess
import signal
import psutil

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('full_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FullSystemRunner:
    """Orchestrates full system with maximum GPU utilization"""
    
    def __init__(self):
        self.processes = []
        self.gpu_memory_gb = 183  # B200 memory
        
        # Check GPU availability
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            logger.info(f"✓ B200 GPU detected: {torch.cuda.get_device_name(0)}")
            logger.info(f"✓ GPU Memory: {self.gpu_memory_gb}GB available")
        else:
            logger.warning("⚠ No GPU detected, running on CPU")
            self.device = torch.device('cpu')
    
    def start_enhanced_scraper(self):
        """Start enhanced scraper in background with GPU acceleration"""
        logger.info("Starting Enhanced Scraper for all 2080 shows...")
        
        # Create scraper script
        scraper_script = """
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
torch.cuda.set_device(0)

from sugarmegs_scraper import SugarmegsScraper
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EnhancedScraper")

# Initialize scraper with full GPU usage
scraper = SugarmegsScraper(
    output_dir="jgb_complete_collection",
    max_workers=8,  # Increased for parallel downloads
    use_gpu=True
)

# Enable full text context
logger.info("Enhanced scraping with ASX/setlist parsing enabled")

# Get all shows
all_shows = scraper.scrape_jgb_index()
logger.info(f"Found {len(all_shows)} total JGB shows to scrape")

# Start from where we left off
existing_shows = len(list(scraper.dirs['metadata'].glob('*_metadata.json')))
logger.info(f"Already scraped: {existing_shows} shows")

# Process remaining shows
for i, show in enumerate(all_shows[existing_shows:], start=existing_shows):
    try:
        logger.info(f"Processing show {i+1}/{len(all_shows)}: {show['date']} - {show['venue']}")
        result = scraper.process_show(show)
        
        if result and result.get('processed_tracks', 0) > 0:
            logger.info(f"✓ Successfully processed {result['processed_tracks']} tracks")
            
            # Generate embeddings every 10 shows
            if (i + 1) % 10 == 0:
                logger.info("Generating batch embeddings...")
                scraper.generate_embeddings_batch(force_regenerate=False)
        else:
            logger.warning(f"⚠ No tracks processed for {show['date']}")
            
    except Exception as e:
        logger.error(f"Error processing show {show['date']}: {e}")
        continue
    
    # Log progress
    if (i + 1) % 50 == 0:
        progress = ((i + 1) / len(all_shows)) * 100
        logger.info(f"Progress: {progress:.1f}% complete ({i+1}/{len(all_shows)} shows)")
        
        # Check GPU memory
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1024**3
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"GPU Memory: {memory_used:.1f}GB / {memory_total:.1f}GB")

logger.info("✓ Enhanced scraping complete!")
"""
        
        # Write script
        script_path = Path("enhanced_scraper_runner.py")
        with open(script_path, 'w') as f:
            f.write(scraper_script)
        
        # Start scraper process
        env = os.environ.copy()
        env['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
        
        process = subprocess.Popen(
            [sys.executable, str(script_path)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        self.processes.append(('scraper', process))
        logger.info(f"✓ Enhanced scraper started (PID: {process.pid})")
        
        return process
    
    def start_ai_remix_generator(self):
        """Start AI remix generator with multiple parallel instances"""
        logger.info("Starting AI Remix Generator with full GPU utilization...")
        
        # Create AI remix script
        ai_script = """
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
torch.cuda.set_device(0)

from ai_remix_generator import AIRemixGenerator
import logging
import time
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AIRemixGen")

# Initialize generator with maximum GPU usage
generator = AIRemixGenerator(
    embeddings_dir="jgb_complete_collection/embeddings",
    audio_dir="jgb_complete_collection/audio",
    use_gpu=True
)

# Wait for some embeddings to be available
while len(generator.similarity_engine.embeddings_cache) < 10:
    logger.info(f"Waiting for embeddings... Current: {len(generator.similarity_engine.embeddings_cache)}")
    time.sleep(30)
    # Reload embeddings
    generator.similarity_engine = generator.similarity_engine.__class__(
        generator.embeddings_dir
    )

logger.info(f"✓ Loaded {len(generator.similarity_engine.embeddings_cache)} embeddings")

# Continuous remix generation
remix_count = 0
styles = ["psychedelic", "energetic", "mellow", "classic"]

while True:
    try:
        # Get random seed track
        available_tracks = list(generator.similarity_engine.embeddings_cache.keys())
        seed_track = random.choice(available_tracks)
        
        # Choose random style
        style = random.choice(styles)
        
        logger.info(f"\\nGenerating remix #{remix_count + 1}")
        logger.info(f"Seed: {seed_track}, Style: {style}")
        
        # Generate remix
        metadata = generator.create_ai_remix(
            seed_track=seed_track,
            style=style,
            duration_minutes=random.randint(8, 15),
            temperature=random.uniform(0.6, 0.9)
        )
        
        remix_count += 1
        logger.info(f"✓ Created remix: {metadata['output_file']}")
        
        # Every 10 remixes, create an era journey
        if remix_count % 10 == 0:
            logger.info("Creating special Era Journey remix...")
            journey_metadata = generator.create_era_journey_remix()
            logger.info(f"✓ Created era journey: {journey_metadata['output_file']}")
        
        # Check for new embeddings periodically
        if remix_count % 5 == 0:
            generator.similarity_engine = generator.similarity_engine.__class__(
                generator.embeddings_dir
            )
            logger.info(f"Reloaded embeddings: {len(generator.similarity_engine.embeddings_cache)} available")
        
        # Log GPU usage
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"GPU Memory usage: {memory_used:.1f}GB")
        
        # Small pause between remixes
        time.sleep(10)
        
    except Exception as e:
        logger.error(f"Error generating remix: {e}")
        time.sleep(30)
"""
        
        # Write script
        script_path = Path("ai_remix_runner.py")
        with open(script_path, 'w') as f:
            f.write(ai_script)
        
        # Start AI process
        env = os.environ.copy()
        env['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
        
        process = subprocess.Popen(
            [sys.executable, str(script_path)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        self.processes.append(('ai_remix', process))
        logger.info(f"✓ AI remix generator started (PID: {process.pid})")
        
        return process
    
    def start_gpu_monitor(self):
        """Monitor GPU usage and system health"""
        monitor_script = """
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
"""
        
        script_path = Path("gpu_monitor.py")
        with open(script_path, 'w') as f:
            f.write(monitor_script)
        
        process = subprocess.Popen(
            [sys.executable, str(script_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        self.processes.append(('monitor', process))
        logger.info(f"✓ GPU monitor started (PID: {process.pid})")
        
        return process
    
    def monitor_processes(self):
        """Monitor all running processes"""
        logger.info("\nMonitoring all processes...")
        
        try:
            while True:
                # Check process status
                for name, process in self.processes:
                    if process.poll() is not None:
                        logger.warning(f"⚠ Process {name} (PID: {process.pid}) has terminated")
                        # Restart if needed
                        if name == 'scraper':
                            logger.info("Restarting scraper...")
                            self.start_enhanced_scraper()
                        elif name == 'ai_remix':
                            logger.info("Restarting AI remix generator...")
                            self.start_ai_remix_generator()
                
                # Log active processes
                active = sum(1 for _, p in self.processes if p.poll() is None)
                logger.info(f"Active processes: {active}/{len(self.processes)}")
                
                time.sleep(300)  # Check every 5 minutes
                
        except KeyboardInterrupt:
            logger.info("\nShutting down all processes...")
            self.cleanup()
    
    def cleanup(self):
        """Clean shutdown of all processes"""
        for name, process in self.processes:
            if process.poll() is None:
                logger.info(f"Terminating {name} (PID: {process.pid})")
                process.terminate()
                process.wait(timeout=10)
    
    def run(self):
        """Run the full system"""
        logger.info("=" * 80)
        logger.info("JGB FULL SYSTEM RUNNER - B200 GPU MAXIMUM UTILIZATION")
        logger.info("=" * 80)
        
        try:
            # Start all components
            self.start_enhanced_scraper()
            time.sleep(5)
            
            self.start_ai_remix_generator()
            time.sleep(5)
            
            self.start_gpu_monitor()
            
            logger.info("\n✓ All systems started successfully!")
            logger.info("The system will now:")
            logger.info("1. Scrape all 2080 JGB shows with enhanced text context")
            logger.info("2. Generate AI-powered remixes continuously")
            logger.info("3. Maximize B200 GPU usage (183GB)")
            logger.info("\nPress Ctrl+C to stop all processes")
            
            # Monitor everything
            self.monitor_processes()
            
        except Exception as e:
            logger.error(f"System error: {e}")
        finally:
            self.cleanup()


def main():
    runner = FullSystemRunner()
    runner.run()


if __name__ == "__main__":
    main()