#!/usr/bin/env python3
"""
Real-time Status Dashboard for JGB Full System
"""

import os
import time
import subprocess
from pathlib import Path
from datetime import datetime
import json

def get_gpu_stats():
    """Get current GPU statistics"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw', 
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        
        if result.returncode == 0:
            stats = result.stdout.strip().split(', ')
            return {
                'gpu_util': float(stats[0]),
                'memory_used_gb': float(stats[1]) / 1024,
                'memory_total_gb': float(stats[2]) / 1024,
                'temperature': float(stats[3]),
                'power_watts': float(stats[4])
            }
    except:
        pass
    return None

def get_scraping_progress():
    """Get current scraping progress"""
    metadata_dir = Path("jgb_complete_collection/metadata")
    if metadata_dir.exists():
        scraped = len(list(metadata_dir.glob("*_metadata.json")))
        return scraped
    return 0

def get_embeddings_count():
    """Get number of embeddings"""
    embeddings_dir = Path("jgb_complete_collection/embeddings")
    if embeddings_dir.exists():
        count = len(list(embeddings_dir.glob("*.h5")))
        return count
    return 0

def get_remixes_count():
    """Count generated remixes"""
    remix_count = len(list(Path(".").glob("ai_remix_*.wav")))
    remix_count += len(list(Path(".").glob("ai_era_journey_*.wav")))
    return remix_count

def display_dashboard():
    """Display real-time dashboard"""
    print("\033[2J\033[H")  # Clear screen
    
    while True:
        # Get current stats
        gpu_stats = get_gpu_stats()
        scraped_shows = get_scraping_progress()
        embeddings = get_embeddings_count()
        remixes = get_remixes_count()
        
        # Clear and display
        print("\033[2J\033[H")  # Clear screen
        print("=" * 80)
        print("JGB FULL SYSTEM STATUS DASHBOARD".center(80))
        print("=" * 80)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # GPU Status
        print("🖥️  B200 GPU STATUS")
        print("-" * 40)
        if gpu_stats:
            util_bar = "█" * int(gpu_stats['gpu_util'] / 5) + "░" * (20 - int(gpu_stats['gpu_util'] / 5))
            mem_pct = (gpu_stats['memory_used_gb'] / gpu_stats['memory_total_gb']) * 100
            mem_bar = "█" * int(mem_pct / 5) + "░" * (20 - int(mem_pct / 5))
            
            print(f"GPU Utilization: [{util_bar}] {gpu_stats['gpu_util']:.1f}%")
            print(f"Memory Usage:    [{mem_bar}] {gpu_stats['memory_used_gb']:.1f}/{gpu_stats['memory_total_gb']:.1f} GB ({mem_pct:.1f}%)")
            print(f"Temperature:     {gpu_stats['temperature']:.0f}°C")
            print(f"Power Draw:      {gpu_stats['power_watts']:.0f}W / 1000W")
        else:
            print("GPU stats unavailable")
        print()
        
        # Scraping Progress
        print("📥 SCRAPING PROGRESS")
        print("-" * 40)
        progress_pct = (scraped_shows / 2080) * 100
        progress_bar = "█" * int(progress_pct / 5) + "░" * (20 - int(progress_pct / 5))
        print(f"Shows Scraped:   [{progress_bar}] {scraped_shows}/2080 ({progress_pct:.1f}%)")
        
        # Calculate ETA
        if scraped_shows > 100:
            # Estimate based on current rate (rough)
            hours_remaining = (2080 - scraped_shows) / 10  # ~10 shows per hour
            print(f"Estimated Time:  {hours_remaining:.1f} hours remaining")
        print()
        
        # Embeddings
        print("🧠 EMBEDDINGS")
        print("-" * 40)
        print(f"Generated:       {embeddings} embeddings")
        print(f"Tracks/Show:     {embeddings / max(scraped_shows, 1):.1f} average")
        print()
        
        # AI Remixes
        print("🎵 AI REMIXES")
        print("-" * 40)
        print(f"Generated:       {remixes} remixes")
        print(f"Styles:          Psychedelic, Energetic, Mellow, Classic")
        print(f"Era Journeys:    {len(list(Path('.').glob('ai_era_journey_*.wav')))}")
        print()
        
        # Active Processes
        print("⚙️  ACTIVE PROCESSES")
        print("-" * 40)
        processes = [
            ("Enhanced Scraper", "enhanced_scraper_runner.py"),
            ("AI Remix Generator", "ai_remix_runner.py"),
            ("GPU Monitor", "gpu_monitor.py")
        ]
        
        for name, script in processes:
            result = subprocess.run(['pgrep', '-f', script], capture_output=True, text=True)
            if result.returncode == 0:
                pid = result.stdout.strip()
                print(f"✓ {name:<20} PID: {pid}")
            else:
                print(f"✗ {name:<20} NOT RUNNING")
        
        print()
        print("=" * 80)
        print("Press Ctrl+C to exit dashboard (processes will continue running)")
        
        time.sleep(5)  # Update every 5 seconds

if __name__ == "__main__":
    try:
        display_dashboard()
    except KeyboardInterrupt:
        print("\n\nDashboard closed. Background processes continue running.")
        print("To stop all processes, run: pkill -f 'enhanced_scraper|ai_remix|gpu_monitor'")