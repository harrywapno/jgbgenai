#!/usr/bin/env python3
"""
Scraper Status Dashboard
"""

import time
import subprocess
from pathlib import Path
from datetime import datetime, timedelta

def get_scraper_stats():
    """Get current scraping statistics"""
    stats = {
        'original': 0,
        'ultra_fast': 0,
        'total_size_gb': 0,
        'processes': []
    }
    
    # Count shows in different directories
    dirs = [
        ('jgb_complete_collection/metadata', 'original'),
        ('jgb_ultra_fast/metadata', 'ultra_fast')
    ]
    
    for dir_path, name in dirs:
        path = Path(dir_path)
        if path.exists():
            count = len(list(path.glob('*.json')))
            stats[name] = count
    
    # Calculate total size
    audio_dirs = [
        'jgb_complete_collection/audio',
        'jgb_ultra_fast/audio'
    ]
    
    for audio_dir in audio_dirs:
        if Path(audio_dir).exists():
            result = subprocess.run(['du', '-sb', audio_dir], capture_output=True, text=True)
            if result.returncode == 0:
                size_bytes = int(result.stdout.split()[0])
                stats['total_size_gb'] += size_bytes / 1024 / 1024 / 1024
    
    # Check running processes
    process_names = [
        'enhanced_scraper_runner.py',
        'ultra_fast_scraper.py',
        'fast_parallel_scraper.py',
        'turbo_scraper.py'
    ]
    
    for proc_name in process_names:
        result = subprocess.run(['pgrep', '-f', proc_name], capture_output=True, text=True)
        if result.returncode == 0:
            stats['processes'].append(proc_name.replace('.py', ''))
    
    return stats

def main():
    print("\033[2J\033[H")  # Clear screen
    
    start_time = datetime.now()
    
    while True:
        stats = get_scraper_stats()
        total_shows = stats['original'] + stats['ultra_fast']
        
        # Clear and display
        print("\033[2J\033[H")
        print("=" * 80)
        print("JGB SCRAPER STATUS DASHBOARD".center(80))
        print("=" * 80)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Overall progress
        progress_pct = (total_shows / 2080) * 100
        bar_length = 50
        filled = int(progress_pct * bar_length / 100)
        bar = "█" * filled + "░" * (bar_length - filled)
        
        print(f"OVERALL PROGRESS: [{bar}] {progress_pct:.1f}%")
        print(f"Total Shows: {total_shows} / 2080")
        print(f"Total Size: {stats['total_size_gb']:.1f} GB")
        print()
        
        # Breakdown by scraper
        print("SCRAPER BREAKDOWN:")
        print("-" * 40)
        print(f"Original Enhanced Scraper: {stats['original']} shows")
        print(f"Ultra Fast Scraper: {stats['ultra_fast']} shows")
        print()
        
        # Running processes
        print("ACTIVE SCRAPERS:")
        print("-" * 40)
        if stats['processes']:
            for proc in stats['processes']:
                print(f"✓ {proc}")
        else:
            print("⚠️  No scrapers running!")
        print()
        
        # Time estimates
        elapsed = datetime.now() - start_time
        if total_shows > 0 and elapsed.total_seconds() > 60:
            rate = total_shows / (elapsed.total_seconds() / 60)
            remaining = 2080 - total_shows
            eta_minutes = remaining / rate if rate > 0 else 0
            eta_time = datetime.now() + timedelta(minutes=eta_minutes)
            
            print("PERFORMANCE:")
            print("-" * 40)
            print(f"Download Rate: {rate:.1f} shows/minute")
            print(f"Time Remaining: {eta_minutes/60:.1f} hours")
            print(f"Estimated Completion: {eta_time.strftime('%H:%M tomorrow')}")
        
        # Check logs for errors
        print()
        print("RECENT ACTIVITY:")
        print("-" * 40)
        
        log_files = ['ultra_fast.log', 'turbo_scraper.log', 'enhanced_scraper_runner.py.log']
        for log_file in log_files:
            if Path(log_file).exists():
                result = subprocess.run(['tail', '-1', log_file], capture_output=True, text=True)
                if result.stdout:
                    print(f"{log_file}: {result.stdout.strip()[:60]}...")
        
        print()
        print("Press Ctrl+C to exit")
        
        time.sleep(5)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDashboard closed.")