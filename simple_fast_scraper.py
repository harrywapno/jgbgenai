#!/usr/bin/env python3
"""
Simple Fast Scraper - Works with existing tools
"""

import subprocess
import sys
import time

def main():
    print("=" * 80)
    print("SIMPLE FAST SCRAPER - Using existing enhanced scraper")
    print("=" * 80)
    
    # Check if enhanced scraper is already running
    result = subprocess.run(['pgrep', '-f', 'enhanced_scraper_runner.py'], capture_output=True)
    
    if result.returncode == 0:
        print("✓ Enhanced scraper already running!")
        print(f"PID: {result.stdout.decode().strip()}")
    else:
        print("Starting enhanced scraper...")
        # The enhanced scraper was already restarted by the system
        subprocess.Popen([sys.executable, 'enhanced_scraper_runner.py'], 
                        stdout=open('enhanced_scraper.log', 'a'),
                        stderr=subprocess.STDOUT)
        print("✓ Enhanced scraper started")
    
    # Show current progress
    print("\nChecking current progress...")
    time.sleep(2)
    
    # Count shows
    original_count = subprocess.run(
        ['ls', 'jgb_complete_collection/metadata/', '2>/dev/null', '|', 'wc', '-l'],
        shell=True, capture_output=True, text=True
    )
    
    if original_count.returncode == 0:
        count = int(original_count.stdout.strip() or 0)
        print(f"\nShows downloaded: {count} / 2080")
        print(f"Progress: {count/2080*100:.1f}%")
        
        # Estimate time
        # Enhanced scraper was running for ~45 minutes and got 66 shows
        # That's about 1.5 shows per minute
        rate = 1.5
        remaining = 2080 - count
        hours = remaining / rate / 60
        
        print(f"\nEstimated time remaining: {hours:.1f} hours")
        print(f"Estimated completion: tomorrow ~{int(22 + hours) % 24}:00")
    
    print("\nThe enhanced scraper is downloading with:")
    print("- ASX playlist parsing ✓")
    print("- Setlist parsing ✓") 
    print("- Text context for embeddings ✓")
    print("- 8 parallel workers ✓")
    
    print("\nTo monitor progress, run:")
    print("  watch -n 5 'ls jgb_complete_collection/metadata/*.json | wc -l'")
    
    print("\nTo see latest downloads:")
    print("  tail -f enhanced_scraper_runner.py.log")

if __name__ == "__main__":
    main()