#!/usr/bin/env python3
"""
Check progress of JGB scraping
"""

import os
import json
from pathlib import Path
from datetime import datetime

def check_progress():
    print("JGB Scraping Progress Report")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check directories
    base_dir = Path("jgb_complete_collection")
    
    if base_dir.exists():
        # Count audio files
        audio_dir = base_dir / "audio"
        if audio_dir.exists():
            show_dirs = list(audio_dir.iterdir())
            mp3_files = list(audio_dir.rglob("*.mp3"))
            
            print(f"Shows downloaded: {len(show_dirs)}")
            print(f"MP3 files: {len(mp3_files)}")
            
            # Calculate total size
            total_size = sum(f.stat().st_size for f in mp3_files)
            print(f"Total size: {total_size / 1e9:.2f} GB")
            
            # Show recent downloads
            print("\nRecent downloads:")
            sorted_files = sorted(mp3_files, key=lambda f: f.stat().st_mtime, reverse=True)
            for f in sorted_files[:10]:
                size_mb = f.stat().st_size / 1e6
                print(f"  - {f.name} ({size_mb:.1f} MB)")
        
        # Check embeddings
        embeddings_dir = base_dir / "embeddings"
        if embeddings_dir.exists():
            embedding_files = list(embeddings_dir.glob("*.h5"))
            print(f"\nEmbeddings created: {len(embedding_files)}")
        
        # Check metadata
        metadata_dir = base_dir / "metadata"
        if metadata_dir.exists():
            metadata_files = list(metadata_dir.glob("*.json"))
            print(f"Metadata files: {len(metadata_files)}")
        
        # Check summary
        summary_file = base_dir / "processing_summary.json"
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            print(f"\nLast summary update: {summary.get('timestamp', 'N/A')}")
            print(f"Total processed: {summary.get('total_processed', 0)}")
            print(f"Total failed: {summary.get('total_failed', 0)}")
    else:
        print("Collection directory not found!")
    
    # Check if still running
    import subprocess
    try:
        result = subprocess.run(['pgrep', '-f', 'start_batch_scraping.py'], 
                              capture_output=True, text=True)
        if result.stdout.strip():
            print("\n✓ Scraping is still running (PID: {})".format(result.stdout.strip()))
        else:
            print("\n✗ Scraping is not currently running")
    except:
        pass
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    check_progress()