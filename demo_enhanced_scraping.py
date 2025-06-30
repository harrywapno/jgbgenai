#!/usr/bin/env python3
"""
Demo of enhanced scraping with ASX and setlist
"""

import logging
from sugarmegs_scraper import SugarmegsScraper

logging.basicConfig(level=logging.INFO)

def demo_enhanced_scraping():
    print("Demo: Enhanced Scraping with Text Context")
    print("=" * 60)
    
    # Initialize scraper
    scraper = SugarmegsScraper(
        output_dir="jgb_enhanced_demo",
        max_workers=1,
        use_gpu=False
    )
    
    # Get a few shows from the index
    shows = scraper.scrape_jgb_index()[:3]
    
    for show in shows:
        print(f"\nProcessing: {show['date']} - {show['venue']}")
        print(f"ASX URL: {show.get('asx_url', 'N/A')}")
        print(f"Setlist URL: {show.get('setlist_url', 'N/A')}")
        
        # Process the show
        result = scraper.process_show(show)
        
        if result:
            print(f"✓ Successfully processed show")
            print(f"  - Tracks processed: {result.get('processed_tracks', 0)}")
            
            # Check what was downloaded
            show_dir = scraper.dirs['audio'] / f"{show['date']}_{result['show_id'].split('_')[1]}"
            if show_dir.exists():
                files = list(show_dir.iterdir())
                print(f"  - Files in directory: {len(files)}")
                for f in files:
                    print(f"    • {f.name}")
            
            # Check metadata
            metadata_file = scraper.dirs['metadata'] / f"{result['show_id']}_metadata.json"
            if metadata_file.exists():
                import json
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                print(f"\n  Metadata:")
                print(f"    - Era: {metadata.get('era', 'N/A')}")
                print(f"    - ASX tracks: {len(metadata.get('asx_data', {}).get('entries', []))}")
                print(f"    - Setlist songs: {len(metadata.get('setlist', {}).get('songs', []))}")
                
                # Show track info
                for track in metadata.get('track_info', [])[:3]:
                    print(f"\n    Track: {track['file']}")
                    print(f"      - Song: {track.get('song_title', 'Unknown')}")
                    print(f"      - ASX title: {track.get('asx_title', 'N/A')}")
                    print(f"      - Tempo: {track['tempo']:.1f} BPM")
                    print(f"      - Key: {track['key']}")
        
        break  # Just do one for demo
    
    print("\n" + "=" * 60)
    print("This demonstrates how the enhanced scraper:")
    print("1. Downloads ASX playlist files")
    print("2. Downloads and parses setlist HTML pages") 
    print("3. Stores text context with embeddings")
    print("4. Enables reinforcement of same songs across shows")

if __name__ == "__main__":
    demo_enhanced_scraping()