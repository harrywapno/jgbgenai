#!/usr/bin/env python3
"""
Test the updated scraper with setlist parsing
"""

import logging
from sugarmegs_scraper import SugarmegsScraper

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_scraper():
    """Test scraper functionality"""
    
    print("Testing JGB Scraper with Setlist Support")
    print("=" * 50)
    
    # Initialize scraper
    scraper = SugarmegsScraper(
        output_dir="test_scraping",
        max_workers=2,
        use_gpu=False  # CPU for testing
    )
    
    # Get index
    print("\n1. Scraping JGB index...")
    shows = scraper.scrape_jgb_index()
    print(f"Found {len(shows)} shows")
    
    # Test first few shows
    if shows:
        print("\n2. Testing first 3 shows:")
        for i, show in enumerate(shows[:3]):
            print(f"\n--- Show {i+1} ---")
            print(f"Date: {show['date']}")
            print(f"Venue: {show['venue']}")
            print(f"ASX: {show.get('asx_name', 'N/A')}")
            print(f"Setlist URL: {show.get('setlist_url', 'N/A')}")
            print(f"MP3 URL: {show.get('mp3_url', 'N/A')}")
            
            # Get audio files
            audio_files = scraper.scrape_show_page(show)
            print(f"Audio files found: {len(audio_files)}")
            
            for j, file_info in enumerate(audio_files[:5]):  # First 5 files
                print(f"  {j+1}. {file_info['filename']} ({file_info['type']})")
                if 'track_name' in file_info:
                    print(f"     Track: {file_info['track_name']}")
            
            # Check setlist
            if 'setlist' in show and show['setlist']:
                print(f"\nSetlist parsed: {len(show['setlist'])} tracks")
                for track in show['setlist'][:5]:  # First 5 tracks
                    print(f"  - {track['track_name']}")
    
    print("\n" + "=" * 50)
    print("Test complete!")

if __name__ == "__main__":
    test_scraper()