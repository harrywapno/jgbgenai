#!/usr/bin/env python3
"""
JGB Audio Scraper - Fast Batch Collection
========================================

Quick scraper for Jerry Garcia Band audio from archive.org and other sources.
Designed for rapid collection building to reach 270GB target.
"""

import os
import sys
import requests
from bs4 import BeautifulSoup
import json
import concurrent.futures
from pathlib import Path
import time
from urllib.parse import urljoin, urlparse
import hashlib
from tqdm import tqdm

class JGBScraper:
    def __init__(self, base_dir="jgb_collection"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
        })
        
    def scrape_archive_org(self):
        """Scrape Jerry Garcia Band shows from archive.org"""
        print("🎸 Scraping archive.org for JGB shows...")
        
        # Archive.org API for JGB
        search_urls = [
            "https://archive.org/advancedsearch.php?q=creator%3A%22Jerry+Garcia+Band%22&fl%5B%5D=identifier&fl%5B%5D=title&fl%5B%5D=date&sort%5B%5D=date+asc&rows=500&page=1&output=json",
            "https://archive.org/advancedsearch.php?q=subject%3A%22Jerry+Garcia%22+AND+collection%3A%22etree%22&fl%5B%5D=identifier&fl%5B%5D=title&fl%5B%5D=date&sort%5B%5D=date+asc&rows=500&page=1&output=json",
            "https://archive.org/advancedsearch.php?q=title%3A%22Garcia%22+AND+mediatype%3A%22audio%22&fl%5B%5D=identifier&fl%5B%5D=title&fl%5B%5D=date&sort%5B%5D=date+asc&rows=500&page=1&output=json"
        ]
        
        shows = []
        for url in search_urls:
            try:
                response = self.session.get(url, timeout=30)
                data = response.json()
                shows.extend(data.get('response', {}).get('docs', []))
                print(f"Found {len(data.get('response', {}).get('docs', []))} items from search")
            except Exception as e:
                print(f"Error fetching from {url}: {e}")
        
        print(f"Total JGB items found: {len(shows)}")
        return shows
    
    def download_show(self, show_id, show_info):
        """Download a complete show from archive.org"""
        try:
            # Get show metadata
            metadata_url = f"https://archive.org/metadata/{show_id}"
            response = self.session.get(metadata_url, timeout=30)
            metadata = response.json()
            
            # Create show directory
            date = show_info.get('date', 'unknown')
            show_dir = self.base_dir / f"{date}_{show_id}"
            show_dir.mkdir(exist_ok=True)
            
            # Get audio files
            files = metadata.get('files', [])
            audio_files = [f for f in files if f.get('format', '').lower() in ['flac', 'mp3', 'wav', 'ogg']]
            
            if not audio_files:
                print(f"No audio files found for {show_id}")
                return 0
            
            downloaded_size = 0
            for file_info in audio_files:
                filename = file_info.get('name', '')
                if not filename:
                    continue
                    
                file_url = f"https://archive.org/download/{show_id}/{filename}"
                local_path = show_dir / filename
                
                if local_path.exists():
                    downloaded_size += local_path.stat().st_size
                    continue
                
                try:
                    print(f"Downloading: {filename}")
                    response = self.session.get(file_url, stream=True, timeout=60)
                    response.raise_for_status()
                    
                    with open(local_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    
                    downloaded_size += local_path.stat().st_size
                    print(f"✓ Downloaded {filename} ({local_path.stat().st_size / 1e6:.1f} MB)")
                    
                except Exception as e:
                    print(f"Error downloading {filename}: {e}")
                    if local_path.exists():
                        local_path.unlink()
            
            return downloaded_size
            
        except Exception as e:
            print(f"Error processing show {show_id}: {e}")
            return 0
    
    def scrape_deadlists(self):
        """Scrape additional sources for JGB shows"""
        print("🎵 Scraping additional JGB sources...")
        
        # DeadLists.com has good JGB setlists/info
        try:
            url = "http://www.deadlists.com/garcia/garcia.htm"
            response = self.session.get(url, timeout=30)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract show dates and venues
            shows = []
            for link in soup.find_all('a', href=True):
                href = link.get('href', '')
                if 'garcia' in href and any(year in href for year in ['75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94', '95']):
                    shows.append({
                        'url': urljoin(url, href),
                        'text': link.text.strip()
                    })
            
            print(f"Found {len(shows)} potential JGB shows from deadlists")
            return shows
            
        except Exception as e:
            print(f"Error scraping deadlists: {e}")
            return []
    
    def batch_download(self, max_workers=5, target_size_gb=50):
        """Fast batch download of JGB collection"""
        print(f"🚀 Starting batch download (target: {target_size_gb}GB)")
        
        # Get shows from archive.org
        shows = self.scrape_archive_org()
        
        if not shows:
            print("No shows found to download")
            return
        
        # Sort by date for chronological collection
        shows.sort(key=lambda x: x.get('date', ''))
        
        total_downloaded = 0
        target_bytes = target_size_gb * 1e9
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for show in shows[:100]:  # Limit to first 100 shows for speed
                if total_downloaded >= target_bytes:
                    break
                    
                show_id = show.get('identifier', '')
                if show_id:
                    future = executor.submit(self.download_show, show_id, show)
                    futures.append((future, show_id))
            
            # Process completed downloads
            for future, show_id in tqdm(futures, desc="Downloading shows"):
                try:
                    size = future.result(timeout=300)  # 5 minute timeout per show
                    total_downloaded += size
                    print(f"✓ {show_id}: {size / 1e6:.1f} MB (Total: {total_downloaded / 1e9:.1f} GB)")
                    
                    if total_downloaded >= target_bytes:
                        print(f"🎯 Reached target size: {total_downloaded / 1e9:.1f} GB")
                        break
                        
                except Exception as e:
                    print(f"Error downloading {show_id}: {e}")
        
        print(f"📊 Download complete: {total_downloaded / 1e9:.1f} GB in {self.base_dir}")
        return total_downloaded
    
    def create_sample_collection(self):
        """Create a smaller sample collection for immediate testing"""
        print("🎪 Creating sample JGB collection...")
        
        sample_dir = self.base_dir / "sample_shows"
        sample_dir.mkdir(exist_ok=True)
        
        # Create some realistic sample shows with metadata
        sample_shows = [
            {
                'date': '1975-08-13',
                'venue': 'Great American Music Hall',
                'location': 'San Francisco, CA',
                'songs': ['Fire on the Mountain', 'Scarlet Begonias', 'Eyes of the World']
            },
            {
                'date': '1976-07-18', 
                'venue': 'Orpheum Theatre',
                'location': 'San Francisco, CA',
                'songs': ['Sugaree', 'Deal', 'Catfish John']
            },
            {
                'date': '1990-09-10',
                'venue': 'Madison Square Garden',
                'location': 'New York, NY', 
                'songs': ['Deal', 'The Way You Do The Things You Do', 'Tangled Up in Blue']
            }
        ]
        
        total_size = 0
        for show in sample_shows:
            show_dir = sample_dir / f"{show['date']}_{show['venue'].replace(' ', '_')}"
            show_dir.mkdir(exist_ok=True)
            
            # Create sample audio files (sine waves with different frequencies)
            import numpy as np
            import soundfile as sf
            
            for i, song in enumerate(show['songs']):
                # Generate 2-3 minutes of audio per song
                duration = 120 + np.random.randint(0, 60)
                sr = 44100
                t = np.linspace(0, duration, int(sr * duration))
                
                # Create complex audio with harmonics
                freq = 220 * (2 ** (i / 12))  # Different frequency per song
                audio = 0.3 * np.sin(2 * np.pi * freq * t)
                audio += 0.1 * np.sin(2 * np.pi * freq * 2 * t)
                audio += 0.05 * np.sin(2 * np.pi * freq * 3 * t)
                
                # Add some variation
                audio += 0.02 * np.random.randn(len(audio))
                
                filename = f"{song.replace(' ', '_').lower()}.flac"
                filepath = show_dir / filename
                
                sf.write(str(filepath), audio, sr)
                file_size = filepath.stat().st_size
                total_size += file_size
                
                print(f"✓ Created {filename} ({file_size / 1e6:.1f} MB)")
            
            # Create setlist file
            setlist_file = show_dir / "setlist.txt"
            with open(setlist_file, 'w') as f:
                f.write(f"Jerry Garcia Band\n")
                f.write(f"{show['date']} - {show['venue']}, {show['location']}\n\n")
                f.write("Setlist:\n")
                for song in show['songs']:
                    f.write(f"- {song}\n")
        
        print(f"📁 Sample collection created: {total_size / 1e6:.1f} MB in {sample_dir}")
        return str(sample_dir)

def main():
    scraper = JGBScraper()
    
    print("JGB Audio Scraper - Fast Collection Builder")
    print("=" * 50)
    
    # Create sample collection for immediate use
    sample_path = scraper.create_sample_collection()
    
    # Try to download real shows (comment out if not working)
    try:
        scraper.batch_download(max_workers=3, target_size_gb=5)  # Start with 5GB
    except Exception as e:
        print(f"Real download failed: {e}")
        print("Using sample collection for now")
    
    print(f"\n🎸 JGB Collection ready at: {scraper.base_dir}")
    print(f"Sample shows available at: {sample_path}")
    print("\nReady for remixing!")

if __name__ == "__main__":
    main()