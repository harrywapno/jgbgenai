#!/usr/bin/env python3
"""
Ultra Fast JGB Scraper - Direct Download Mode
============================================

Downloads MP3s directly without processing.
Embeddings can be generated later in batch.
"""

import os
import requests
from bs4 import BeautifulSoup
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import json
import hashlib
from urllib.parse import urljoin
import re

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


class UltraFastScraper:
    def __init__(self, output_dir="jgb_ultra_fast", max_workers=32):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.audio_dir = self.output_dir / "audio"
        self.audio_dir.mkdir(exist_ok=True)
        
        self.metadata_dir = self.output_dir / "metadata"
        self.metadata_dir.mkdir(exist_ok=True)
        
        self.max_workers = max_workers
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; JGB-Scraper/1.0)'
        })
        
        # Track progress
        self.completed = 0
        self.failed = 0
        self.total_size_gb = 0
        self.start_time = time.time()
        
        # Load existing
        self.existing_shows = set()
        for f in self.metadata_dir.glob("*.json"):
            self.existing_shows.add(f.stem)
    
    def get_all_shows(self):
        """Get all JGB shows from sugarmegs"""
        url = "http://tela.sugarmegs.org/alpha/j.html"
        response = self.session.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        shows = []
        # Find all Jerry Garcia Band links
        for link in soup.find_all('a', href=True):
            href = link['href']
            if 'jerrygarcia' in href.lower() and href.endswith('.asx'):
                # Extract date
                match = re.search(r'(\d{4})-(\d{2})-(\d{2})', href)
                if match:
                    year = int(match.group(1))
                    if 1975 <= year <= 1995:  # JGB years
                        date = f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
                        show_id = f"{date}_{hashlib.md5(href.encode()).hexdigest()[:8]}"
                        
                        if show_id not in self.existing_shows:
                            shows.append({
                                'date': date,
                                'url': urljoin(url, href),
                                'show_id': show_id
                            })
        
        return shows
    
    def download_mp3_direct(self, url, filepath):
        """Download MP3 file directly"""
        try:
            with self.session.get(url, stream=True, timeout=300) as r:
                if r.status_code == 200:
                    total_size = 0
                    with open(filepath, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=1024*1024):  # 1MB chunks
                            if chunk:
                                f.write(chunk)
                                total_size += len(chunk)
                    return total_size
        except Exception as e:
            logger.debug(f"Download failed {url}: {e}")
            return 0
    
    def process_show_ultra_fast(self, show_info):
        """Download show as fast as possible"""
        show_id = show_info['show_id']
        show_dir = self.audio_dir / show_id
        
        try:
            # Get ASX file
            asx_response = self.session.get(show_info['url'], timeout=30)
            asx_content = asx_response.text
            
            # Extract MP3 URLs - try multiple patterns
            mp3_urls = []
            
            # Pattern 1: href="..."
            mp3_urls.extend(re.findall(r'href="([^"]+\.mp3)"', asx_content, re.IGNORECASE))
            
            # Pattern 2: HREF="..."
            mp3_urls.extend(re.findall(r'HREF="([^"]+\.mp3)"', asx_content))
            
            # Pattern 3: ref href=...
            soup = BeautifulSoup(asx_content, 'xml')
            for ref in soup.find_all(['Ref', 'ref', 'REF']):
                href = ref.get('href') or ref.get('HREF')
                if href and href.endswith('.mp3'):
                    mp3_urls.append(href)
            
            # Remove duplicates
            mp3_urls = list(dict.fromkeys(mp3_urls))
            
            if not mp3_urls:
                logger.warning(f"No MP3s found for {show_id}")
                self.failed += 1
                return
            
            # Create show directory
            show_dir.mkdir(exist_ok=True)
            
            # Download MP3s in parallel
            downloaded = 0
            total_size = 0
            
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {}
                
                for i, mp3_url in enumerate(mp3_urls[:15]):  # Max 15 tracks
                    if not mp3_url.startswith('http'):
                        mp3_url = urljoin(show_info['url'], mp3_url)
                    
                    filename = f"track_{i+1:02d}.mp3"
                    filepath = show_dir / filename
                    
                    if not filepath.exists():
                        future = executor.submit(self.download_mp3_direct, mp3_url, filepath)
                        futures[future] = filename
                
                for future in as_completed(futures):
                    size = future.result()
                    if size > 0:
                        downloaded += 1
                        total_size += size
            
            if downloaded > 0:
                # Save metadata
                metadata = {
                    'show_id': show_id,
                    'date': show_info['date'],
                    'url': show_info['url'],
                    'tracks': downloaded,
                    'size_mb': total_size / 1024 / 1024,
                    'timestamp': time.time()
                }
                
                with open(self.metadata_dir / f"{show_id}.json", 'w') as f:
                    json.dump(metadata, f)
                
                self.completed += 1
                self.total_size_gb += total_size / 1024 / 1024 / 1024
            else:
                self.failed += 1
            
            # Progress
            self.print_progress()
            
        except Exception as e:
            logger.error(f"Error processing {show_id}: {e}")
            self.failed += 1
    
    def print_progress(self):
        """Print progress update"""
        total = self.completed + self.failed
        elapsed = time.time() - self.start_time
        rate = total / (elapsed / 60) if elapsed > 0 else 0
        
        logger.info(
            f"Progress: {self.completed} downloaded | "
            f"{self.failed} failed | "
            f"{self.total_size_gb:.1f}GB | "
            f"Rate: {rate:.1f} shows/min"
        )
    
    def run(self):
        """Run ultra fast scraping"""
        logger.info("=" * 80)
        logger.info("ULTRA FAST JGB SCRAPER")
        logger.info(f"Workers: {self.max_workers}")
        logger.info("=" * 80)
        
        # Get all shows
        shows = self.get_all_shows()
        logger.info(f"Found {len(shows)} new shows to download")
        
        if not shows:
            logger.info("No new shows!")
            return
        
        # Process in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.process_show_ultra_fast, show) 
                      for show in shows]
            
            # Wait for all to complete
            for future in as_completed(futures):
                future.result()
        
        # Final stats
        elapsed = time.time() - self.start_time
        logger.info("=" * 80)
        logger.info(f"COMPLETE! Time: {elapsed/60:.1f} minutes")
        logger.info(f"Downloaded: {self.completed} shows ({self.total_size_gb:.1f}GB)")
        logger.info(f"Failed: {self.failed} shows")
        logger.info(f"Rate: {self.completed/(elapsed/60):.1f} shows/minute")
        logger.info("=" * 80)


if __name__ == "__main__":
    import multiprocessing as mp
    max_workers = min(32, mp.cpu_count() * 4)
    
    scraper = UltraFastScraper(max_workers=max_workers)
    scraper.run()