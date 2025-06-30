#!/usr/bin/env python3
"""
TURBO JGB Scraper - Maximum Speed Edition
=========================================

Downloads all 2,080 shows as fast as possible using:
- 32 parallel workers
- Async I/O
- Minimal processing (embeddings later)
- Direct downloads without analysis
"""

import os
import asyncio
import aiohttp
import aiofiles
from pathlib import Path
import logging
from bs4 import BeautifulSoup
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
import json
import hashlib
from urllib.parse import urljoin, unquote

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TurboJGBScraper:
    """Ultra-fast JGB scraper"""
    
    def __init__(self, output_dir="jgb_turbo_collection", max_workers=32):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.audio_dir = self.output_dir / "audio"
        self.audio_dir.mkdir(exist_ok=True)
        
        self.metadata_dir = self.output_dir / "metadata"
        self.metadata_dir.mkdir(exist_ok=True)
        
        self.max_workers = max_workers
        self.session = None
        self.download_queue = asyncio.Queue()
        self.completed = 0
        self.total_size_gb = 0
        self.start_time = time.time()
        
        # Track what we've already downloaded
        self.existing_shows = set()
        self._load_existing()
        
    def _load_existing(self):
        """Load already downloaded shows"""
        for meta_file in self.metadata_dir.glob("*.json"):
            self.existing_shows.add(meta_file.stem.replace("_metadata", ""))
        logger.info(f"Found {len(self.existing_shows)} existing shows")
    
    async def get_all_jgb_shows(self):
        """Get all JGB show URLs from sugarmegs"""
        base_url = "http://tela.sugarmegs.org/alpha/j.html"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(base_url) as response:
                html = await response.text()
        
        soup = BeautifulSoup(html, 'html.parser')
        shows = []
        
        # Find Jerry Garcia Band section
        for p in soup.find_all('p'):
            if 'jerry garcia band' in p.text.lower():
                for link in p.find_all('a', href=True):
                    href = link['href']
                    if href.endswith('.asx'):
                        show_url = urljoin(base_url, href)
                        
                        # Extract date from URL
                        filename = href.split('/')[-1]
                        parts = filename.split('-')
                        if len(parts) >= 3:
                            try:
                                year = int(parts[0][-4:])
                                if 1975 <= year <= 1995:  # JGB years
                                    date = f"{parts[0][-4:]}-{parts[1]}-{parts[2][:2]}"
                                    
                                    # Generate unique ID
                                    show_id = f"{date}_{hashlib.md5(show_url.encode()).hexdigest()[:8]}"
                                    
                                    if show_id not in self.existing_shows:
                                        shows.append({
                                            'date': date,
                                            'url': show_url,
                                            'show_id': show_id
                                        })
                            except:
                                continue
        
        logger.info(f"Found {len(shows)} new shows to download")
        return shows
    
    async def download_show_fast(self, show_info):
        """Download show with minimal processing"""
        show_id = show_info['show_id']
        show_dir = self.audio_dir / show_id
        show_dir.mkdir(exist_ok=True)
        
        try:
            # Get ASX file
            async with self.session.get(show_info['url'], timeout=30) as response:
                asx_content = await response.text()
            
            # Parse ASX for MP3 URLs
            mp3_urls = []
            soup = BeautifulSoup(asx_content, 'xml')
            for ref in soup.find_all('Ref'):
                href = ref.get('href')
                if href and href.endswith('.mp3'):
                    mp3_urls.append(href)
            
            if not mp3_urls:
                # Try alternate parsing
                import re
                mp3_urls = re.findall(r'href="([^"]+\.mp3)"', asx_content, re.IGNORECASE)
            
            downloaded = 0
            total_size = 0
            
            # Download MP3s in parallel
            tasks = []
            for i, mp3_url in enumerate(mp3_urls[:20]):  # Max 20 tracks per show
                if not mp3_url.startswith('http'):
                    mp3_url = urljoin(show_info['url'], mp3_url)
                
                filename = f"track_{i+1:02d}.mp3"
                filepath = show_dir / filename
                
                if not filepath.exists():
                    task = self.download_file(mp3_url, filepath)
                    tasks.append(task)
            
            # Download all tracks for this show
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for result in results:
                    if isinstance(result, int):
                        downloaded += 1
                        total_size += result
            
            # Save minimal metadata
            metadata = {
                'show_id': show_id,
                'date': show_info['date'],
                'url': show_info['url'],
                'tracks_downloaded': downloaded,
                'total_size_mb': total_size / 1024 / 1024,
                'timestamp': time.time()
            }
            
            meta_path = self.metadata_dir / f"{show_id}_metadata.json"
            async with aiofiles.open(meta_path, 'w') as f:
                await f.write(json.dumps(metadata, indent=2))
            
            self.completed += 1
            self.total_size_gb += total_size / 1024 / 1024 / 1024
            
            # Progress update
            elapsed = time.time() - self.start_time
            rate = self.completed / (elapsed / 60) if elapsed > 0 else 0
            eta_minutes = (self.total_shows - self.completed) / rate if rate > 0 else 0
            
            logger.info(
                f"[{self.completed}/{self.total_shows}] "
                f"{show_info['date']} - {downloaded} tracks, "
                f"{total_size/1024/1024:.1f}MB | "
                f"Rate: {rate:.1f} shows/min | "
                f"ETA: {eta_minutes:.0f} min | "
                f"Total: {self.total_size_gb:.1f}GB"
            )
            
            return downloaded
            
        except Exception as e:
            logger.error(f"Error downloading {show_id}: {e}")
            return 0
    
    async def download_file(self, url, filepath):
        """Download single file"""
        try:
            async with self.session.get(url, timeout=300) as response:
                if response.status == 200:
                    content = await response.read()
                    async with aiofiles.open(filepath, 'wb') as f:
                        await f.write(content)
                    return len(content)
        except Exception as e:
            logger.debug(f"Failed to download {url}: {e}")
        return 0
    
    async def worker(self):
        """Worker to process download queue"""
        while True:
            try:
                show_info = await self.download_queue.get()
                if show_info is None:
                    break
                await self.download_show_fast(show_info)
                self.download_queue.task_done()
            except Exception as e:
                logger.error(f"Worker error: {e}")
    
    async def run_turbo(self):
        """Run turbo scraping"""
        logger.info("=" * 80)
        logger.info("TURBO JGB SCRAPER - MAXIMUM SPEED MODE")
        logger.info("=" * 80)
        
        # Get all shows
        shows = await self.get_all_jgb_shows()
        self.total_shows = len(shows)
        
        if self.total_shows == 0:
            logger.info("No new shows to download!")
            return
        
        logger.info(f"Starting download of {self.total_shows} shows with {self.max_workers} workers")
        
        # Create session with connection pooling
        connector = aiohttp.TCPConnector(
            limit=self.max_workers * 2,
            limit_per_host=self.max_workers
        )
        self.session = aiohttp.ClientSession(connector=connector)
        
        try:
            # Create workers
            workers = [asyncio.create_task(self.worker()) for _ in range(self.max_workers)]
            
            # Add all shows to queue
            for show in shows:
                await self.download_queue.put(show)
            
            # Add stop signals
            for _ in range(self.max_workers):
                await self.download_queue.put(None)
            
            # Wait for completion
            await asyncio.gather(*workers)
            
            # Final stats
            elapsed = time.time() - self.start_time
            logger.info("=" * 80)
            logger.info(f"DOWNLOAD COMPLETE!")
            logger.info(f"Shows downloaded: {self.completed}")
            logger.info(f"Total size: {self.total_size_gb:.1f}GB")
            logger.info(f"Total time: {elapsed/60:.1f} minutes")
            logger.info(f"Average rate: {self.completed/(elapsed/60):.1f} shows/minute")
            logger.info("=" * 80)
            
        finally:
            await self.session.close()


async def main():
    # Use maximum workers based on CPU cores
    max_workers = min(32, mp.cpu_count() * 4)
    
    scraper = TurboJGBScraper(
        output_dir="jgb_turbo_collection",
        max_workers=max_workers
    )
    
    await scraper.run_turbo()


if __name__ == "__main__":
    # Run with event loop
    asyncio.run(main())