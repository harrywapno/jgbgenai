#!/usr/bin/env python3
"""
Fast Parallel JGB Scraper
========================

Uses maximum parallel workers to download all shows quickly.
"""

import os
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from sugarmegs_scraper import SugarmegsScraper
import multiprocessing as mp

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FastParallelScraper:
    def __init__(self, max_workers=24):
        self.max_workers = max_workers
        self.start_time = time.time()
        self.completed = 0
        self.failed = 0
        
        # Initialize base scraper with minimal processing
        self.base_scraper = SugarmegsScraper(
            output_dir="jgb_complete_collection",
            max_workers=4,  # Each instance gets 4 workers
            use_gpu=False   # No GPU for downloads
        )
        
        # Get existing shows
        self.existing_shows = set()
        metadata_dir = Path("jgb_complete_collection/metadata")
        if metadata_dir.exists():
            for f in metadata_dir.glob("*_metadata.json"):
                self.existing_shows.add(f.stem.replace("_metadata", ""))
        
        logger.info(f"Found {len(self.existing_shows)} existing shows")
    
    def download_show_minimal(self, show_info):
        """Download show with minimal processing"""
        try:
            # Skip if already exists
            show_id = f"{show_info['date']}_{show_info['url'].split('/')[-1][:8]}"
            if show_id in self.existing_shows:
                return None
            
            # Download without embeddings
            result = self.base_scraper.process_show(show_info, generate_embeddings=False)
            
            if result and result.get('processed_tracks', 0) > 0:
                self.completed += 1
                return result
            else:
                self.failed += 1
                return None
                
        except Exception as e:
            logger.error(f"Error downloading {show_info['date']}: {e}")
            self.failed += 1
            return None
    
    def run_parallel_download(self):
        """Download all shows in parallel"""
        logger.info("=" * 80)
        logger.info("FAST PARALLEL JGB SCRAPER")
        logger.info(f"Using {self.max_workers} parallel workers")
        logger.info("=" * 80)
        
        # Get all shows
        all_shows = self.base_scraper.scrape_jgb_index()
        
        # Filter out existing
        new_shows = []
        for show in all_shows:
            show_id = f"{show['date']}_{show['url'].split('/')[-1][:8]}"
            if show_id not in self.existing_shows:
                new_shows.append(show)
        
        total_shows = len(new_shows)
        logger.info(f"Found {total_shows} new shows to download")
        
        if total_shows == 0:
            logger.info("All shows already downloaded!")
            return
        
        # Process in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(self.download_show_minimal, show): show 
                      for show in new_shows}
            
            # Process as completed
            for future in as_completed(futures):
                show = futures[future]
                try:
                    result = future.result()
                    
                    # Progress update
                    total_processed = self.completed + self.failed
                    elapsed = time.time() - self.start_time
                    rate = total_processed / (elapsed / 60) if elapsed > 0 else 0
                    remaining = total_shows - total_processed
                    eta_minutes = remaining / rate if rate > 0 else 0
                    
                    progress_pct = (total_processed / total_shows) * 100
                    
                    logger.info(
                        f"[{total_processed}/{total_shows}] "
                        f"{progress_pct:.1f}% | "
                        f"Rate: {rate:.1f} shows/min | "
                        f"ETA: {eta_minutes:.0f} min | "
                        f"Success: {self.completed} | "
                        f"Failed: {self.failed}"
                    )
                    
                except Exception as e:
                    logger.error(f"Future error: {e}")
        
        # Final stats
        elapsed = time.time() - self.start_time
        logger.info("=" * 80)
        logger.info("DOWNLOAD COMPLETE!")
        logger.info(f"Total time: {elapsed/60:.1f} minutes")
        logger.info(f"Shows downloaded: {self.completed}")
        logger.info(f"Shows failed: {self.failed}")
        logger.info(f"Average rate: {self.completed/(elapsed/60):.1f} shows/minute")
        logger.info("=" * 80)


def main():
    # Use high number of workers
    max_workers = min(24, mp.cpu_count() * 3)
    
    scraper = FastParallelScraper(max_workers=max_workers)
    scraper.run_parallel_download()


if __name__ == "__main__":
    main()