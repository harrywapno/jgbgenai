#!/usr/bin/env python3
"""
Start batch scraping with improved MP3 and setlist handling
"""

import os
import sys
import torch
import logging
from datetime import datetime
from sugarmegs_scraper import SugarmegsScraper

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'batch_scraping_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    logger.info("="*60)
    logger.info("JGB BATCH SCRAPING WITH EMBEDDINGS")
    logger.info("="*60)
    
    # Check GPU
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("GPU: Not available, using CPU")
    
    # Initialize scraper
    scraper = SugarmegsScraper(
        output_dir="jgb_complete_collection",
        max_workers=6,
        use_gpu=torch.cuda.is_available()
    )
    
    logger.info("\nStarting batch processing...")
    logger.info("This will:")
    logger.info("  1. Download MP3 files from archive.org")
    logger.info("  2. Parse setlists for track information")
    logger.info("  3. Extract 250+ audio features")
    logger.info("  4. Generate deep learning embeddings")
    logger.info("  5. Save everything for similarity-based remixing")
    
    # Process shows (limit for testing, remove limit for full run)
    processed, failed = scraper.batch_process_all_shows(limit=100)  # Process first 100 shows
    
    logger.info(f"\nCompleted: {len(processed)} processed, {len(failed)} failed")
    
    # Show some statistics
    if processed:
        total_tracks = sum(p.get('processed_tracks', 0) for p in processed)
        logger.info(f"Total tracks with embeddings: {total_tracks}")
        
        # Test similarity on first few tracks
        from sugarmegs_scraper import EmbeddingSimilarityEngine
        engine = EmbeddingSimilarityEngine(scraper.dirs['embeddings'])
        
        if engine.embeddings_cache:
            track = list(engine.embeddings_cache.keys())[0]
            logger.info(f"\nExample: Finding similar tracks to {track}")
            similar = engine.find_similar_tracks(track, n_similar=3)
            for t, s, d in similar:
                logger.info(f"  → {t} (similarity: {s:.3f})")

if __name__ == "__main__":
    main()