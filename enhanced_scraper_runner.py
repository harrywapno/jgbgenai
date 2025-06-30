
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
torch.cuda.set_device(0)

from sugarmegs_scraper import SugarmegsScraper
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EnhancedScraper")

# Initialize scraper with full GPU usage
scraper = SugarmegsScraper(
    output_dir="jgb_complete_collection",
    max_workers=8,  # Increased for parallel downloads
    use_gpu=True
)

# Enable full text context
logger.info("Enhanced scraping with ASX/setlist parsing enabled")

# Get all shows
all_shows = scraper.scrape_jgb_index()
logger.info(f"Found {len(all_shows)} total JGB shows to scrape")

# Start from where we left off
existing_shows = len(list(scraper.dirs['metadata'].glob('*_metadata.json')))
logger.info(f"Already scraped: {existing_shows} shows")

# Process remaining shows
for i, show in enumerate(all_shows[existing_shows:], start=existing_shows):
    try:
        logger.info(f"Processing show {i+1}/{len(all_shows)}: {show['date']} - {show['venue']}")
        result = scraper.process_show(show)
        
        if result and result.get('processed_tracks', 0) > 0:
            logger.info(f"✓ Successfully processed {result['processed_tracks']} tracks")
            
            # Generate embeddings every 10 shows
            if (i + 1) % 10 == 0:
                logger.info("Generating batch embeddings...")
                scraper.generate_embeddings_batch(force_regenerate=False)
        else:
            logger.warning(f"⚠ No tracks processed for {show['date']}")
            
    except Exception as e:
        logger.error(f"Error processing show {show['date']}: {e}")
        continue
    
    # Log progress
    if (i + 1) % 50 == 0:
        progress = ((i + 1) / len(all_shows)) * 100
        logger.info(f"Progress: {progress:.1f}% complete ({i+1}/{len(all_shows)} shows)")
        
        # Check GPU memory
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1024**3
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"GPU Memory: {memory_used:.1f}GB / {memory_total:.1f}GB")

logger.info("✓ Enhanced scraping complete!")
