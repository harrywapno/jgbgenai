#!/usr/bin/env python3
"""
Run Full JGB Scraping with Deep Learning Embeddings
==================================================

Orchestrates the complete scraping and embedding generation process.
"""

import os
import sys
import torch
import logging
from pathlib import Path
from datetime import datetime

# Import our scrapers
from sugarmegs_scraper import SugarmegsScraper, EmbeddingSimilarityEngine
from real_audio_remixer import JGBRemixer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'full_scraping_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def check_gpu_status():
    """Check and report GPU status"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            logger.info(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # Check for B200
        if 'B200' in torch.cuda.get_device_name(0):
            logger.info("✓ NVIDIA B200 detected - 183GB memory available!")
        
        return True
    else:
        logger.warning("No GPU detected - will use CPU (slower)")
        return False


def run_comprehensive_scraping():
    """Run the full scraping pipeline"""
    
    logger.info("="*60)
    logger.info("JGB COMPREHENSIVE SCRAPING & EMBEDDING SYSTEM")
    logger.info("="*60)
    
    # Check GPU
    use_gpu = check_gpu_status()
    
    # Initialize scraper
    logger.info("\n1. Initializing Sugarmegs scraper...")
    scraper = SugarmegsScraper(
        output_dir="jgb_complete_collection",
        max_workers=8 if use_gpu else 4,
        use_gpu=use_gpu
    )
    
    # Run scraping
    logger.info("\n2. Starting batch scraping of all JGB shows...")
    logger.info("This will:")
    logger.info("  - Scrape all Jerry Garcia Band shows from sugarmegs.org")
    logger.info("  - Download audio files for each show")
    logger.info("  - Extract comprehensive audio features")
    logger.info("  - Generate deep learning embeddings for similarity matching")
    logger.info("  - Save everything for fast remixing\n")
    
    start_time = datetime.now()
    
    try:
        # Process all shows
        processed, failed = scraper.batch_process_all_shows(limit=None)
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info(f"\n3. Scraping completed in {duration}")
        logger.info(f"   - Successfully processed: {len(processed)} shows")
        logger.info(f"   - Failed: {len(failed)} shows")
        
        if processed:
            # Test similarity engine
            logger.info("\n4. Testing similarity-based matching...")
            test_similarity_engine(scraper.dirs['embeddings'])
            
            # Create sample remix
            logger.info("\n5. Creating sample similarity-based remix...")
            create_sample_remix(scraper.dirs['embeddings'], scraper.dirs['audio'])
        
    except KeyboardInterrupt:
        logger.info("\n\nScraping interrupted by user")
    except Exception as e:
        logger.error(f"\n\nError during scraping: {e}")
        raise


def test_similarity_engine(embeddings_dir: Path):
    """Test the similarity matching system"""
    try:
        engine = EmbeddingSimilarityEngine(embeddings_dir)
        
        if not engine.embeddings_cache:
            logger.warning("No embeddings found to test")
            return
        
        # Get a sample track
        sample_tracks = list(engine.embeddings_cache.keys())[:5]
        
        for track in sample_tracks:
            logger.info(f"\nFinding similar tracks to: {track}")
            similar = engine.find_similar_tracks(track, n_similar=3)
            
            for similar_track, similarity, details in similar:
                logger.info(f"  → {similar_track}")
                logger.info(f"    Similarity: {similarity:.3f}")
                logger.info(f"    Tempo: {details['tempo']:.1f} BPM, Key: {details['key']}")
        
    except Exception as e:
        logger.error(f"Error testing similarity engine: {e}")


def create_sample_remix(embeddings_dir: Path, audio_dir: Path):
    """Create a sample remix using similarity-based selection"""
    try:
        # Initialize similarity engine
        engine = EmbeddingSimilarityEngine(embeddings_dir)
        
        if not engine.embeddings_cache:
            logger.warning("No embeddings available for remix")
            return
        
        # Get seed track
        seed_track = list(engine.embeddings_cache.keys())[0]
        
        logger.info(f"Creating smart remix sequence starting with: {seed_track}")
        
        # Generate remix sequence
        sequence = engine.create_smart_remix_sequence(
            seed_track=seed_track,
            target_length=6,
            diversity_factor=0.4
        )
        
        logger.info("\nRemix sequence:")
        for i, item in enumerate(sequence):
            logger.info(f"  {i+1}. {item['track']} ({item['transition_type']})")
        
        # Save remix plan
        remix_plan = {
            'created': datetime.now().isoformat(),
            'seed_track': seed_track,
            'sequence': sequence,
            'total_tracks': len(sequence)
        }
        
        import json
        with open('similarity_remix_plan.json', 'w') as f:
            json.dump(remix_plan, f, indent=2)
        
        logger.info(f"\nRemix plan saved to: similarity_remix_plan.json")
        
    except Exception as e:
        logger.error(f"Error creating sample remix: {e}")


def main():
    """Main entry point"""
    
    # Print startup banner
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║          JGB Complete Collection Scraper v2.0            ║
    ║                                                          ║
    ║  Features:                                               ║
    ║  • Scrapes all JGB shows from sugarmegs.org            ║
    ║  • Downloads high-quality audio files                   ║
    ║  • Extracts 250+ audio features per track              ║
    ║  • Creates deep learning embeddings                     ║
    ║  • Enables similarity-based remixing                    ║
    ║  • GPU-accelerated with B200 support                    ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Auto-confirm for batch processing
    logger.info("\nStarting automatic batch processing of all JGB shows...")
    logger.info("This will download and process all available shows.")
    
    # Run the scraping
    run_comprehensive_scraping()
    
    logger.info("\n" + "="*60)
    logger.info("ALL OPERATIONS COMPLETE!")
    logger.info("="*60)
    logger.info("\nYour JGB collection with embeddings is ready for:")
    logger.info("  • Similarity-based track selection")
    logger.info("  • Intelligent remix creation")
    logger.info("  • Musical pattern analysis")
    logger.info("  • GPU-accelerated processing")
    logger.info("\nEnjoy the music! 🎸")


if __name__ == "__main__":
    main()