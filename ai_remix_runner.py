
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
torch.cuda.set_device(0)

from ai_remix_generator import AIRemixGenerator
import logging
import time
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AIRemixGen")

# Initialize generator with maximum GPU usage
generator = AIRemixGenerator(
    embeddings_dir="jgb_complete_collection/embeddings",
    audio_dir="jgb_complete_collection/audio",
    use_gpu=True
)

# Wait for some embeddings to be available
while len(generator.similarity_engine.embeddings_cache) < 10:
    logger.info(f"Waiting for embeddings... Current: {len(generator.similarity_engine.embeddings_cache)}")
    time.sleep(30)
    # Reload embeddings
    generator.similarity_engine = generator.similarity_engine.__class__(
        generator.embeddings_dir
    )

logger.info(f"✓ Loaded {len(generator.similarity_engine.embeddings_cache)} embeddings")

# Continuous remix generation
remix_count = 0
styles = ["psychedelic", "energetic", "mellow", "classic"]

while True:
    try:
        # Get random seed track
        available_tracks = list(generator.similarity_engine.embeddings_cache.keys())
        seed_track = random.choice(available_tracks)
        
        # Choose random style
        style = random.choice(styles)
        
        logger.info(f"\nGenerating remix #{remix_count + 1}")
        logger.info(f"Seed: {seed_track}, Style: {style}")
        
        # Generate remix
        metadata = generator.create_ai_remix(
            seed_track=seed_track,
            style=style,
            duration_minutes=random.randint(8, 15),
            temperature=random.uniform(0.6, 0.9)
        )
        
        remix_count += 1
        logger.info(f"✓ Created remix: {metadata['output_file']}")
        
        # Every 10 remixes, create an era journey
        if remix_count % 10 == 0:
            logger.info("Creating special Era Journey remix...")
            journey_metadata = generator.create_era_journey_remix()
            logger.info(f"✓ Created era journey: {journey_metadata['output_file']}")
        
        # Check for new embeddings periodically
        if remix_count % 5 == 0:
            generator.similarity_engine = generator.similarity_engine.__class__(
                generator.embeddings_dir
            )
            logger.info(f"Reloaded embeddings: {len(generator.similarity_engine.embeddings_cache)} available")
        
        # Log GPU usage
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"GPU Memory usage: {memory_used:.1f}GB")
        
        # Small pause between remixes
        time.sleep(10)
        
    except Exception as e:
        logger.error(f"Error generating remix: {e}")
        time.sleep(30)
