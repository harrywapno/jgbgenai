#!/usr/bin/env python3
"""
Create AI Remixes from Available Audio
"""

import os
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_remixes():
    """Create demo remixes from available audio"""
    
    # Find audio files
    audio_dir = Path("jgb_complete_collection/audio")
    output_dir = Path("ai_remixes_output")
    output_dir.mkdir(exist_ok=True)
    
    audio_files = []
    if audio_dir.exists():
        for show_dir in audio_dir.iterdir():
            if show_dir.is_dir():
                mp3_files = list(show_dir.glob("*.mp3"))
                if mp3_files:
                    audio_files.append({
                        'path': mp3_files[0],
                        'show_id': show_dir.name,
                        'date': show_dir.name[:10]
                    })
    
    logger.info(f"Found {len(audio_files)} shows with audio")
    
    if len(audio_files) < 2:
        logger.error("Not enough audio files")
        return
    
    # Sort by date
    audio_files.sort(key=lambda x: x['date'])
    
    remixes_created = []
    
    # 1. Era Transition Remix
    logger.info("Creating era transition remix...")
    try:
        # Load early and late show segments
        early = audio_files[0]
        late = audio_files[-1]
        
        audio1, sr = librosa.load(early['path'], sr=22050, duration=45)
        audio2, sr = librosa.load(late['path'], sr=22050, duration=45)
        
        # Create smooth transition
        fade_len = int(sr * 5)  # 5 second fade
        fade_out = np.linspace(1, 0, fade_len)
        fade_in = np.linspace(0, 1, fade_len)
        
        # Apply fade
        audio1[-fade_len:] *= fade_out
        audio2[:fade_len] *= fade_in
        
        # Combine with overlap
        transition = np.concatenate([
            audio1[:-fade_len],
            audio1[-fade_len:] + audio2[:fade_len],
            audio2[fade_len:]
        ])
        
        filename = f"ai_era_transition_{early['date']}_to_{late['date']}.wav"
        output_path = output_dir / filename
        sf.write(output_path, transition, sr)
        remixes_created.append(str(output_path))
        logger.info(f"✓ Created: {filename}")
        
    except Exception as e:
        logger.error(f"Error creating era transition: {e}")
    
    # 2. Psychedelic Mix
    logger.info("Creating psychedelic remix...")
    try:
        audio, sr = librosa.load(audio_files[len(audio_files)//2]['path'], sr=22050, duration=60)
        
        # Apply effects
        # Pitch shift
        pitched = librosa.effects.pitch_shift(audio, sr=sr, n_steps=0.5)
        
        # Time stretch
        stretched = librosa.effects.time_stretch(audio, rate=0.95)
        
        # Mix
        psychedelic = audio * 0.5 + pitched * 0.3 + stretched[:len(audio)] * 0.2
        
        # Normalize
        psychedelic = psychedelic / np.max(np.abs(psychedelic)) * 0.8
        
        filename = f"ai_psychedelic_remix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        output_path = output_dir / filename
        sf.write(output_path, psychedelic, sr)
        remixes_created.append(str(output_path))
        logger.info(f"✓ Created: {filename}")
        
    except Exception as e:
        logger.error(f"Error creating psychedelic mix: {e}")
    
    # 3. Multi-show Mashup
    logger.info("Creating multi-show mashup...")
    try:
        segments = []
        for i in range(min(4, len(audio_files))):
            audio, sr = librosa.load(audio_files[i]['path'], sr=22050, duration=30, offset=30)
            segments.append(audio)
        
        # Combine with crossfades
        mashup = segments[0]
        for seg in segments[1:]:
            fade_len = int(sr * 2)
            mashup[-fade_len:] *= np.linspace(1, 0, fade_len)
            seg[:fade_len] *= np.linspace(0, 1, fade_len)
            mashup = np.concatenate([mashup[:-fade_len], mashup[-fade_len:] + seg[:fade_len], seg[fade_len:]])
        
        # Normalize
        mashup = mashup / np.max(np.abs(mashup)) * 0.8
        
        filename = f"ai_mashup_{len(segments)}shows_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        output_path = output_dir / filename
        sf.write(output_path, mashup, sr)
        remixes_created.append(str(output_path))
        logger.info(f"✓ Created: {filename}")
        
    except Exception as e:
        logger.error(f"Error creating mashup: {e}")
    
    # 4. Classic Remix
    logger.info("Creating classic JGB remix...")
    try:
        # Use the classic remix we already have
        classic_path = Path("jgb_classic_remix.wav")
        if classic_path.exists():
            # Load and enhance it
            audio, sr = librosa.load(classic_path, sr=None)
            
            # Add subtle reverb effect
            reverb = np.convolve(audio, np.ones(200)/200, mode='same')
            enhanced = audio * 0.8 + reverb * 0.2
            
            filename = f"ai_enhanced_classic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            output_path = output_dir / filename
            sf.write(output_path, enhanced, sr)
            remixes_created.append(str(output_path))
            logger.info(f"✓ Created: {filename}")
    
    except Exception as e:
        logger.error(f"Error creating classic remix: {e}")
    
    return remixes_created


def main():
    print("=" * 60)
    print("🎵 Creating AI JGB Remixes")
    print("=" * 60)
    
    remixes = create_remixes()
    
    if remixes:
        print(f"\n✅ Successfully created {len(remixes)} remixes:")
        for remix in remixes:
            print(f"  - {remix}")
        
        print("\n📤 To push these remixes:")
        print("1. Upload to cloud storage")
        print("2. Share via transfer service")
        print("3. Host on web server")
        
        print("\nRemixes are in: ai_remixes_output/")
    else:
        print("❌ No remixes created")


if __name__ == "__main__":
    main()