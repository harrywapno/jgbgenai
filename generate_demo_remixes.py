#!/usr/bin/env python3
"""
Generate Demo AI Remixes from Available Audio
"""

import os
import torch
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import logging
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DemoRemixGenerator:
    """Generate demo remixes from available audio"""
    
    def __init__(self):
        self.audio_dir = Path("jgb_complete_collection/audio")
        self.output_dir = Path("ai_remixes_demo")
        self.output_dir.mkdir(exist_ok=True)
        
        # Check for GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
    
    def find_audio_files(self):
        """Find available audio files"""
        audio_files = []
        
        if self.audio_dir.exists():
            for show_dir in self.audio_dir.iterdir():
                if show_dir.is_dir():
                    mp3_files = list(show_dir.glob("*.mp3"))
                    if mp3_files:
                        # Get metadata if available
                        show_id = show_dir.name
                        metadata_file = Path(f"jgb_complete_collection/metadata/{show_id}_metadata.json")
                        
                        metadata = {}
                        if metadata_file.exists():
                            with open(metadata_file) as f:
                                metadata = json.load(f)
                        
                        audio_files.append({
                            'path': mp3_files[0],  # First track
                            'show_id': show_id,
                            'date': metadata.get('date', show_id[:10]),
                            'era': metadata.get('era', 'unknown'),
                            'songs': metadata.get('setlist', {}).get('songs', [])
                        })
        
        logger.info(f"Found {len(audio_files)} shows with audio")
        return audio_files
    
    def create_era_blend(self, audio1, audio2, sr=22050):
        """Create a blend between two era styles"""
        # Ensure same length
        min_len = min(len(audio1), len(audio2))
        audio1 = audio1[:min_len]
        audio2 = audio2[:min_len]
        
        # Apply different processing to simulate era differences
        
        # Early era - more raw, less processed
        early_style = audio1 * 0.8  # Slightly quieter
        early_style = librosa.effects.preemphasis(early_style)  # Brighter
        
        # Late era - more polished, fuller
        late_style = audio2 * 1.0
        # Add subtle reverb simulation
        late_style = np.convolve(late_style, np.ones(100)/100, mode='same')
        
        # Spectral morphing
        stft1 = librosa.stft(early_style)
        stft2 = librosa.stft(late_style)
        
        # Magnitude blending
        mag1 = np.abs(stft1)
        mag2 = np.abs(stft2)
        phase1 = np.angle(stft1)
        phase2 = np.angle(stft2)
        
        # Create morphed spectrum
        blend_factor = np.linspace(0, 1, mag1.shape[1])
        mag_blend = mag1 * (1 - blend_factor) + mag2 * blend_factor
        
        # Use phase from the more prominent signal
        phase_blend = np.where(mag1 > mag2, phase1, phase2)
        
        # Reconstruct
        stft_blend = mag_blend * np.exp(1j * phase_blend)
        blend = librosa.istft(stft_blend)
        
        return blend
    
    def create_psychedelic_effect(self, audio, sr=22050):
        """Add psychedelic effects"""
        # Phaser effect
        lfo_freq = 0.5  # Hz
        t = np.arange(len(audio)) / sr
        lfo = np.sin(2 * np.pi * lfo_freq * t)
        
        # Variable delay
        delay_samples = (20 + 10 * lfo).astype(int)
        delayed = np.zeros_like(audio)
        
        for i in range(len(audio)):
            if i >= delay_samples[i]:
                delayed[i] = audio[i - delay_samples[i]]
        
        # Mix with original
        phased = audio * 0.7 + delayed * 0.3
        
        # Add subtle pitch modulation
        pitch_shifted = librosa.effects.pitch_shift(phased, sr=sr, n_steps=0.1)
        
        # Final mix
        psychedelic = phased * 0.8 + pitch_shifted * 0.2
        
        return psychedelic
    
    def generate_demo_remixes(self):
        """Generate several demo remixes"""
        audio_files = self.find_audio_files()
        
        if len(audio_files) < 2:
            logger.error("Not enough audio files for remixing")
            return
        
        # Sort by date to get different eras
        audio_files.sort(key=lambda x: x['date'])
        
        # Pick early and late examples
        early_shows = [f for f in audio_files if f['date'].startswith('197')]
        late_shows = [f for f in audio_files if f['date'].startswith('199') or f['date'].startswith('198')]
        
        remixes_created = []
        
        # 1. Era Blend Remix
        if early_shows and late_shows:
            logger.info("Creating era blend remix...")
            early_audio, sr = librosa.load(early_shows[0]['path'], sr=22050, duration=60)
            late_audio, sr = librosa.load(late_shows[0]['path'], sr=22050, duration=60)
            
            blend = self.create_era_blend(early_audio, late_audio, sr)
            
            filename = f"ai_era_blend_{early_shows[0]['date']}_to_{late_shows[0]['date']}.wav"
            output_path = self.output_dir / filename
            sf.write(output_path, blend, sr)
            
            remixes_created.append({
                'file': str(output_path),
                'type': 'era_blend',
                'description': f"Blend from {early_shows[0]['date']} to {late_shows[0]['date']}"
            })
            logger.info(f"Created: {filename}")
        
        # 2. Psychedelic Remix
        if audio_files:
            logger.info("Creating psychedelic remix...")
            audio, sr = librosa.load(audio_files[0]['path'], sr=22050, duration=90)
            
            psychedelic = self.create_psychedelic_effect(audio, sr)
            
            filename = f"ai_psychedelic_{audio_files[0]['show_id']}.wav"
            output_path = self.output_dir / filename
            sf.write(output_path, psychedelic, sr)
            
            remixes_created.append({
                'file': str(output_path),
                'type': 'psychedelic',
                'description': f"Psychedelic version of {audio_files[0]['date']}"
            })
            logger.info(f"Created: {filename}")
        
        # 3. Multi-show Mashup
        if len(audio_files) >= 3:
            logger.info("Creating multi-show mashup...")
            segments = []
            
            for i in range(min(5, len(audio_files))):
                audio, sr = librosa.load(audio_files[i]['path'], sr=22050, duration=20)
                
                # Apply different effects to each segment
                if i % 2 == 0:
                    audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=1)
                else:
                    audio = librosa.effects.time_stretch(audio, rate=0.9)
                
                segments.append(audio)
            
            # Crossfade between segments
            mashup = segments[0]
            for i in range(1, len(segments)):
                fade_len = int(sr * 2)  # 2 second crossfade
                
                # Create fade curves
                fade_out = np.linspace(1, 0, fade_len)
                fade_in = np.linspace(0, 1, fade_len)
                
                # Apply crossfade
                mashup[-fade_len:] *= fade_out
                segments[i][:fade_len] *= fade_in
                mashup[-fade_len:] += segments[i][:fade_len]
                
                # Append rest
                mashup = np.concatenate([mashup, segments[i][fade_len:]])
            
            filename = f"ai_mashup_{len(segments)}shows_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            output_path = self.output_dir / filename
            sf.write(output_path, mashup, sr)
            
            remixes_created.append({
                'file': str(output_path),
                'type': 'mashup',
                'description': f"Mashup of {len(segments)} different shows"
            })
            logger.info(f"Created: {filename}")
        
        # 4. Tempo-shifted Journey
        if audio_files:
            logger.info("Creating tempo journey remix...")
            audio, sr = librosa.load(audio_files[-1]['path'], sr=22050, duration=120)
            
            # Split into segments
            segment_len = len(audio) // 4
            segments = [audio[i*segment_len:(i+1)*segment_len] for i in range(4)]
            
            # Apply gradual tempo changes
            rates = [1.0, 0.9, 1.1, 0.95]
            processed = []
            
            for seg, rate in zip(segments, rates):
                stretched = librosa.effects.time_stretch(seg, rate=rate)
                processed.append(stretched)
            
            journey = np.concatenate(processed)
            
            filename = f"ai_tempo_journey_{audio_files[-1]['show_id']}.wav"
            output_path = self.output_dir / filename
            sf.write(output_path, journey, sr)
            
            remixes_created.append({
                'file': str(output_path),
                'type': 'tempo_journey',
                'description': f"Tempo journey through {audio_files[-1]['date']}"
            })
            logger.info(f"Created: {filename}")
        
        # Save remix metadata
        metadata = {
            'created': datetime.now().isoformat(),
            'remixes': remixes_created,
            'source_shows': len(audio_files),
            'techniques': ['era_blending', 'spectral_morphing', 'psychedelic_effects', 'tempo_shifting']
        }
        
        with open(self.output_dir / 'remix_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"\n✓ Created {len(remixes_created)} demo remixes in {self.output_dir}/")
        return remixes_created


def main():
    generator = DemoRemixGenerator()
    remixes = generator.generate_demo_remixes()
    
    if remixes:
        print("\n🎵 Demo AI Remixes Created:")
        print("=" * 60)
        for remix in remixes:
            print(f"Type: {remix['type']}")
            print(f"File: {remix['file']}")
            print(f"Description: {remix['description']}")
            print("-" * 60)


if __name__ == "__main__":
    main()