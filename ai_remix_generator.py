#!/usr/bin/env python3
"""
AI-Powered JGB Remix Generator
==============================

Combines enhanced embeddings with AI music generation models
to create new versions of Jerry Garcia Band remixes.
"""

import os
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
import logging
from datetime import datetime

# Audio processing
import librosa
import soundfile as sf
from pydub import AudioSegment

# ML/AI imports
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import torchaudio

# Our modules
from sugarmegs_scraper import EmbeddingSimilarityEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AIRemixGenerator:
    """AI-powered remix generator using enhanced embeddings"""
    
    def __init__(self, embeddings_dir: str, audio_dir: str, use_gpu: bool = True):
        self.embeddings_dir = Path(embeddings_dir)
        self.audio_dir = Path(audio_dir)
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        
        # Initialize similarity engine with enhanced embeddings
        self.similarity_engine = EmbeddingSimilarityEngine(embeddings_dir)
        logger.info(f"Loaded {len(self.similarity_engine.embeddings_cache)} track embeddings")
        
        # Initialize AI models
        self._init_ai_models()
        
        # Cache for loaded audio
        self.audio_cache = {}
        
    def _init_ai_models(self):
        """Initialize AI music generation models"""
        try:
            # Initialize MusicGen model (Facebook's music generation model)
            logger.info("Loading MusicGen model...")
            self.processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
            self.musicgen_model = MusicgenForConditionalGeneration.from_pretrained(
                "facebook/musicgen-small"
            ).to(self.device)
            
            logger.info(f"AI models loaded on {self.device}")
        except Exception as e:
            logger.error(f"Error loading AI models: {e}")
            logger.info("Falling back to basic generation")
            self.processor = None
            self.musicgen_model = None
    
    def create_ai_remix(self, 
                       seed_track: str,
                       style: str = "psychedelic",
                       duration_minutes: int = 10,
                       temperature: float = 0.8) -> Dict:
        """Create an AI-powered remix using enhanced embeddings"""
        
        logger.info(f"Creating AI remix - Style: {style}, Seed: {seed_track}")
        
        # 1. Use embeddings to find similar tracks with text context
        track_sequence = self._build_smart_sequence(seed_track, style, duration_minutes)
        
        # 2. Load and analyze audio segments
        audio_segments = self._load_audio_segments(track_sequence)
        
        # 3. Generate AI transitions and variations
        ai_enhanced_segments = self._generate_ai_variations(
            audio_segments, style, temperature
        )
        
        # 4. Create the final remix
        final_remix = self._assemble_ai_remix(ai_enhanced_segments, style)
        
        # 5. Save and return metadata
        output_path = f"ai_remix_{style}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        sf.write(output_path, final_remix, 22050)
        
        metadata = {
            "output_file": output_path,
            "style": style,
            "duration": len(final_remix) / 22050 / 60,
            "seed_track": seed_track,
            "tracks_used": len(track_sequence),
            "ai_enhanced": True,
            "generation_params": {
                "temperature": temperature,
                "device": str(self.device)
            }
        }
        
        # Save detailed metadata
        with open(output_path.replace('.wav', '_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✓ AI Remix created: {output_path}")
        return metadata
    
    def _build_smart_sequence(self, seed_track: str, style: str, 
                            duration_minutes: int) -> List[Dict]:
        """Build track sequence using enhanced embeddings with text context"""
        
        # Style-specific preferences
        style_prefs = {
            "psychedelic": {
                "prefer_era": ["late_jgb", "classic_jgb"],
                "prefer_songs": ["space", "eyes of the world", "shining star"],
                "tempo_range": (70, 100),
                "diversity": 0.7
            },
            "energetic": {
                "prefer_era": ["middle_jgb", "classic_jgb"],
                "prefer_songs": ["deal", "mighty high", "cats under the stars"],
                "tempo_range": (110, 140),
                "diversity": 0.5
            },
            "mellow": {
                "prefer_era": ["early_jgb", "final_jgb"],
                "prefer_songs": ["mission in the rain", "sisters and brothers", "lucky old sun"],
                "tempo_range": (60, 90),
                "diversity": 0.3
            },
            "classic": {
                "prefer_era": ["classic_jgb"],
                "prefer_songs": ["sugaree", "run for the roses", "tangled up in blue"],
                "tempo_range": (90, 120),
                "diversity": 0.4
            }
        }
        
        prefs = style_prefs.get(style, style_prefs["classic"])
        target_tracks = int(duration_minutes / 2.5)  # ~2.5 min per segment
        
        sequence = []
        current_track = seed_track
        used_tracks = {seed_track}
        
        # Get seed track data
        seed_data = self.similarity_engine.embeddings_cache.get(seed_track, {})
        seed_context = seed_data.get('text_context', {})
        
        sequence.append({
            'track_id': seed_track,
            'context': seed_context,
            'role': 'seed'
        })
        
        # Build sequence using enhanced similarity
        for i in range(target_tracks - 1):
            # Find similar tracks with context weighting
            similar = self.similarity_engine.find_similar_tracks(
                current_track,
                n_similar=50,
                tempo_weight=0.25,
                key_weight=0.15,
                text_weight=0.20  # Increased for AI remixing
            )
            
            # Filter and score candidates
            candidates = []
            for track_id, sim, details in similar:
                if track_id in used_tracks:
                    continue
                
                context = details.get('context', {})
                score = sim
                
                # Boost score for preferred era
                if context.get('era') in prefs['prefer_era']:
                    score *= 1.2
                
                # Boost score for preferred songs
                if context.get('song_title') in prefs['prefer_songs']:
                    score *= 1.3
                
                # Check tempo range
                tempo = details['tempo']
                if prefs['tempo_range'][0] <= tempo <= prefs['tempo_range'][1]:
                    score *= 1.1
                
                candidates.append((track_id, score, details, context))
            
            if not candidates:
                break
            
            # Select with controlled randomness
            candidates.sort(key=lambda x: x[1], reverse=True)
            weights = np.array([c[1] for c in candidates[:10]])
            weights = weights ** (1 / prefs['diversity'])
            weights = weights / weights.sum()
            
            idx = np.random.choice(min(10, len(candidates)), p=weights)
            selected = candidates[idx]
            
            sequence.append({
                'track_id': selected[0],
                'context': selected[3],
                'role': 'continuation',
                'similarity': selected[1]
            })
            
            used_tracks.add(selected[0])
            current_track = selected[0]
        
        logger.info(f"Built sequence with {len(sequence)} tracks")
        return sequence
    
    def _load_audio_segments(self, track_sequence: List[Dict]) -> List[Dict]:
        """Load audio segments with context"""
        segments = []
        
        for item in track_sequence:
            track_id = item['track_id']
            
            # Find audio file
            audio_path = self._find_audio_file(track_id)
            if not audio_path:
                continue
            
            # Load audio (use cache)
            if track_id in self.audio_cache:
                y, sr = self.audio_cache[track_id]
            else:
                y, sr = librosa.load(audio_path, sr=22050, duration=180)  # 3 min max
                self.audio_cache[track_id] = (y, sr)
            
            # Extract key moments based on energy
            moments = self._extract_key_moments(y, sr)
            
            segments.append({
                'track_id': track_id,
                'audio': y,
                'sr': sr,
                'context': item['context'],
                'role': item['role'],
                'moments': moments
            })
        
        return segments
    
    def _find_audio_file(self, track_id: str) -> Optional[Path]:
        """Find audio file for track ID"""
        # Extract components from track_id
        parts = track_id.split('_')
        if len(parts) >= 3:
            date = parts[0]
            hash_id = parts[1]
            show_dir = self.audio_dir / f"{date}_{hash_id}"
            
            if show_dir.exists():
                # Look for MP3 file
                for audio_file in show_dir.glob("*.mp3"):
                    return audio_file
        
        return None
    
    def _extract_key_moments(self, y: np.ndarray, sr: int) -> List[Dict]:
        """Extract key musical moments from audio"""
        # Calculate energy
        hop_length = 512
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        
        # Find peaks
        peaks = []
        for i in range(1, len(rms) - 1):
            if rms[i] > rms[i-1] and rms[i] > rms[i+1]:
                peaks.append({
                    'time': librosa.frames_to_time(i, sr=sr, hop_length=hop_length),
                    'energy': float(rms[i])
                })
        
        # Sort by energy and return top moments
        peaks.sort(key=lambda x: x['energy'], reverse=True)
        return peaks[:10]
    
    def _generate_ai_variations(self, segments: List[Dict], style: str, 
                               temperature: float) -> List[Dict]:
        """Generate AI variations and transitions"""
        enhanced_segments = []
        
        for i, segment in enumerate(segments):
            enhanced = segment.copy()
            
            # Generate AI variations if model is available
            if self.musicgen_model is not None:
                try:
                    # Create text prompt based on context
                    context = segment['context']
                    prompt = self._create_generation_prompt(context, style)
                    
                    # Generate variation
                    logger.info(f"Generating AI variation for {segment['track_id']}")
                    variation = self._generate_with_musicgen(
                        segment['audio'], prompt, temperature
                    )
                    
                    enhanced['ai_variation'] = variation
                    enhanced['prompt'] = prompt
                    
                except Exception as e:
                    logger.error(f"Error generating AI variation: {e}")
                    enhanced['ai_variation'] = None
            
            # Generate transition to next segment
            if i < len(segments) - 1:
                transition = self._generate_transition(
                    segment, segments[i + 1], style
                )
                enhanced['transition'] = transition
            
            enhanced_segments.append(enhanced)
        
        return enhanced_segments
    
    def _create_generation_prompt(self, context: Dict, style: str) -> str:
        """Create text prompt for AI generation based on context"""
        prompts = []
        
        # Add song information
        if context.get('song_title'):
            prompts.append(f"Jerry Garcia Band playing {context['song_title']}")
        
        # Add era context
        era_descriptions = {
            "early_jgb": "early 1970s Jerry Garcia Band, raw and experimental",
            "classic_jgb": "classic mid-70s Jerry Garcia Band, polished and groovy",
            "middle_jgb": "early 80s Jerry Garcia Band, refined and soulful",
            "late_jgb": "late 80s Jerry Garcia Band, mature and exploratory",
            "final_jgb": "early 90s Jerry Garcia Band, reflective and deep"
        }
        
        if context.get('era'):
            prompts.append(era_descriptions.get(context['era'], ""))
        
        # Add style descriptors
        style_descriptors = {
            "psychedelic": "psychedelic, spacey, ambient, exploratory",
            "energetic": "upbeat, danceable, funky, driving rhythm",
            "mellow": "relaxed, contemplative, smooth, gentle",
            "classic": "traditional, bluesy, soulful, authentic"
        }
        
        prompts.append(style_descriptors.get(style, ""))
        
        # Combine prompts
        full_prompt = ", ".join([p for p in prompts if p])
        return full_prompt
    
    def _generate_with_musicgen(self, audio: np.ndarray, prompt: str, 
                               temperature: float) -> np.ndarray:
        """Generate audio variation using MusicGen"""
        try:
            # Prepare audio for model
            audio_tensor = torch.tensor(audio).float().unsqueeze(0).to(self.device)
            
            # Process prompt
            inputs = self.processor(
                text=[prompt],
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate with conditioning on original audio
            with torch.no_grad():
                audio_values = self.musicgen_model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=temperature,
                    max_new_tokens=256
                )
            
            # Convert back to numpy
            generated = audio_values[0].cpu().numpy()
            
            # Mix with original (keeping JGB essence)
            mix_ratio = 0.3  # 30% AI, 70% original
            mixed = (1 - mix_ratio) * audio[:len(generated)] + mix_ratio * generated
            
            return mixed
            
        except Exception as e:
            logger.error(f"MusicGen generation failed: {e}")
            # Return original audio if generation fails
            return audio
    
    def _generate_transition(self, segment1: Dict, segment2: Dict, 
                           style: str) -> np.ndarray:
        """Generate AI-powered transition between segments"""
        # Extract transition region
        fade_duration = 4.0  # seconds
        sr = segment1['sr']
        fade_samples = int(fade_duration * sr)
        
        # Get audio regions
        audio1_end = segment1['audio'][-fade_samples:]
        audio2_start = segment2['audio'][:fade_samples]
        
        # Create transition based on style
        if style == "psychedelic":
            # Reverse reverb effect
            transition = self._psychedelic_transition(audio1_end, audio2_start, sr)
        elif style == "energetic":
            # Beat-matched crossfade
            transition = self._energetic_transition(audio1_end, audio2_start, sr)
        else:
            # Smooth crossfade
            transition = self._smooth_transition(audio1_end, audio2_start, sr)
        
        return transition
    
    def _psychedelic_transition(self, audio1: np.ndarray, audio2: np.ndarray, 
                               sr: int) -> np.ndarray:
        """Create psychedelic transition with effects"""
        # Add reverse reverb to outgoing
        reversed_audio1 = audio1[::-1]
        
        # Create ethereal fade
        fade_curve = np.linspace(0, 1, len(audio1)) ** 2
        
        # Apply effects
        transition = np.zeros_like(audio1)
        transition += audio1 * (1 - fade_curve)
        transition += reversed_audio1 * fade_curve * 0.3
        transition += audio2 * fade_curve
        
        return transition
    
    def _energetic_transition(self, audio1: np.ndarray, audio2: np.ndarray,
                            sr: int) -> np.ndarray:
        """Create energetic beat-matched transition"""
        # Detect beats
        tempo1, beats1 = librosa.beat.beat_track(y=audio1, sr=sr)
        tempo2, beats2 = librosa.beat.beat_track(y=audio2, sr=sr)
        
        # Time-stretch to match tempo
        if abs(tempo1 - tempo2) > 5:
            rate = tempo1 / tempo2
            audio2_stretched = librosa.effects.time_stretch(audio2, rate=rate)
        else:
            audio2_stretched = audio2
        
        # Beat-aligned crossfade
        fade_curve = np.linspace(0, 1, len(audio1))
        transition = audio1 * (1 - fade_curve) + audio2_stretched[:len(audio1)] * fade_curve
        
        return transition
    
    def _smooth_transition(self, audio1: np.ndarray, audio2: np.ndarray,
                         sr: int) -> np.ndarray:
        """Create smooth crossfade transition"""
        # S-curve fade
        fade_curve = np.linspace(0, 1, len(audio1))
        fade_curve = (np.sin((fade_curve - 0.5) * np.pi) + 1) / 2
        
        transition = audio1 * (1 - fade_curve) + audio2 * fade_curve
        return transition
    
    def _assemble_ai_remix(self, segments: List[Dict], style: str) -> np.ndarray:
        """Assemble final AI-enhanced remix"""
        remix_parts = []
        
        for i, segment in enumerate(segments):
            # Add main audio or AI variation
            if segment.get('ai_variation') is not None:
                audio = segment['ai_variation']
            else:
                audio = segment['audio']
            
            # Trim to key moments for dynamic remix
            if style == "energetic" and segment.get('moments'):
                # Use high energy moments
                start_time = segment['moments'][0]['time'] if segment['moments'] else 0
                start_sample = int(start_time * segment['sr'])
                audio = audio[start_sample:start_sample + segment['sr'] * 60]  # 1 min
            
            remix_parts.append(audio)
            
            # Add transition
            if segment.get('transition') is not None:
                remix_parts.append(segment['transition'])
        
        # Concatenate all parts
        final_remix = np.concatenate(remix_parts)
        
        # Apply final mastering
        final_remix = self._apply_mastering(final_remix, style)
        
        return final_remix
    
    def _apply_mastering(self, audio: np.ndarray, style: str) -> np.ndarray:
        """Apply style-specific mastering"""
        # Normalize
        audio = audio / np.max(np.abs(audio))
        
        # Style-specific EQ
        if style == "psychedelic":
            # Boost low-mids for warmth
            audio = self._apply_eq(audio, 22050, [(200, 500, 1.2)])
        elif style == "energetic":
            # Boost presence
            audio = self._apply_eq(audio, 22050, [(2000, 5000, 1.3)])
        
        # Soft limiting
        threshold = 0.95
        audio = np.tanh(audio / threshold) * threshold
        
        return audio
    
    def _apply_eq(self, audio: np.ndarray, sr: int, 
                  bands: List[Tuple[float, float, float]]) -> np.ndarray:
        """Simple EQ implementation"""
        # This is a placeholder - in production, use proper DSP
        return audio
    
    def create_era_journey_remix(self) -> Dict:
        """Create a remix that journeys through JGB eras using AI"""
        logger.info("Creating Era Journey Remix with AI enhancements")
        
        # Define era progression
        era_sequence = [
            "early_jgb",      # Start raw
            "classic_jgb",    # Peak groove
            "late_jgb",       # Mature exploration
            "final_jgb"       # Reflective end
        ]
        
        segments = []
        
        for era in era_sequence:
            # Find best track for each era
            era_tracks = []
            for track_id, data in self.similarity_engine.embeddings_cache.items():
                context = data.get('text_context', {})
                if context.get('era') == era:
                    era_tracks.append((track_id, data))
            
            if era_tracks:
                # Pick a representative track
                track_id, data = era_tracks[np.random.randint(0, len(era_tracks))]
                segments.append({
                    'track_id': track_id,
                    'context': data.get('text_context', {}),
                    'role': f'era_{era}'
                })
        
        # Load and process segments
        audio_segments = self._load_audio_segments(segments)
        
        # Generate with era-specific AI enhancements
        enhanced_segments = []
        for i, segment in enumerate(audio_segments):
            enhanced = segment.copy()
            
            # Era-specific processing
            era = era_sequence[i] if i < len(era_sequence) else "classic_jgb"
            
            if self.musicgen_model:
                prompt = f"Jerry Garcia Band {era.replace('_', ' ')} era, authentic vintage sound"
                variation = self._generate_with_musicgen(
                    segment['audio'], prompt, temperature=0.7
                )
                enhanced['ai_variation'] = variation
            
            enhanced_segments.append(enhanced)
        
        # Assemble journey
        final_remix = self._assemble_ai_remix(enhanced_segments, "journey")
        
        # Save
        output_path = f"ai_era_journey_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        sf.write(output_path, final_remix, 22050)
        
        metadata = {
            "output_file": output_path,
            "type": "era_journey",
            "eras": era_sequence,
            "duration": len(final_remix) / 22050 / 60,
            "ai_enhanced": True
        }
        
        logger.info(f"✓ Era Journey Remix created: {output_path}")
        return metadata


def main():
    """Demo AI remix generation"""
    logger.info("Starting AI-Powered JGB Remix Generator")
    
    # Initialize generator
    generator = AIRemixGenerator(
        embeddings_dir="jgb_complete_collection/embeddings",
        audio_dir="jgb_complete_collection/audio",
        use_gpu=torch.cuda.is_available()
    )
    
    # Find a good seed track
    seed_candidates = []
    for track_id, data in generator.similarity_engine.embeddings_cache.items():
        context = data.get('text_context', {})
        if context.get('song_title') in ['sugaree', 'deal', 'mission in the rain']:
            seed_candidates.append(track_id)
    
    if seed_candidates:
        seed_track = seed_candidates[0]
        
        # Generate different style remixes
        styles = ["psychedelic", "energetic", "mellow", "classic"]
        
        for style in styles:
            try:
                logger.info(f"\nGenerating {style} AI remix...")
                metadata = generator.create_ai_remix(
                    seed_track=seed_track,
                    style=style,
                    duration_minutes=8,
                    temperature=0.8
                )
                logger.info(f"Created: {metadata['output_file']}")
            except Exception as e:
                logger.error(f"Failed to create {style} remix: {e}")
        
        # Create era journey
        try:
            logger.info("\nGenerating Era Journey remix...")
            journey_metadata = generator.create_era_journey_remix()
            logger.info(f"Created: {journey_metadata['output_file']}")
        except Exception as e:
            logger.error(f"Failed to create era journey: {e}")
    else:
        logger.error("No suitable seed tracks found")


if __name__ == "__main__":
    main()