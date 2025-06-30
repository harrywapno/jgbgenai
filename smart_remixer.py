#!/usr/bin/env python3
"""
Smart JGB Remixer with Deep Learning
====================================

Combines embedding-based similarity with real audio remixing.
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

from real_audio_remixer import JGBRemixer
from sugarmegs_scraper import EmbeddingSimilarityEngine

logger = logging.getLogger(__name__)


class SmartJGBRemixer:
    """Intelligent remixer using deep learning embeddings"""
    
    def __init__(self, embeddings_dir: str, audio_dir: str, use_gpu: bool = True):
        self.embeddings_dir = Path(embeddings_dir)
        self.audio_dir = Path(audio_dir)
        
        # Initialize components
        self.similarity_engine = EmbeddingSimilarityEngine(embeddings_dir)
        self.audio_remixer = JGBRemixer(use_gpu=use_gpu)
        
        # Map embeddings to audio files
        self._build_audio_mapping()
        
        logger.info(f"Smart Remixer initialized with {len(self.audio_mapping)} tracks")
    
    def _build_audio_mapping(self):
        """Build mapping from embedding IDs to actual audio files"""
        self.audio_mapping = {}
        
        for track_id, data in self.similarity_engine.embeddings_cache.items():
            show_id = data['show_id']
            filename = track_id.split('_')[-1]
            
            # Find corresponding audio file
            show_dir = self.audio_dir / show_id
            if show_dir.exists():
                # Look for the audio file
                for audio_file in show_dir.glob(f"*{filename}*"):
                    if audio_file.suffix in ['.mp3', '.flac', '.wav', '.ogg']:
                        self.audio_mapping[track_id] = audio_file
                        break
    
    def create_smart_remix(self, 
                          style: str = "smooth",
                          duration_minutes: int = 20,
                          output_path: str = "smart_remix.wav") -> Dict:
        """Create an intelligent remix based on style preferences"""
        
        styles = {
            "smooth": {
                "tempo_range": (80, 110),
                "diversity": 0.3,
                "transition_type": "crossfade",
                "prefer_keys": ["G", "D", "A"]
            },
            "energetic": {
                "tempo_range": (110, 140),
                "diversity": 0.5,
                "transition_type": "beat_matched",
                "prefer_keys": ["E", "A", "D"]
            },
            "psychedelic": {
                "tempo_range": (70, 100),
                "diversity": 0.7,
                "transition_type": "ambient",
                "prefer_keys": ["F", "C", "G"]
            },
            "classic": {
                "tempo_range": (90, 120),
                "diversity": 0.4,
                "transition_type": "harmonic",
                "prefer_keys": ["G", "C", "D"]
            }
        }
        
        style_config = styles.get(style, styles["smooth"])
        
        # Find suitable seed track
        seed_track = self._find_seed_track(style_config)
        if not seed_track:
            raise ValueError("No suitable seed track found")
        
        logger.info(f"Creating {style} remix starting with: {seed_track}")
        
        # Generate track sequence
        target_tracks = int(duration_minutes / 3)  # Assume ~3 min per track
        sequence = self._generate_smart_sequence(
            seed_track, 
            target_tracks,
            style_config
        )
        
        # Load audio files
        audio_files = []
        for item in sequence:
            track_id = item['track']
            if track_id in self.audio_mapping:
                audio_files.append(self.audio_mapping[track_id])
        
        if len(audio_files) < 2:
            raise ValueError("Not enough audio files found for remix")
        
        # Create the actual audio remix
        logger.info(f"Loading {len(audio_files)} audio files...")
        self.audio_remixer.load_audio_library(str(self.audio_dir))
        
        # Generate remix with style-specific parameters
        if style == "smooth":
            target_tempo = 95
            crossfade_duration = 4.0
        elif style == "energetic":
            target_tempo = 125
            crossfade_duration = 2.0
        elif style == "psychedelic":
            target_tempo = 85
            crossfade_duration = 6.0
        else:  # classic
            target_tempo = 110
            crossfade_duration = 3.0
        
        # Create remix
        remix_audio = self.audio_remixer.create_remix(
            track_names=[str(f) for f in audio_files],
            target_tempo=target_tempo,
            crossfade_duration=crossfade_duration
        )
        
        # Save remix
        self.audio_remixer.save_remix(remix_audio, output_path)
        
        # Generate remix metadata
        remix_metadata = {
            "style": style,
            "duration_minutes": len(remix_audio) / (self.audio_remixer.sr * 60),
            "tracks": len(sequence),
            "tempo": target_tempo,
            "sequence": sequence,
            "output_file": output_path
        }
        
        # Save metadata
        metadata_path = output_path.replace('.wav', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(remix_metadata, f, indent=2)
        
        logger.info(f"✓ Smart remix created: {output_path}")
        logger.info(f"  Style: {style}")
        logger.info(f"  Duration: {remix_metadata['duration_minutes']:.1f} minutes")
        logger.info(f"  Tracks: {len(sequence)}")
        
        return remix_metadata
    
    def _find_seed_track(self, style_config: Dict) -> Optional[str]:
        """Find a suitable seed track based on style preferences"""
        candidates = []
        
        tempo_min, tempo_max = style_config["tempo_range"]
        prefer_keys = style_config["prefer_keys"]
        
        for track_id, data in self.similarity_engine.embeddings_cache.items():
            # Check if we have the audio file
            if track_id not in self.audio_mapping:
                continue
            
            tempo = data['tempo']
            key = data['key'].split()[0]  # Get just the key letter
            
            # Score the track
            score = 0
            
            # Tempo score
            if tempo_min <= tempo <= tempo_max:
                # Closer to middle of range is better
                tempo_mid = (tempo_min + tempo_max) / 2
                tempo_score = 1 - abs(tempo - tempo_mid) / (tempo_max - tempo_min)
                score += tempo_score * 0.5
            
            # Key score
            if key in prefer_keys:
                key_score = 1 - (prefer_keys.index(key) / len(prefer_keys))
                score += key_score * 0.5
            
            if score > 0:
                candidates.append((track_id, score))
        
        if not candidates:
            return None
        
        # Sort by score and pick from top candidates
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Add some randomness
        top_n = min(5, len(candidates))
        weights = np.array([c[1] for c in candidates[:top_n]])
        weights = weights / weights.sum()
        
        idx = np.random.choice(top_n, p=weights)
        return candidates[idx][0]
    
    def _generate_smart_sequence(self, seed_track: str, target_length: int, 
                               style_config: Dict) -> List[Dict]:
        """Generate a track sequence based on style"""
        sequence = [{'track': seed_track, 'transition': 'start'}]
        used_tracks = {seed_track}
        current_track = seed_track
        
        tempo_min, tempo_max = style_config["tempo_range"]
        diversity = style_config["diversity"]
        transition_type = style_config["transition_type"]
        
        for i in range(target_length - 1):
            # Find similar tracks
            similar = self.similarity_engine.find_similar_tracks(
                current_track, 
                n_similar=30,
                tempo_weight=0.3 if transition_type == "beat_matched" else 0.1
            )
            
            # Filter by tempo range and availability
            candidates = []
            for track, sim, details in similar:
                if track in used_tracks:
                    continue
                if track not in self.audio_mapping:
                    continue
                
                tempo = details['tempo']
                if tempo_min <= tempo <= tempo_max:
                    candidates.append((track, sim, details))
            
            if not candidates:
                break
            
            # Select next track with style-appropriate diversity
            weights = np.array([s for _, s, _ in candidates])
            weights = weights ** (1 / diversity)
            weights = weights / weights.sum()
            
            # For the last track, prefer something that creates closure
            if i == target_length - 2:
                # Prefer tracks similar to the seed for circular structure
                seed_similar = self.similarity_engine.find_similar_tracks(
                    seed_track, n_similar=50
                )
                for track, sim, _ in seed_similar:
                    if track in [c[0] for c in candidates]:
                        # Boost weight for tracks similar to seed
                        idx = [c[0] for c in candidates].index(track)
                        weights[idx] *= 2
                weights = weights / weights.sum()
            
            idx = np.random.choice(len(candidates), p=weights)
            next_track, similarity, details = candidates[idx]
            
            sequence.append({
                'track': next_track,
                'transition': transition_type,
                'similarity': similarity,
                'tempo': details['tempo'],
                'key': details['key']
            })
            
            used_tracks.add(next_track)
            current_track = next_track
        
        return sequence
    
    def create_journey_remix(self, theme: str = "cosmic") -> Dict:
        """Create a thematic journey remix"""
        
        themes = {
            "cosmic": {
                "phases": [
                    {"name": "Launch", "tempo": 90, "energy": 0.3, "duration": 2},
                    {"name": "Orbit", "tempo": 105, "energy": 0.5, "duration": 3},
                    {"name": "Deep Space", "tempo": 85, "energy": 0.7, "duration": 4},
                    {"name": "Return", "tempo": 95, "energy": 0.4, "duration": 2}
                ]
            },
            "mountain": {
                "phases": [
                    {"name": "Valley", "tempo": 85, "energy": 0.2, "duration": 2},
                    {"name": "Ascent", "tempo": 110, "energy": 0.6, "duration": 3},
                    {"name": "Peak", "tempo": 125, "energy": 0.9, "duration": 2},
                    {"name": "Descent", "tempo": 95, "energy": 0.3, "duration": 3}
                ]
            },
            "ocean": {
                "phases": [
                    {"name": "Shore", "tempo": 80, "energy": 0.2, "duration": 3},
                    {"name": "Waves", "tempo": 95, "energy": 0.5, "duration": 4},
                    {"name": "Depths", "tempo": 75, "energy": 0.8, "duration": 3},
                    {"name": "Surface", "tempo": 90, "energy": 0.3, "duration": 2}
                ]
            }
        }
        
        journey = themes.get(theme, themes["cosmic"])
        
        logger.info(f"Creating '{theme}' journey remix...")
        
        all_tracks = []
        
        for phase in journey["phases"]:
            logger.info(f"  Phase: {phase['name']} ({phase['duration']} tracks)")
            
            # Find tracks matching phase characteristics
            phase_tracks = self._find_phase_tracks(
                tempo_target=phase["tempo"],
                energy_level=phase["energy"],
                num_tracks=phase["duration"]
            )
            
            all_tracks.extend(phase_tracks)
        
        # Create the remix
        output_path = f"journey_{theme}_remix.wav"
        
        # Load audio files
        audio_files = []
        for track_id in all_tracks:
            if track_id in self.audio_mapping:
                audio_files.append(self.audio_mapping[track_id])
        
        if len(audio_files) < 4:
            raise ValueError("Not enough tracks for journey remix")
        
        # Create remix with dynamic tempo changes
        self.audio_remixer.load_audio_library(str(self.audio_dir))
        
        remix_audio = self.audio_remixer.create_remix(
            track_names=[str(f) for f in audio_files],
            target_tempo=journey["phases"][0]["tempo"],  # Start tempo
            crossfade_duration=4.0
        )
        
        self.audio_remixer.save_remix(remix_audio, output_path)
        
        # Save journey metadata
        journey_metadata = {
            "theme": theme,
            "phases": journey["phases"],
            "total_tracks": len(all_tracks),
            "duration_minutes": len(remix_audio) / (self.audio_remixer.sr * 60),
            "output_file": output_path
        }
        
        with open(f"journey_{theme}_metadata.json", 'w') as f:
            json.dump(journey_metadata, f, indent=2)
        
        logger.info(f"✓ Journey remix created: {output_path}")
        
        return journey_metadata
    
    def _find_phase_tracks(self, tempo_target: float, energy_level: float, 
                          num_tracks: int) -> List[str]:
        """Find tracks matching phase characteristics"""
        candidates = []
        
        for track_id, data in self.similarity_engine.embeddings_cache.items():
            if track_id not in self.audio_mapping:
                continue
            
            tempo = data['tempo']
            
            # Score based on tempo proximity
            tempo_score = np.exp(-abs(tempo - tempo_target) / 20)
            
            # Estimate energy from embedding (simplified)
            embedding = data['embedding']
            estimated_energy = np.mean(np.abs(embedding[:20]))  # First 20 dims
            energy_score = 1 - abs(estimated_energy - energy_level)
            
            total_score = tempo_score * 0.7 + energy_score * 0.3
            
            candidates.append((track_id, total_score))
        
        # Sort and select
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        selected = []
        for track_id, _ in candidates[:num_tracks * 3]:  # Get more candidates
            if len(selected) < num_tracks:
                selected.append(track_id)
        
        return selected


def main():
    """Example usage of smart remixer"""
    
    # Check if we have embeddings
    embeddings_dir = Path("jgb_complete_collection/embeddings")
    audio_dir = Path("jgb_complete_collection/audio")
    
    if not embeddings_dir.exists():
        logger.error("No embeddings found. Run scraping first!")
        return
    
    # Initialize smart remixer
    remixer = SmartJGBRemixer(
        embeddings_dir=str(embeddings_dir),
        audio_dir=str(audio_dir),
        use_gpu=torch.cuda.is_available()
    )
    
    # Create different style remixes
    styles = ["smooth", "energetic", "psychedelic", "classic"]
    
    for style in styles:
        try:
            logger.info(f"\nCreating {style} remix...")
            metadata = remixer.create_smart_remix(
                style=style,
                duration_minutes=15,
                output_path=f"jgb_{style}_smart_remix.wav"
            )
            logger.info(f"Success! Created {metadata['output_file']}")
        except Exception as e:
            logger.error(f"Failed to create {style} remix: {e}")
    
    # Create a journey remix
    try:
        logger.info("\nCreating cosmic journey remix...")
        journey_metadata = remixer.create_journey_remix(theme="cosmic")
        logger.info(f"Success! Created {journey_metadata['output_file']}")
    except Exception as e:
        logger.error(f"Failed to create journey remix: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()