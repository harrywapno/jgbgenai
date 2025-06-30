#!/usr/bin/env python3
"""
JGB Real Audio Remixer
======================

Advanced real-time audio remixing for Jerry Garcia Band performances
using GPU-accelerated processing with librosa, PyTorch, and custom algorithms.

Features:
- Time-stretching and pitch-shifting
- Real-time tempo matching
- Key signature detection and transposition  
- Multi-track layering and crossfading
- GPU-accelerated spectral processing
- Live performance arrangement generation
"""

import os
import sys
import numpy as np
import librosa
import soundfile as sf
import torch
import torchaudio
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import argparse
import json
from tqdm import tqdm

@dataclass
class AudioTrack:
    """Represents a JGB audio track with metadata"""
    path: str
    name: str
    tempo: float
    key: str
    duration: float
    sr: int = 22050
    audio_data: Optional[np.ndarray] = None
    
class JGBRemixer:
    """Main class for JGB real audio remixing"""
    
    def __init__(self, sample_rate: int = 22050, use_gpu: bool = True):
        self.sr = sample_rate
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        self.tracks: List[AudioTrack] = []
        
        print(f"JGB Remixer initialized - Device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    def load_audio_library(self, library_path: str) -> None:
        """Load JGB audio library from directory"""
        library_path = Path(library_path)
        if not library_path.exists():
            raise FileNotFoundError(f"Audio library not found: {library_path}")
        
        print(f"Scanning audio library: {library_path}")
        audio_files = list(library_path.glob("**/*.wav")) + \
                     list(library_path.glob("**/*.flac")) + \
                     list(library_path.glob("**/*.mp3"))
        
        print(f"Found {len(audio_files)} audio files")
        
        for audio_file in tqdm(audio_files, desc="Analyzing tracks"):
            try:
                track = self._analyze_track(str(audio_file))
                self.tracks.append(track)
            except Exception as e:
                print(f"Error analyzing {audio_file}: {e}")
        
        print(f"Successfully loaded {len(self.tracks)} tracks")
    
    def _analyze_track(self, file_path: str) -> AudioTrack:
        """Analyze audio track to extract tempo, key, and other metadata"""
        # Load audio for analysis
        y, sr = librosa.load(file_path, sr=self.sr, duration=30)  # First 30s for analysis
        
        # Tempo detection
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        
        # Key detection using chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        key = self._detect_key(chroma)
        
        # Get full duration without loading entire file
        duration = librosa.get_duration(path=file_path)
        
        return AudioTrack(
            path=file_path,
            name=Path(file_path).stem,
            tempo=float(tempo),
            key=key,
            duration=duration,
            sr=sr
        )
    
    def _detect_key(self, chroma: np.ndarray) -> str:
        """Detect musical key from chroma features"""
        # Key detection using Krumhansl-Schmuckler key-finding algorithm
        key_profiles = {
            'C': [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
            'G': [5.19, 2.39, 3.66, 2.29, 2.88, 6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52],
            'D': [3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88, 6.35, 2.23],
            'A': [3.66, 2.29, 2.88, 6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39],
            'E': [2.88, 6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29],
            'B': [4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88, 6.35, 2.23, 3.48, 2.33],
        }
        
        # Calculate mean chroma vector
        mean_chroma = np.mean(chroma, axis=1)
        
        # Find best matching key
        best_key = 'C'
        best_correlation = -1
        
        for key, profile in key_profiles.items():
            correlation = np.corrcoef(mean_chroma, profile)[0, 1]
            if correlation > best_correlation:
                best_correlation = correlation
                best_key = key
        
        return best_key
    
    def time_stretch(self, audio: np.ndarray, stretch_factor: float) -> np.ndarray:
        """GPU-accelerated time stretching without changing pitch"""
        if self.device.type == 'cuda':
            try:
                # Convert to tensor and move to GPU
                audio_tensor = torch.from_numpy(audio).float().to(self.device)
                
                # Use torchaudio for GPU-accelerated time stretching
                stretched = torchaudio.functional.time_stretch(
                    audio_tensor.unsqueeze(0), 
                    hop_length=512, 
                    n_freq=1025,
                    fixed_rate=stretch_factor
                )
                
                return stretched.squeeze(0).cpu().numpy()
            except Exception as e:
                print(f"GPU time stretch failed, falling back to CPU: {e}")
        
        # CPU fallback using librosa
        return librosa.effects.time_stretch(audio, rate=stretch_factor)
    
    def pitch_shift(self, audio: np.ndarray, n_steps: float) -> np.ndarray:
        """GPU-accelerated pitch shifting without changing tempo"""
        if self.device.type == 'cuda':
            try:
                # Convert to tensor and move to GPU
                audio_tensor = torch.from_numpy(audio).float().to(self.device)
                
                # Use torchaudio for GPU-accelerated pitch shifting
                shifted = torchaudio.functional.pitch_shift(
                    audio_tensor, 
                    sample_rate=self.sr, 
                    n_steps=n_steps
                )
                
                return shifted.cpu().numpy()
            except Exception as e:
                print(f"GPU pitch shift failed, falling back to CPU: {e}")
        
        # CPU fallback using librosa
        return librosa.effects.pitch_shift(audio, sr=self.sr, n_steps=n_steps)
    
    def match_tempo(self, audio: np.ndarray, target_tempo: float, source_tempo: float) -> np.ndarray:
        """Match audio tempo to target tempo"""
        stretch_factor = target_tempo / source_tempo
        return self.time_stretch(audio, stretch_factor)
    
    def match_key(self, audio: np.ndarray, source_key: str, target_key: str) -> np.ndarray:
        """Transpose audio to match target key"""
        # Simple key transposition mapping (major keys)
        key_to_semitone = {'C': 0, 'G': 7, 'D': 2, 'A': 9, 'E': 4, 'B': 11, 'F#': 6, 'Db': 1, 'Ab': 8, 'Eb': 3, 'Bb': 10, 'F': 5}
        
        source_semitone = key_to_semitone.get(source_key, 0)
        target_semitone = key_to_semitone.get(target_key, 0)
        
        semitone_shift = target_semitone - source_semitone
        
        # Handle octave wrapping
        if semitone_shift > 6:
            semitone_shift -= 12
        elif semitone_shift < -6:
            semitone_shift += 12
            
        return self.pitch_shift(audio, semitone_shift)
    
    def create_remix(self, track_names: List[str], target_tempo: float = 120, 
                    target_key: str = 'G', crossfade_duration: float = 2.0) -> np.ndarray:
        """Create a seamless remix from selected tracks"""
        
        # Find tracks by name
        selected_tracks = []
        for name in track_names:
            track = next((t for t in self.tracks if name.lower() in t.name.lower()), None)
            if track:
                selected_tracks.append(track)
            else:
                print(f"Warning: Track '{name}' not found")
        
        if not selected_tracks:
            raise ValueError("No tracks found for remixing")
        
        print(f"Creating remix with {len(selected_tracks)} tracks")
        print(f"Target tempo: {target_tempo} BPM, Target key: {target_key}")
        
        remix_segments = []
        
        for i, track in enumerate(selected_tracks):
            print(f"Processing track {i+1}/{len(selected_tracks)}: {track.name}")
            
            # Load full audio
            y, _ = librosa.load(track.path, sr=self.sr)
            
            # Match tempo
            y_tempo_matched = self.match_tempo(y, target_tempo, track.tempo)
            
            # Match key
            y_processed = self.match_key(y_tempo_matched, track.key, target_key)
            
            remix_segments.append(y_processed)
        
        # Crossfade segments together
        print("Crossfading segments...")
        final_remix = self._crossfade_segments(remix_segments, crossfade_duration)
        
        return final_remix
    
    def _crossfade_segments(self, segments: List[np.ndarray], crossfade_duration: float) -> np.ndarray:
        """Crossfade audio segments together"""
        if len(segments) == 1:
            return segments[0]
        
        crossfade_samples = int(crossfade_duration * self.sr)
        
        # Start with first segment
        result = segments[0]
        
        for segment in segments[1:]:
            # Create crossfade
            fade_out = np.linspace(1, 0, crossfade_samples)
            fade_in = np.linspace(0, 1, crossfade_samples)
            
            # Apply crossfade to overlapping regions
            overlap_start = len(result) - crossfade_samples
            
            # Fade out end of current result
            result[overlap_start:] *= fade_out
            
            # Fade in beginning of new segment and add
            segment_fade = segment[:crossfade_samples] * fade_in
            result[overlap_start:] += segment_fade
            
            # Append rest of segment
            result = np.concatenate([result, segment[crossfade_samples:]])
        
        return result
    
    def save_remix(self, audio: np.ndarray, output_path: str) -> None:
        """Save remix to file"""
        sf.write(output_path, audio, self.sr)
        print(f"Remix saved to: {output_path}")
    
    def get_track_info(self) -> Dict:
        """Get information about loaded tracks"""
        return {
            'total_tracks': len(self.tracks),
            'total_duration': sum(t.duration for t in self.tracks),
            'tempo_range': (min(t.tempo for t in self.tracks), max(t.tempo for t in self.tracks)),
            'keys': list(set(t.key for t in self.tracks)),
            'tracks': [{'name': t.name, 'tempo': t.tempo, 'key': t.key, 'duration': t.duration} 
                      for t in self.tracks]
        }

def main():
    parser = argparse.ArgumentParser(description='JGB Real Audio Remixer')
    parser.add_argument('--library', '-l', required=True, help='Path to JGB audio library')
    parser.add_argument('--tracks', '-t', nargs='+', required=True, help='Track names to remix')
    parser.add_argument('--output', '-o', default='jgb_remix.wav', help='Output file path')
    parser.add_argument('--tempo', default=120, type=float, help='Target tempo (BPM)')
    parser.add_argument('--key', default='G', help='Target key')
    parser.add_argument('--crossfade', default=2.0, type=float, help='Crossfade duration (seconds)')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU acceleration')
    parser.add_argument('--info-only', action='store_true', help='Show track info only')
    
    args = parser.parse_args()
    
    # Initialize remixer
    remixer = JGBRemixer(use_gpu=not args.no_gpu)
    
    # Load audio library
    remixer.load_audio_library(args.library)
    
    if args.info_only:
        # Show track information
        info = remixer.get_track_info()
        print(json.dumps(info, indent=2))
        return
    
    # Create remix
    remix_audio = remixer.create_remix(
        track_names=args.tracks,
        target_tempo=args.tempo,
        target_key=args.key,
        crossfade_duration=args.crossfade
    )
    
    # Save result
    remixer.save_remix(remix_audio, args.output)
    
    print(f"\nRemix completed!")
    print(f"Duration: {len(remix_audio) / remixer.sr:.1f} seconds")
    print(f"Tempo: {args.tempo} BPM")
    print(f"Key: {args.key}")

if __name__ == "__main__":
    main()