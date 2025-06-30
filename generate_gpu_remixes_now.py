#!/usr/bin/env python3
"""
Force GPU Remix Generation with Available Data
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import logging
import json
import random
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_gpu_remixes():
    """Generate actual GPU-accelerated remixes"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Get available audio
    audio_dir = Path("jgb_complete_collection/audio")
    metadata_dir = Path("jgb_complete_collection/metadata")
    output_dir = Path("gpu_remixes_actual")
    output_dir.mkdir(exist_ok=True)
    
    # Also check for metadata files
    if not metadata_dir.exists():
        logger.warning("Metadata directory not found, will work with audio files directly")
    
    # Collect shows by era
    shows_by_era = {
        'early_jgb': [],    # 1975-1977
        'classic_jgb': [],  # 1977-1981
        'middle_jgb': [],   # 1981-1987
        'late_jgb': [],     # 1987-1990
        'final_jgb': []     # 1991-1995
    }
    
    # Load metadata and categorize
    for meta_file in metadata_dir.glob("*.json"):
        try:
            with open(meta_file) as f:
                metadata = json.load(f)
            
            date = metadata.get('date', '')
            year = int(date[:4]) if date else 0
            
            # Try to find the audio directory
            show_id = metadata.get('show_id', '')
            show_dir = audio_dir / show_id
            
            # If exact match doesn't exist, try to find by date pattern
            if not show_dir.exists() and date:
                possible_dirs = list(audio_dir.glob(f"{date}*"))
                if possible_dirs:
                    show_dir = possible_dirs[0]
            
            if show_dir.exists():
                mp3_files = list(show_dir.glob("*.mp3"))
                if mp3_files:
                    # Categorize by year
                    if 1975 <= year <= 1977:
                        era = 'early_jgb'
                    elif 1977 <= year <= 1981:
                        era = 'classic_jgb'
                    elif 1981 <= year <= 1987:
                        era = 'middle_jgb'
                    elif 1987 <= year <= 1990:
                        era = 'late_jgb'
                    elif 1991 <= year <= 1995:
                        era = 'final_jgb'
                    else:
                        continue
                    
                    shows_by_era[era].append({
                        'audio': mp3_files[0],
                        'date': date,
                        'metadata': metadata
                    })
        except:
            continue
    
    # If no shows found via metadata, work directly with audio files
    if sum(len(shows) for shows in shows_by_era.values()) == 0:
        logger.info("No shows found via metadata, scanning audio directories...")
        for show_dir in audio_dir.iterdir():
            if show_dir.is_dir():
                mp3_files = list(show_dir.glob("*.mp3"))
                if mp3_files:
                    # Try to extract year from directory name
                    dir_name = show_dir.name
                    year = None
                    # Try different date patterns
                    for i in range(len(dir_name) - 3):
                        if dir_name[i:i+4].isdigit():
                            potential_year = int(dir_name[i:i+4])
                            if 1960 <= potential_year <= 2030:
                                year = potential_year
                                break
                    
                    if year:
                        # Categorize by year
                        if 1975 <= year <= 1977:
                            era = 'early_jgb'
                        elif 1977 <= year <= 1981:
                            era = 'classic_jgb'
                        elif 1981 <= year <= 1987:
                            era = 'middle_jgb'
                        elif 1987 <= year <= 1990:
                            era = 'late_jgb'
                        elif 1991 <= year <= 1995:
                            era = 'final_jgb'
                        else:
                            continue
                        
                        shows_by_era[era].append({
                            'audio': mp3_files[0],
                            'date': dir_name,
                            'metadata': {'show_id': dir_name}
                        })
    
    logger.info("Shows by era:")
    for era, shows in shows_by_era.items():
        logger.info(f"  {era}: {len(shows)} shows")
    
    # Generate GPU-accelerated remixes
    remixes = []
    
    # 1. GPU Era Morph Remix
    logger.info("\n🔥 Generating GPU Era Morph Remix...")
    if shows_by_era['early_jgb'] and shows_by_era['late_jgb']:
        early = random.choice(shows_by_era['early_jgb'])
        late = random.choice(shows_by_era['late_jgb'])
        
        # Load audio chunks
        early_audio, sr = librosa.load(early['audio'], sr=44100, duration=60, offset=30)
        late_audio, sr = librosa.load(late['audio'], sr=44100, duration=60, offset=30)
        
        # Convert to GPU tensors
        early_tensor = torch.from_numpy(early_audio).float().to(device)
        late_tensor = torch.from_numpy(late_audio).float().to(device)
        
        # GPU-accelerated spectral morphing
        early_stft = torch.stft(early_tensor, n_fft=2048, hop_length=512, return_complex=True)
        late_stft = torch.stft(late_tensor, n_fft=2048, hop_length=512, return_complex=True)
        
        # Magnitude and phase
        early_mag = torch.abs(early_stft)
        late_mag = torch.abs(late_stft)
        early_phase = torch.angle(early_stft)
        late_phase = torch.angle(late_stft)
        
        # GPU morphing with gradient
        morph_steps = early_mag.shape[1]
        morph_factor = torch.linspace(0, 1, morph_steps, device=device).unsqueeze(0)
        
        # Smooth interpolation
        morphed_mag = early_mag * (1 - morph_factor) + late_mag * morph_factor
        morphed_phase = early_phase * (1 - morph_factor) + late_phase * morph_factor
        
        # Reconstruct
        morphed_stft = morphed_mag * torch.exp(1j * morphed_phase)
        morphed_audio = torch.istft(morphed_stft, n_fft=2048, hop_length=512)
        
        # Apply GPU effects
        # Reverb simulation
        reverb_kernel = torch.exp(-torch.linspace(0, 5, 1000, device=device))
        reverb = torch.nn.functional.conv1d(
            morphed_audio.unsqueeze(0).unsqueeze(0),
            reverb_kernel.unsqueeze(0).unsqueeze(0),
            padding=500
        ).squeeze()
        
        # Ensure same length
        min_len = min(morphed_audio.shape[0], reverb.shape[0])
        final = morphed_audio[:min_len] * 0.7 + reverb[:min_len] * 0.3
        
        # Save
        filename = f"gpu_era_morph_{early['date']}_to_{late['date']}_{datetime.now().strftime('%H%M%S')}.wav"
        output_path = output_dir / filename
        sf.write(output_path, final.cpu().numpy(), sr)
        remixes.append(str(output_path))
        logger.info(f"✓ Created: {filename}")
    
    # 2. GPU Neural Synthesis
    logger.info("\n🔥 Generating GPU Neural Synthesis Remix...")
    all_shows = []
    for shows in shows_by_era.values():
        all_shows.extend(shows)
    
    if len(all_shows) >= 3:
        selected = random.sample(all_shows, 3)
        
        # Load segments
        segments = []
        for show in selected:
            audio, sr = librosa.load(show['audio'], sr=44100, duration=40, offset=60)
            tensor = torch.from_numpy(audio).float().to(device)
            segments.append(tensor)
        
        # Neural network processing on GPU
        class NeuralProcessor(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv1d(1, 32, kernel_size=64, stride=1, padding=32)
                self.conv2 = torch.nn.Conv1d(32, 64, kernel_size=32, stride=1, padding=16)
                self.conv3 = torch.nn.Conv1d(64, 32, kernel_size=16, stride=1, padding=8)
                self.conv4 = torch.nn.Conv1d(32, 1, kernel_size=8, stride=1, padding=4)
                
            def forward(self, x):
                x = torch.relu(self.conv1(x))
                x = torch.relu(self.conv2(x))
                x = torch.relu(self.conv3(x))
                x = torch.tanh(self.conv4(x))
                return x
        
        processor = NeuralProcessor().to(device)
        
        # Process each segment
        processed = []
        for seg in segments:
            with torch.no_grad():
                input_tensor = seg.unsqueeze(0).unsqueeze(0)
                output = processor(input_tensor).squeeze()
                processed.append(output)
        
        # GPU-accelerated crossfading
        fade_len = int(sr * 3)
        result = processed[0]
        
        for i in range(1, len(processed)):
            fade_out = torch.linspace(1, 0, fade_len, device=device)
            fade_in = torch.linspace(0, 1, fade_len, device=device)
            
            result[-fade_len:] *= fade_out
            processed[i][:fade_len] *= fade_in
            result = torch.cat([result, processed[i][fade_len:]])
        
        # Final GPU processing
        # Dynamic range compression
        result = torch.tanh(result * 0.5)
        
        filename = f"gpu_neural_synthesis_{len(segments)}shows_{datetime.now().strftime('%H%M%S')}.wav"
        output_path = output_dir / filename
        sf.write(output_path, result.cpu().numpy(), sr)
        remixes.append(str(output_path))
        logger.info(f"✓ Created: {filename}")
    
    # 3. GPU Tensor Decomposition Remix
    logger.info("\n🔥 Generating GPU Tensor Decomposition Remix...")
    if all_shows:
        show = random.choice(all_shows)
        audio, sr = librosa.load(show['audio'], sr=44100, duration=90, offset=45)
        
        # Create spectrogram tensor on GPU
        audio_tensor = torch.from_numpy(audio).float().to(device)
        stft = torch.stft(audio_tensor, n_fft=2048, hop_length=512, return_complex=True)
        mag = torch.abs(stft)
        
        # SVD decomposition on GPU
        U, S, V = torch.svd(mag)
        
        # Modify singular values
        S_modified = S.clone()
        S_modified[10:50] *= 2.0  # Enhance mid frequencies
        S_modified[100:] *= 0.5   # Reduce high frequencies
        
        # Reconstruct with modifications
        mag_modified = torch.mm(torch.mm(U, torch.diag(S_modified)), V.t())
        
        # Phase reconstruction
        phase = torch.angle(stft)
        stft_modified = mag_modified * torch.exp(1j * phase)
        
        # Back to audio
        modified_audio = torch.istft(stft_modified, n_fft=2048, hop_length=512)
        
        filename = f"gpu_tensor_decomposition_{show['date']}_{datetime.now().strftime('%H%M%S')}.wav"
        output_path = output_dir / filename
        sf.write(output_path, modified_audio.cpu().numpy(), sr)
        remixes.append(str(output_path))
        logger.info(f"✓ Created: {filename}")
    
    # Log GPU usage
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        logger.info(f"\n💾 GPU Memory used: {allocated:.1f}GB")
    
    return remixes


def main():
    logger.info("=" * 60)
    logger.info("🔥 GPU-ACCELERATED REMIX GENERATION")
    logger.info("=" * 60)
    
    remixes = generate_gpu_remixes()
    
    if remixes:
        logger.info(f"\n✅ Successfully created {len(remixes)} GPU remixes:")
        for remix in remixes:
            logger.info(f"  - {remix}")
        
        logger.info("\nThese are the ACTUAL GPU-generated remixes!")
        logger.info("Using tensor operations, neural processing, and GPU acceleration.")
    else:
        logger.info("❌ Could not generate GPU remixes")


if __name__ == "__main__":
    main()