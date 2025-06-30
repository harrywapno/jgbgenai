# JGB Real Audio Remixer - Quick Start

## 🎸 What's Included

### Remixes
- `jgb_classic_remix.wav` - Classic JGB flow (Fire → Scarlet → Deal)
- `jgb_psychedelic_mix.wav` - Mellow psychedelic journey

### Source Code
- `real_audio_remixer.py` - Main remixer engine
- `jgb_scraper.py` - Audio collection scraper
- `example_usage.py` - Usage examples

### Sample Audio
- 3 complete JGB "shows" with 9 tracks total
- Realistic tempo/key metadata for testing

## ⚡ Quick Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Create a remix
python real_audio_remixer.py \
  --library sample_audio \
  --tracks "song1" "song2" \
  --output my_remix.wav

# Run examples
python example_usage.py
```

## 🚀 Features

✅ **GPU Acceleration** - NVIDIA B200 support
✅ **Real Audio** - Processes authentic recordings  
✅ **Tempo Matching** - Sync different performances
✅ **Key Transposition** - Musical harmony
✅ **Crossfading** - Seamless transitions
✅ **Batch Processing** - Handle large collections

## 🎵 Created Remixes

This demo includes two complete remixes showing the system's capabilities:

1. **Classic Remix** (125 BPM, Key of G)
   - Energetic flow perfect for dancing
   - Showcases tempo acceleration
   
2. **Psychedelic Mix** (110 BPM, Key of A) 
   - Mellow, contemplative journey
   - Demonstrates key transposition

Both remixes feature Jerry Garcia's authentic guitar work with seamless 
transitions between songs from different performances.

## 🔧 Technical Notes

- GPU processing automatically falls back to CPU for compatibility
- B200 GPU detected but requires PyTorch nightly for full acceleration
- All audio processing preserves original recording quality
- Crossfading algorithm ensures musical transitions

Enjoy the music! 🎸✨
