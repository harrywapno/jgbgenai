# JGB Real Audio Remixer

Advanced GPU-accelerated real audio remixing for Jerry Garcia Band performances. This tool processes authentic JGB recordings to create seamless remixes with tempo matching, key transposition, and intelligent crossfading.

## Features

### 🎵 Real Audio Processing
- **Time-stretching**: Change tempo without affecting pitch
- **Pitch-shifting**: Transpose keys while preserving timing  
- **Tempo matching**: Sync multiple tracks to target BPM
- **Key detection**: Automatic musical key analysis
- **Crossfading**: Seamless transitions between tracks

### 🚀 GPU Acceleration
- **NVIDIA B200 Support**: Optimized for high-end GPU processing
- **PyTorch Integration**: GPU-accelerated audio transformations
- **Memory Efficient**: Handles large audio collections
- **Real-time Processing**: Low-latency for performance use

### 🎸 JGB-Specific Features
- **270 GB Library Support**: Designed for extensive JGB collections
- **Performance Metadata**: Tempo, key, and song structure analysis
- **Authentic Sound**: Preserves Jerry Garcia's original recordings
- **Setlist Generation**: Create complete performance arrangements

## Installation

### Environment Setup
```bash
# Activate virtual environment
source venv/bin/activate

# All dependencies are already installed:
# - librosa (audio analysis)
# - soundfile (audio I/O) 
# - pydub (audio manipulation)
# - torch + torchaudio (GPU acceleration)
# - numpy, scipy (numerical processing)
```

### GPU Setup
The system includes:
- ✅ NVIDIA B200 GPU (183GB memory)
- ✅ CUDA 12.8 drivers
- ✅ PyTorch with CUDA support
- ⚠️ B200 requires PyTorch nightly build for full compatibility

## Usage

### Command Line Interface
```bash
# Basic remix with 2 tracks
python real_audio_remixer.py \
  --library /path/to/jgb/audio \
  --tracks "Fire on the Mountain" "Scarlet Begonias" \
  --output my_remix.wav \
  --tempo 120 \
  --key G

# Advanced remix with custom settings
python real_audio_remixer.py \
  --library /path/to/jgb/collection \
  --tracks "Sugar Magnolia" "Sunshine Daydream" "Deal" \
  --tempo 130 \
  --key A \
  --crossfade 4.0 \
  --output setlist_remix.wav

# Show track information only
python real_audio_remixer.py \
  --library /path/to/jgb \
  --info-only
```

### Python API
```python
from real_audio_remixer import JGBRemixer

# Initialize with GPU acceleration
remixer = JGBRemixer(use_gpu=True)

# Load your JGB audio library
remixer.load_audio_library("/path/to/jgb/collection")

# Create remix
remix = remixer.create_remix(
    track_names=["Fire on the Mountain", "Scarlet Begonias"],
    target_tempo=125,
    target_key="G",
    crossfade_duration=3.0
)

# Save result
remixer.save_remix(remix, "jgb_remix.wav")
```

## Audio Library Setup

### Supported Formats
- **WAV**: Uncompressed, highest quality
- **FLAC**: Lossless compression, recommended  
- **MP3**: Lossy compression, acceptable

### Directory Structure
```
jgb_audio_library/
├── 1975/
│   ├── 1975-08-13_great_american_music_hall/
│   │   ├── fire_on_the_mountain.flac
│   │   └── scarlet_begonias.flac
├── 1976/
│   └── 1976-07-18_orpheum_theatre/
└── 1990/
    └── 1990-09-10_madison_square_garden/
```

### Metadata Requirements
The remixer automatically analyzes:
- **Tempo**: BPM detection using beat tracking
- **Key**: Musical key using chroma analysis
- **Duration**: Track length for arrangement planning
- **Audio Quality**: Sample rate and bit depth

## GPU Performance

### B200 Optimization
```python
# Check GPU status
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# GPU vs CPU benchmarks
remixer_gpu = JGBRemixer(use_gpu=True)
remixer_cpu = JGBRemixer(use_gpu=False)
```

### Performance Features
- **Parallel Processing**: Multiple tracks simultaneously
- **Memory Management**: Efficient GPU memory usage
- **Real-time Capability**: Live performance remixing
- **Batch Processing**: Handle large collections

## Examples

### Run Examples
```bash
# See all examples
python example_usage.py

# Basic remix example
python example_usage.py --basic

# GPU benchmark
python example_usage.py --benchmark
```

### Real-time Performance
```python
# Performance mode for live use
remixer = JGBRemixer(sample_rate=44100, use_gpu=True)

# Pre-load frequently used tracks
preloaded_tracks = remixer.preload_tracks([
    "Fire on the Mountain",
    "Scarlet Begonias", 
    "Deal",
    "Sugar Magnolia"
])

# Real-time tempo adjustment
current_remix = remixer.adjust_tempo_live(remix, new_tempo=135)
```

## Technical Details

### Audio Processing Pipeline
1. **Analysis**: Extract tempo, key, and structure
2. **Preprocessing**: Normalize levels and quality
3. **Transformation**: Apply tempo/pitch changes
4. **Crossfading**: Create seamless transitions
5. **Output**: Render final remix

### GPU Acceleration
- **Time-stretching**: GPU-accelerated STFT processing
- **Pitch-shifting**: Real-time frequency domain manipulation
- **Convolution**: Fast crossfading using GPU convolution
- **Memory**: Optimized for large audio buffers

### Quality Settings
```python
# High quality for archival
remixer = JGBRemixer(sample_rate=96000, use_gpu=True)

# Performance optimized  
remixer = JGBRemixer(sample_rate=44100, use_gpu=True)

# Real-time optimized
remixer = JGBRemixer(sample_rate=22050, use_gpu=True)
```

## Troubleshooting

### Common Issues

**GPU Compatibility Warning**
```
NVIDIA B200 with CUDA capability sm_100 is not compatible...
```
- The B200 GPU is detected but needs PyTorch nightly build
- Processing falls back to CPU automatically
- Performance is still excellent on CPU

**Audio Library Not Found**
```
FileNotFoundError: Audio library not found
```
- Verify the library path exists
- Check directory permissions
- Ensure audio files are present

**Memory Issues**
```
CUDA out of memory
```
- Reduce sample rate temporarily
- Process tracks individually  
- Use CPU mode: `use_gpu=False`

### Performance Tips
- Use FLAC format for best quality/size ratio
- Process in chunks for very large collections
- Pre-analyze tracks to save processing time
- Use GPU for time-critical applications

## Contributing

This tool is designed specifically for Jerry Garcia Band audio processing. Contributions should focus on:
- Improved audio quality algorithms
- Better GPU utilization
- JGB-specific features
- Performance optimizations

## License

For educational and personal use with Jerry Garcia Band recordings. Respect copyright and intellectual property rights.