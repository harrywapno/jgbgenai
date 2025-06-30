#!/usr/bin/env python3
"""
JGB Real Audio Remixer - Example Usage
=====================================

Example scripts showing how to use the JGB Real Audio Remixer
for creating authentic Jerry Garcia Band performance remixes.
"""

import numpy as np
from real_audio_remixer import JGBRemixer, AudioTrack
from pathlib import Path

def example_basic_remix():
    """Basic example: Create a simple 2-track remix"""
    print("=== Basic Remix Example ===")
    
    # Initialize remixer with GPU acceleration
    remixer = JGBRemixer(use_gpu=True)
    
    # Note: Replace with your actual JGB audio library path
    # library_path = "/path/to/your/jgb/audio/collection"
    library_path = "./sample_audio"  # For demo purposes
    
    try:
        # Load audio library
        remixer.load_audio_library(library_path)
        
        # Show available tracks
        info = remixer.get_track_info()
        print(f"Loaded {info['total_tracks']} tracks")
        print(f"Total duration: {info['total_duration']:.1f} seconds")
        print(f"Available keys: {info['keys']}")
        
        # Create remix with first few tracks
        if len(remixer.tracks) >= 2:
            track_names = [track['name'] for track in info['tracks'][:2]]
            print(f"Creating remix with tracks: {track_names}")
            
            remix = remixer.create_remix(
                track_names=track_names,
                target_tempo=120,
                target_key='G',
                crossfade_duration=3.0
            )
            
            # Save remix
            remixer.save_remix(remix, "basic_jgb_remix.wav")
            
        else:
            print("Need at least 2 tracks for remix")
            
    except FileNotFoundError:
        print("Audio library not found. Please set correct path in library_path variable.")

def example_advanced_remix():
    """Advanced example: Multi-track remix with custom processing"""
    print("\n=== Advanced Remix Example ===")
    
    remixer = JGBRemixer(sample_rate=44100, use_gpu=True)  # Higher quality
    
    # Simulate some tracks for demo (in real usage, load from library)
    if not Path("./sample_audio").exists():
        print("Creating demo setup...")
        create_demo_setup(remixer)
    
    # Advanced remix parameters
    remix_config = {
        'tracks': ['fire_on_mountain', 'scarlet_begonias', 'sugar_magnolia'],
        'target_tempo': 130,  # Slightly faster for energy
        'target_key': 'A',    # Jerry's favorite key
        'crossfade_duration': 4.0,  # Longer crossfades for smooth transitions
    }
    
    print(f"Advanced remix configuration: {remix_config}")
    
    # Custom processing would go here
    # This is where you'd add JGB-specific processing like:
    # - Jerry's guitar tone matching
    # - Specific reverb/delay settings
    # - Dynamic tempo changes
    # - Key modulations between sections
    
def example_performance_mode():
    """Example: Real-time performance mode simulation"""
    print("\n=== Performance Mode Example ===")
    
    remixer = JGBRemixer(use_gpu=True)
    
    # In performance mode, you'd have:
    # 1. Pre-loaded track segments
    # 2. Real-time tempo/key matching
    # 3. Live crossfading based on user input
    # 4. Loop detection and seamless transitions
    
    print("Performance mode features:")
    print("- Real-time GPU processing")
    print("- Live tempo adjustment")
    print("- Seamless looping")
    print("- Dynamic key changes")
    print("- Crossfade control")

def create_demo_setup(remixer):
    """Create demo audio files for testing (generates sine waves)"""
    import soundfile as sf
    import os
    
    os.makedirs("sample_audio", exist_ok=True)
    
    # Generate some demo audio files
    demo_tracks = [
        {'name': 'fire_on_mountain', 'tempo': 120, 'key': 'G', 'freq': 440},
        {'name': 'scarlet_begonias', 'tempo': 110, 'key': 'A', 'freq': 523},
        {'name': 'sugar_magnolia', 'tempo': 140, 'key': 'D', 'freq': 587},
    ]
    
    for track in demo_tracks:
        # Generate 10 seconds of sine wave
        duration = 10
        t = np.linspace(0, duration, int(remixer.sr * duration))
        audio = 0.3 * np.sin(2 * np.pi * track['freq'] * t)
        
        # Add some harmonics for more interesting sound
        audio += 0.1 * np.sin(2 * np.pi * track['freq'] * 2 * t)
        audio += 0.05 * np.sin(2 * np.pi * track['freq'] * 3 * t)
        
        filepath = f"sample_audio/{track['name']}.wav"
        sf.write(filepath, audio, remixer.sr)
        
    print("Demo audio files created in ./sample_audio/")

def example_gpu_benchmarks():
    """Benchmark GPU vs CPU processing"""
    print("\n=== GPU Benchmarks ===")
    
    # Test with GPU
    remixer_gpu = JGBRemixer(use_gpu=True)
    print(f"GPU Device: {remixer_gpu.device}")
    
    # Test with CPU
    remixer_cpu = JGBRemixer(use_gpu=False) 
    print(f"CPU Device: {remixer_cpu.device}")
    
    # Generate test audio
    test_audio = np.random.randn(22050 * 30)  # 30 seconds
    
    print("Benchmarking time stretch performance...")
    
    import time
    
    # GPU benchmark
    if remixer_gpu.device.type == 'cuda':
        start_time = time.time()
        try:
            gpu_result = remixer_gpu.time_stretch(test_audio, 1.2)
            gpu_time = time.time() - start_time
            print(f"GPU time stretch: {gpu_time:.3f} seconds")
        except Exception as e:
            print(f"GPU processing failed: {e}")
            gpu_time = float('inf')
    else:
        gpu_time = float('inf')
        print("GPU not available")
    
    # CPU benchmark
    start_time = time.time()
    cpu_result = remixer_cpu.time_stretch(test_audio, 1.2)
    cpu_time = time.time() - start_time
    print(f"CPU time stretch: {cpu_time:.3f} seconds")
    
    if gpu_time != float('inf'):
        speedup = cpu_time / gpu_time
        print(f"GPU speedup: {speedup:.2f}x")

def main():
    """Run all examples"""
    print("JGB Real Audio Remixer - Examples")
    print("=" * 40)
    
    # Run examples
    example_basic_remix()
    example_advanced_remix() 
    example_performance_mode()
    example_gpu_benchmarks()
    
    print("\n" + "=" * 40)
    print("Examples completed!")
    print("\nTo use with real JGB audio:")
    print("1. Set library_path to your JGB audio collection")
    print("2. Ensure audio files are in supported formats (WAV, FLAC, MP3)")
    print("3. Run: python real_audio_remixer.py --library /path/to/jgb --tracks 'song1' 'song2'")

if __name__ == "__main__":
    main()