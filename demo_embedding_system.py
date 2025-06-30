#!/usr/bin/env python3
"""
Demo: JGB Embedding System
==========================

Quick demonstration of the embedding and similarity system.
"""

import numpy as np
import torch
from datetime import datetime
from pathlib import Path
import json

from sugarmegs_scraper import SugarmegsScraper, EmbeddingSimilarityEngine

def run_demo():
    """Run a quick demo of the embedding system"""
    
    print("="*60)
    print("JGB Deep Learning Embedding System Demo")
    print("="*60)
    print()
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"✓ GPU Detected: {torch.cuda.get_device_name(0)}")
        if 'B200' in torch.cuda.get_device_name(0):
            print("  → NVIDIA B200 with 183GB memory!")
    else:
        print("⚠ No GPU detected - using CPU")
    
    print()
    
    # Initialize scraper for demo
    print("1. Initializing scraper with deep learning models...")
    scraper = SugarmegsScraper(
        output_dir="jgb_demo_embeddings",
        max_workers=2,
        use_gpu=torch.cuda.is_available()
    )
    
    # Demo embedding creation
    print("\n2. Demonstrating embedding creation...")
    
    # Create sample feature vector
    sample_features = {
        'global_features': np.random.randn(300),  # Simulated features
        'segment_features': [np.random.randn(19) for _ in range(5)],
        'feature_dict': {
            'tempo': 95.5,
            'key': 'G major'
        },
        'tempo': 95.5,
        'key': 'G major',
        'duration': 180.0
    }
    
    # Create embedding
    embedding = scraper.create_deep_embedding(sample_features)
    print(f"   → Created embedding with shape: {embedding.shape}")
    print(f"   → Embedding values (first 10): {embedding[:10]}")
    
    # Demo similarity calculation
    print("\n3. Demonstrating similarity calculation...")
    
    # Create multiple embeddings
    embeddings = {}
    for i in range(5):
        features = sample_features.copy()
        features['tempo'] = 90 + i * 5  # Varying tempos
        features['feature_dict']['tempo'] = features['tempo']
        
        track_id = f"demo_track_{i}"
        embeddings[track_id] = {
            'embedding': scraper.create_deep_embedding(features),
            'tempo': features['tempo'],
            'key': 'G major' if i % 2 == 0 else 'D major'
        }
    
    # Calculate similarities
    ref_embedding = embeddings['demo_track_0']['embedding']
    
    print(f"\n   Reference track: demo_track_0 (tempo: 90, key: G major)")
    print("   Similarities to other tracks:")
    
    for track_id, data in embeddings.items():
        if track_id == 'demo_track_0':
            continue
        
        # Cosine similarity
        embedding = data['embedding']
        similarity = np.dot(ref_embedding, embedding) / (
            np.linalg.norm(ref_embedding) * np.linalg.norm(embedding)
        )
        
        print(f"   → {track_id}: {similarity:.3f} (tempo: {data['tempo']}, key: {data['key']})")
    
    # Show feature extraction capabilities
    print("\n4. Feature extraction capabilities:")
    print("   The system extracts 250+ features including:")
    print("   • Spectral features (centroid, rolloff, bandwidth, contrast)")
    print("   • Temporal features (zero-crossing rate, onset detection)")
    print("   • Rhythm features (tempo, beat intervals, groove patterns)")
    print("   • Harmonic features (chroma, tonnetz, key estimation)")
    print("   • MFCC features (40 coefficients + deltas)")
    print("   • Energy features (RMS, dynamics)")
    print("   • Pitch features (fundamental frequency tracking)")
    
    # Show embedding model architecture
    print("\n5. Deep Learning Architecture:")
    print("   • Input: 256-dimensional feature vector (after PCA)")
    print("   • Hidden layers: [512, 256, 128] with BatchNorm and Dropout")
    print("   • Output: 64-dimensional embedding")
    print("   • Attention mechanism for feature weighting")
    print(f"   • Total parameters: ~200K")
    
    # Show practical applications
    print("\n6. Practical Applications:")
    print("   • Find similar tracks across 270GB collection in milliseconds")
    print("   • Create smooth transitions based on musical similarity")
    print("   • Generate themed playlists (mellow, energetic, psychedelic)")
    print("   • Discover musical patterns across different eras")
    print("   • Enable GPU-accelerated batch processing")
    
    # Save demo results
    demo_results = {
        'timestamp': datetime.now().isoformat(),
        'gpu_available': torch.cuda.is_available(),
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'embedding_dim': 64,
        'feature_dim': 256,
        'demo_embeddings': {
            track_id: {
                'tempo': data['tempo'],
                'key': data['key'],
                'embedding_sample': data['embedding'][:5].tolist()
            }
            for track_id, data in embeddings.items()
        }
    }
    
    with open('embedding_demo_results.json', 'w') as f:
        json.dump(demo_results, f, indent=2)
    
    print("\n7. Demo complete!")
    print(f"   Results saved to: embedding_demo_results.json")
    print("\n" + "="*60)
    print("Ready to process the full JGB collection!")
    print("Run ./start_scraping.sh to begin")
    print("="*60)


if __name__ == "__main__":
    run_demo()