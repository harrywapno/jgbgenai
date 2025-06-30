#!/usr/bin/env python3
"""
Test enhanced embeddings with text context
"""

import json
from pathlib import Path
from sugarmegs_scraper import EmbeddingSimilarityEngine

def test_enhanced_embeddings():
    print("Testing Enhanced Embeddings with Text Context")
    print("=" * 60)
    
    # Load embeddings
    engine = EmbeddingSimilarityEngine("jgb_complete_collection/embeddings")
    
    if not engine.embeddings_cache:
        print("No embeddings found!")
        return
    
    # Get a sample track
    sample_track = list(engine.embeddings_cache.keys())[0]
    track_data = engine.embeddings_cache[sample_track]
    
    print(f"\nSample track: {sample_track}")
    print(f"Audio features:")
    print(f"  - Tempo: {float(track_data['tempo']):.1f} BPM")
    print(f"  - Key: {track_data['key']}")
    print(f"  - Duration: {float(track_data['duration']):.1f}s")
    
    # Show text context
    context = track_data.get('text_context', {})
    if context:
        print(f"\nText context:")
        print(f"  - Date: {context.get('date', 'N/A')}")
        print(f"  - Era: {context.get('era', 'N/A')}")
        print(f"  - Venue: {context.get('venue', 'N/A')}")
        print(f"  - Song title: {context.get('song_title', 'N/A')}")
        print(f"  - ASX title: {context.get('asx_title', 'N/A')}")
        print(f"  - Song position: {context.get('song_position', 'N/A')}")
    
    # Find similar tracks
    print(f"\nFinding similar tracks...")
    similar = engine.find_similar_tracks(sample_track, n_similar=5)
    
    print(f"\nTop 5 similar tracks:")
    for i, (track_id, similarity, details) in enumerate(similar):
        print(f"\n{i+1}. {track_id}")
        print(f"   Total similarity: {similarity:.3f}")
        print(f"   - Audio embedding: {details['embedding_sim']:.3f}")
        print(f"   - Tempo match: {details['tempo_sim']:.3f}")
        print(f"   - Key match: {details['key_sim']:.3f}")
        print(f"   - Text context: {details.get('text_sim', 0):.3f}")
        
        # Show context of similar track
        similar_context = details.get('context', {})
        if similar_context:
            print(f"   Context: {similar_context.get('era', 'N/A')} era, "
                  f"{similar_context.get('song_title', 'unknown song')}")
    
    # Test finding same song across different shows
    print("\n" + "=" * 60)
    print("Testing song-based similarity...")
    
    # Find all "Deal" tracks
    deal_tracks = []
    for track_id, data in engine.embeddings_cache.items():
        context = data.get('text_context', {})
        if context.get('song_title') == 'deal':
            deal_tracks.append((track_id, data))
    
    if deal_tracks:
        print(f"\nFound {len(deal_tracks)} 'Deal' performances")
        if len(deal_tracks) >= 2:
            # Compare first two
            track1_id, track1_data = deal_tracks[0]
            track2_id, track2_data = deal_tracks[1]
            
            print(f"\nComparing two 'Deal' performances:")
            print(f"1. {track1_id} - {track1_data.get('text_context', {}).get('date', 'N/A')}")
            print(f"2. {track2_id} - {track2_data.get('text_context', {}).get('date', 'N/A')}")
            
            # Find similarity
            similar = engine.find_similar_tracks(track1_id, n_similar=20)
            for track, sim, details in similar:
                if track == track2_id:
                    print(f"\nSimilarity score: {sim:.3f}")
                    print(f"Text similarity boost: {details.get('text_sim', 0):.3f}")
                    break
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    test_enhanced_embeddings()