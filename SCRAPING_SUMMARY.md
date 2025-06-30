# JGB Scraping & Embedding System Summary

## ✅ Completed

### 1. Batch MP3 Scraping
- Successfully scraped **100 Jerry Garcia Band shows** from sugarmegs.org
- Downloaded **12.15 GB** of MP3 audio files
- Found and indexed **2,080 total JGB shows** available

### 2. Deep Learning Embeddings
- Created **100 audio embeddings** using neural networks
- Extracted **250+ audio features** per track including:
  - Spectral features (centroid, rolloff, bandwidth, contrast)
  - Temporal features (zero-crossing rate, onset detection)
  - Rhythm features (tempo, beat intervals)
  - Harmonic features (chroma, tonnetz, key estimation)
  - MFCC features (40 coefficients + deltas)
  - Energy features (RMS, dynamics)

### 3. Enhanced Text Context System
The scraper now downloads and integrates:

#### ASX Playlist Files
```xml
<ASX>
  <Entry>
    <Title>Fire on the Mountain</Title>
    <Ref href="http://archive.org/..." />
  </Entry>
</ASX>
```

#### Setlist HTML Pages
- Song titles and positions
- Venue information
- Band member details
- Show notes

#### Text Context in Embeddings
Each track embedding now includes:
```python
{
    'text_context': {
        'date': '1976-03-06',
        'venue': 'Keystone Berkeley',
        'song_title': 'deal',
        'song_position': 3,
        'asx_title': 'Deal',
        'year': 1976,
        'era': 'classic_jgb'  # early_jgb, classic_jgb, late_jgb, etc.
    }
}
```

## 🎯 Benefits for Training

### 1. Reinforced Learning
- Same songs across different shows are linked via `song_title`
- Era grouping helps identify stylistic periods
- Venue information can reveal acoustic characteristics

### 2. Enhanced Similarity Matching
The system now uses weighted similarity:
```python
total_similarity = (
    0.55 * audio_embedding_similarity +
    0.20 * tempo_similarity +
    0.10 * key_similarity +
    0.15 * text_context_similarity
)
```

### 3. Text-Audio Relationships
- Songs with same title have boosted similarity
- Tracks from same era are more likely to be selected together
- Setlist position helps maintain show flow in remixes

## 📊 Current Status

```
Shows Downloaded: 100/2080 (4.8%)
Total Audio: 12.15 GB
Embeddings: 100 tracks
With Text Context: Ready for new downloads
```

## 🚀 Next Steps

To continue scraping with full text context:

```bash
# Run enhanced scraping for next batch
python3 start_batch_scraping.py

# Or run full collection (will take many hours)
python3 run_full_scraping.py
```

The enhanced system will:
1. Download ASX files for each show
2. Parse setlist HTML pages
3. Create embeddings with full text context
4. Enable better similarity matching for remixing

## 💡 Key Innovation

By combining audio embeddings with text metadata, the system can:
- Find all versions of "Sugaree" across different years
- Create era-specific remixes (e.g., "Late 70s JGB")
- Build setlists that follow typical show patterns
- Identify rare songs and unique performances

This creates a powerful foundation for similarity-based remixing that understands both the audio characteristics AND the contextual relationships between tracks!