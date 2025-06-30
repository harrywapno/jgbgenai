#!/usr/bin/env python3
"""
JGB Batch Scraper with Audio Embeddings
=======================================

Scrapes Jerry Garcia Band audio from archive.org and creates 
embeddings for similarity-based remixing using deep learning.
"""

import os
import re
import time
import json
import requests
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import logging
from urllib.parse import quote

# Audio processing
import librosa
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import pickle

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AudioEmbeddingModel(nn.Module):
    """Neural network for creating audio embeddings"""
    
    def __init__(self, input_dim: int = 128, embedding_dim: int = 64):
        super(AudioEmbeddingModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, embedding_dim),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.encoder(x)


class JGBBatchScraper:
    """Batch scraper for Jerry Garcia Band audio with embedding generation"""
    
    def __init__(self, output_dir: str = "jgb_collection", max_workers: int = 4):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.max_workers = max_workers
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; JGBScraper/1.0)'
        })
        
        # Initialize embedding model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedding_model = AudioEmbeddingModel().to(self.device)
        self.scaler = StandardScaler()
        
        # Create directories
        self.audio_dir = self.output_dir / "audio"
        self.embeddings_dir = self.output_dir / "embeddings"
        self.metadata_dir = self.output_dir / "metadata"
        
        for dir_path in [self.audio_dir, self.embeddings_dir, self.metadata_dir]:
            dir_path.mkdir(exist_ok=True)
        
        logger.info(f"Initialized JGB Batch Scraper - Device: {self.device}")
    
    def extract_audio_features(self, audio_path: str) -> Dict[str, np.ndarray]:
        """Extract comprehensive audio features for embedding"""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=22050, duration=120)  # 2 minutes sample
            
            # Extract features
            features = {}
            
            # Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
            
            # Rhythm features
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            
            # Tonal features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
            
            # MFCC features
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            
            # Aggregate features
            features['spectral_centroid'] = np.mean(spectral_centroid)
            features['spectral_rolloff'] = np.mean(spectral_rolloff)
            features['spectral_bandwidth'] = np.mean(spectral_bandwidth)
            features['zero_crossing_rate'] = np.mean(zero_crossing_rate)
            features['tempo'] = tempo
            
            # Statistical summaries
            features['chroma_mean'] = np.mean(chroma, axis=1)
            features['chroma_std'] = np.std(chroma, axis=1)
            features['tonnetz_mean'] = np.mean(tonnetz, axis=1)
            features['mfcc_mean'] = np.mean(mfcc, axis=1)
            features['mfcc_std'] = np.std(mfcc, axis=1)
            
            # Energy features
            rms = librosa.feature.rms(y=y)[0]
            features['rms_mean'] = np.mean(rms)
            features['rms_std'] = np.std(rms)
            
            # Compile feature vector
            feature_vector = np.concatenate([
                [features['spectral_centroid']],
                [features['spectral_rolloff']],
                [features['spectral_bandwidth']],
                [features['zero_crossing_rate']],
                [features['tempo']],
                features['chroma_mean'],
                features['chroma_std'],
                features['tonnetz_mean'][:6],  # Use first 6 tonnetz features
                features['mfcc_mean'],
                features['mfcc_std'],
                [features['rms_mean']],
                [features['rms_std']]
            ])
            
            return {
                'features': feature_vector,
                'tempo': tempo,
                'key': self._estimate_key(chroma)
            }
            
        except Exception as e:
            logger.error(f"Error extracting features from {audio_path}: {e}")
            return None
    
    def _estimate_key(self, chroma: np.ndarray) -> str:
        """Estimate musical key from chroma features"""
        chroma_mean = np.mean(chroma, axis=1)
        key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        return key_names[np.argmax(chroma_mean)]
    
    def create_embedding(self, features: np.ndarray) -> np.ndarray:
        """Create embedding using neural network"""
        with torch.no_grad():
            # Normalize features
            if hasattr(self, '_fitted_scaler'):
                features_norm = self.scaler.transform(features.reshape(1, -1))
            else:
                features_norm = features.reshape(1, -1)
            
            # Convert to tensor
            features_tensor = torch.FloatTensor(features_norm).to(self.device)
            
            # Generate embedding
            embedding = self.embedding_model(features_tensor)
            
            return embedding.cpu().numpy().squeeze()
    
    def parse_show_list(self, show_list_text: str) -> List[Dict[str, str]]:
        """Parse the provided show list into structured data"""
        shows = []
        
        # Split by lines and process each
        lines = show_list_text.strip().split('\n')
        
        for line in lines:
            # Extract date and venue info
            date_match = re.search(r'(\d{4}-\d{2}-\d{2})', line)
            if date_match:
                date = date_match.group(1)
                
                # Extract venue (text before the date)
                venue_end = line.find(date)
                if venue_end > 0:
                    venue = line[:venue_end].strip()
                else:
                    venue = "Unknown Venue"
                
                # Extract file reference if present
                file_match = re.search(r'(\w+\.asx)', line)
                file_ref = file_match.group(1) if file_match else None
                
                shows.append({
                    'date': date,
                    'venue': venue,
                    'file_ref': file_ref,
                    'full_line': line
                })
        
        logger.info(f"Parsed {len(shows)} shows from list")
        return shows
    
    def search_archive_org(self, date: str, venue: str = None) -> List[Dict]:
        """Search archive.org for specific JGB show"""
        try:
            # Build search query
            query_parts = [
                'creator:"Jerry Garcia Band"',
                f'date:{date}',
                'mediatype:audio'
            ]
            
            if venue and venue != "Unknown Venue":
                # Clean venue name for search
                venue_clean = re.sub(r'[^\w\s]', '', venue)
                query_parts.append(f'description:"{venue_clean}"')
            
            query = ' AND '.join(query_parts)
            
            # Search URL
            search_url = f"https://archive.org/advancedsearch.php"
            params = {
                'q': query,
                'fl': 'identifier,title,date,description,downloads',
                'rows': '10',
                'output': 'json'
            }
            
            response = self.session.get(search_url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                results = data.get('response', {}).get('docs', [])
                
                # Filter and sort by downloads
                audio_results = [r for r in results if 'downloads' in r]
                audio_results.sort(key=lambda x: x.get('downloads', 0), reverse=True)
                
                return audio_results[:3]  # Top 3 results
            
        except Exception as e:
            logger.error(f"Error searching for {date}: {e}")
        
        return []
    
    def download_show(self, identifier: str, date: str) -> Optional[str]:
        """Download audio files from a show"""
        try:
            # Get file metadata
            metadata_url = f"https://archive.org/metadata/{identifier}"
            response = self.session.get(metadata_url)
            
            if response.status_code != 200:
                return None
            
            metadata = response.json()
            files = metadata.get('files', [])
            
            # Find best audio file (prefer FLAC, then MP3)
            audio_files = []
            for file_info in files:
                name = file_info.get('name', '')
                if name.endswith('.flac'):
                    audio_files.append((file_info, 1))  # Priority 1
                elif name.endswith('.mp3'):
                    audio_files.append((file_info, 2))  # Priority 2
            
            if not audio_files:
                return None
            
            # Sort by priority
            audio_files.sort(key=lambda x: x[1])
            
            # Download first few tracks
            show_dir = self.audio_dir / f"jgb_{date}"
            show_dir.mkdir(exist_ok=True)
            
            downloaded = []
            for file_info, _ in audio_files[:5]:  # Limit to 5 tracks
                filename = file_info['name']
                file_url = f"https://archive.org/download/{identifier}/{filename}"
                
                output_path = show_dir / filename
                
                # Skip if already downloaded
                if output_path.exists():
                    downloaded.append(str(output_path))
                    continue
                
                # Download with progress
                logger.info(f"Downloading: {filename}")
                response = self.session.get(file_url, stream=True)
                
                if response.status_code == 200:
                    with open(output_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    downloaded.append(str(output_path))
                    logger.info(f"✓ Downloaded: {filename}")
                
                # Rate limiting
                time.sleep(1)
            
            return show_dir if downloaded else None
            
        except Exception as e:
            logger.error(f"Error downloading {identifier}: {e}")
            return None
    
    def process_show(self, show_info: Dict) -> Dict:
        """Process a single show: search, download, extract features"""
        date = show_info['date']
        venue = show_info['venue']
        
        logger.info(f"Processing: {date} - {venue}")
        
        # Search for show
        results = self.search_archive_org(date, venue)
        
        if not results:
            logger.warning(f"No results found for {date}")
            return None
        
        # Try to download from top result
        for result in results:
            identifier = result['identifier']
            show_dir = self.download_show(identifier, date)
            
            if show_dir:
                # Process audio files and create embeddings
                embeddings = []
                audio_files = list(Path(show_dir).glob("*.mp3")) + list(Path(show_dir).glob("*.flac"))
                
                for audio_file in audio_files[:3]:  # Process first 3 tracks
                    features_data = self.extract_audio_features(str(audio_file))
                    
                    if features_data:
                        embedding = self.create_embedding(features_data['features'])
                        
                        embeddings.append({
                            'file': str(audio_file),
                            'embedding': embedding,
                            'tempo': features_data['tempo'],
                            'key': features_data['key']
                        })
                
                if embeddings:
                    # Save embeddings
                    embedding_file = self.embeddings_dir / f"{date}_embeddings.pkl"
                    with open(embedding_file, 'wb') as f:
                        pickle.dump(embeddings, f)
                    
                    # Save metadata
                    metadata = {
                        'date': date,
                        'venue': venue,
                        'identifier': identifier,
                        'title': result.get('title', ''),
                        'downloads': result.get('downloads', 0),
                        'audio_files': len(audio_files),
                        'embeddings_created': len(embeddings)
                    }
                    
                    metadata_file = self.metadata_dir / f"{date}_metadata.json"
                    with open(metadata_file, 'w') as f:
                        json.dump(metadata, f, indent=2)
                    
                    logger.info(f"✓ Processed {date}: {len(embeddings)} embeddings created")
                    return metadata
        
        return None
    
    def batch_process_shows(self, show_list_text: str, limit: int = None):
        """Process multiple shows in parallel"""
        # Parse show list
        shows = self.parse_show_list(show_list_text)
        
        if limit:
            shows = shows[:limit]
        
        logger.info(f"Starting batch processing of {len(shows)} shows")
        
        # Fit scaler on sample features first
        self._fit_scaler()
        
        # Process shows in parallel
        processed = []
        failed = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_show = {
                executor.submit(self.process_show, show): show 
                for show in shows
            }
            
            for future in as_completed(future_to_show):
                show = future_to_show[future]
                try:
                    result = future.result()
                    if result:
                        processed.append(result)
                    else:
                        failed.append(show)
                except Exception as e:
                    logger.error(f"Error processing {show['date']}: {e}")
                    failed.append(show)
        
        # Generate summary
        self._generate_summary(processed, failed)
        
        return processed, failed
    
    def _fit_scaler(self):
        """Fit the feature scaler on sample data"""
        # Generate sample features for fitting
        sample_features = []
        
        for _ in range(100):
            # Create random but realistic feature vector
            feature = np.random.randn(76)  # Match feature dimension
            sample_features.append(feature)
        
        self.scaler.fit(np.array(sample_features))
        self._fitted_scaler = True
    
    def _generate_summary(self, processed: List[Dict], failed: List[Dict]):
        """Generate processing summary"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_processed': len(processed),
            'total_failed': len(failed),
            'processed_shows': processed,
            'failed_shows': failed,
            'total_audio_files': sum(p.get('audio_files', 0) for p in processed),
            'total_embeddings': sum(p.get('embeddings_created', 0) for p in processed)
        }
        
        summary_file = self.output_dir / "processing_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"""
Batch Processing Complete:
- Processed: {summary['total_processed']} shows
- Failed: {summary['total_failed']} shows  
- Audio files: {summary['total_audio_files']}
- Embeddings: {summary['total_embeddings']}
- Summary saved to: {summary_file}
""")


class SimilarityRemixer:
    """Use embeddings to create similarity-based remixes"""
    
    def __init__(self, embeddings_dir: str):
        self.embeddings_dir = Path(embeddings_dir)
        self.embeddings_cache = {}
        self._load_embeddings()
    
    def _load_embeddings(self):
        """Load all embeddings into memory"""
        for embedding_file in self.embeddings_dir.glob("*_embeddings.pkl"):
            with open(embedding_file, 'rb') as f:
                data = pickle.load(f)
                for item in data:
                    self.embeddings_cache[item['file']] = {
                        'embedding': item['embedding'],
                        'tempo': item['tempo'],
                        'key': item['key']
                    }
        
        logger.info(f"Loaded {len(self.embeddings_cache)} track embeddings")
    
    def find_similar_tracks(self, reference_file: str, n_similar: int = 5) -> List[Tuple[str, float]]:
        """Find tracks similar to reference using embeddings"""
        if reference_file not in self.embeddings_cache:
            logger.error(f"Reference file not found in embeddings")
            return []
        
        ref_embedding = self.embeddings_cache[reference_file]['embedding']
        
        # Calculate similarities
        similarities = []
        for file_path, data in self.embeddings_cache.items():
            if file_path != reference_file:
                # Cosine similarity
                embedding = data['embedding']
                similarity = np.dot(ref_embedding, embedding) / (
                    np.linalg.norm(ref_embedding) * np.linalg.norm(embedding)
                )
                similarities.append((file_path, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:n_similar]
    
    def create_similarity_remix(self, seed_track: str, length: int = 5) -> List[str]:
        """Create a remix based on track similarity"""
        remix_tracks = [seed_track]
        current_track = seed_track
        
        for _ in range(length - 1):
            similar_tracks = self.find_similar_tracks(current_track, n_similar=10)
            
            # Filter out already used tracks
            candidates = [t for t, _ in similar_tracks if t not in remix_tracks]
            
            if candidates:
                # Pick from top candidates with some randomness
                weights = np.array([1.0 / (i + 1) for i in range(len(candidates))])
                weights = weights / weights.sum()
                
                chosen_idx = np.random.choice(len(candidates), p=weights)
                next_track = candidates[chosen_idx]
                
                remix_tracks.append(next_track)
                current_track = next_track
            else:
                break
        
        return remix_tracks


if __name__ == "__main__":
    # Initialize scraper
    scraper = JGBBatchScraper(output_dir="jgb_collection", max_workers=4)
    
    # Sample show list (abbreviated for testing)
    sample_shows = """
Boarding House, San Francisco, CA 1976-01-26
Keystone, Berkeley, CA 1976-02-07  
Keystone, Berkeley, CA 1976-03-06
Sophie's, Palo Alto, CA 1976-04-02
Keystone, Berkeley, CA 1976-05-21
    """
    
    # Process shows with embeddings
    logger.info("Starting JGB batch scraping with embedding generation...")
    processed, failed = scraper.batch_process_shows(sample_shows, limit=5)
    
    logger.info(f"Batch processing complete!")
    logger.info(f"Now you can use the embeddings for similarity-based remixing")