#!/usr/bin/env python3
"""
Sugarmegs JGB Complete Scraper with Deep Learning Embeddings
============================================================

Scrapes all Jerry Garcia Band audio from sugarmegs.org and creates
comprehensive embeddings for similarity-based remixing.
"""

import os
import re
import time
import json
import requests
from bs4 import BeautifulSoup
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import logging
from urllib.parse import urljoin, urlparse, quote
import hashlib

# Audio processing and ML
import librosa
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle
import h5py

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sugarmegs_scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DeepAudioEmbedder(nn.Module):
    """Deep neural network for comprehensive audio embeddings"""
    
    def __init__(self, input_dim: int = 256, hidden_dims: List[int] = [512, 256, 128], 
                 embedding_dim: int = 64):
        super(DeepAudioEmbedder, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build encoder layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        # Final embedding layer
        layers.extend([
            nn.Linear(prev_dim, embedding_dim),
            nn.Tanh()
        ])
        
        self.encoder = nn.Sequential(*layers)
        
        # Attention mechanism for feature importance
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, input_dim),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        # Apply attention
        attention_weights = self.attention(x)
        x_attended = x * attention_weights
        
        # Generate embedding
        embedding = self.encoder(x_attended)
        return embedding, attention_weights


class SugarmegsScraper:
    """Complete scraper for sugarmegs.org Jerry Garcia Band collection"""
    
    def __init__(self, output_dir: str = "jgb_sugarmegs_collection", 
                 max_workers: int = 6, use_gpu: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.max_workers = max_workers
        self.base_url = "https://tela.sugarmegs.org"
        
        # Setup session
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Setup GPU/CPU
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize deep learning models
        self.embedding_model = DeepAudioEmbedder(input_dim=256).to(self.device)
        self.embedding_model.eval()
        
        # Feature processing
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # Keep 95% of variance instead of fixed components
        
        # Create directory structure
        self.dirs = {
            'audio': self.output_dir / "audio",
            'embeddings': self.output_dir / "embeddings",
            'metadata': self.output_dir / "metadata",
            'features': self.output_dir / "features",
            'models': self.output_dir / "models"
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(exist_ok=True)
        
        # Track processed shows
        self.processed_shows = self._load_processed_shows()
        
        logger.info(f"Initialized Sugarmegs scraper - Output: {self.output_dir}")
    
    def _load_processed_shows(self) -> Set[str]:
        """Load set of already processed shows"""
        processed_file = self.output_dir / "processed_shows.json"
        if processed_file.exists():
            with open(processed_file, 'r') as f:
                return set(json.load(f))
        return set()
    
    def _save_processed_shows(self):
        """Save set of processed shows"""
        processed_file = self.output_dir / "processed_shows.json"
        with open(processed_file, 'w') as f:
            json.dump(list(self.processed_shows), f)
    
    def extract_comprehensive_features(self, audio_path: str, 
                                     segment_duration: int = 30) -> Optional[Dict]:
        """Extract comprehensive audio features for deep learning"""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=22050, duration=180)  # 3 min sample
            
            features = {}
            
            # 1. Spectral Features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
            
            # 2. Temporal Features
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
            
            # 3. Rhythm Features
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            beat_times = librosa.frames_to_time(beats, sr=sr)
            beat_intervals = np.diff(beat_times) if len(beat_times) > 1 else [0]
            
            # 4. Harmonic/Percussive Separation
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            harmonic_ratio = np.mean(np.abs(y_harmonic)) / (np.mean(np.abs(y)) + 1e-6)
            
            # 5. Tonal Features
            chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
            tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
            
            # 6. MFCC Features (extended)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
            
            # 7. Mel Spectrogram Features
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # 8. Energy Features
            rms = librosa.feature.rms(y=y)[0]
            
            # 9. Pitch Features
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_mean = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_mean.append(pitch)
            
            # 10. Onset Detection
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            onset_times = librosa.onset.onset_detect(y=y, sr=sr, units='time')
            
            # Aggregate all features with statistics
            feature_dict = {
                # Spectral statistics
                'spectral_centroid_mean': np.mean(spectral_centroids),
                'spectral_centroid_std': np.std(spectral_centroids),
                'spectral_centroid_skew': self._safe_skew(spectral_centroids),
                'spectral_rolloff_mean': np.mean(spectral_rolloff),
                'spectral_rolloff_std': np.std(spectral_rolloff),
                'spectral_bandwidth_mean': np.mean(spectral_bandwidth),
                'spectral_bandwidth_std': np.std(spectral_bandwidth),
                'spectral_contrast_mean': np.mean(spectral_contrast, axis=1),
                'spectral_contrast_std': np.std(spectral_contrast, axis=1),
                'spectral_flatness_mean': np.mean(spectral_flatness),
                
                # Temporal
                'zero_crossing_rate_mean': np.mean(zero_crossing_rate),
                'zero_crossing_rate_std': np.std(zero_crossing_rate),
                
                # Rhythm
                'tempo': tempo,
                'beat_interval_mean': np.mean(beat_intervals) if len(beat_intervals) > 0 else 0,
                'beat_interval_std': np.std(beat_intervals) if len(beat_intervals) > 0 else 0,
                'onset_rate': len(onset_times) / (len(y) / sr) if len(y) > 0 else 0,
                
                # Harmonic/Percussive
                'harmonic_ratio': harmonic_ratio,
                
                # Tonal (full vectors)
                'chroma_mean': np.mean(chroma_stft, axis=1),
                'chroma_std': np.std(chroma_stft, axis=1),
                'chroma_cqt_mean': np.mean(chroma_cqt, axis=1),
                'tonnetz_mean': np.mean(tonnetz, axis=1),
                'tonnetz_std': np.std(tonnetz, axis=1),
                
                # MFCC (extended)
                'mfcc_mean': np.mean(mfcc, axis=1),
                'mfcc_std': np.std(mfcc, axis=1),
                'mfcc_delta_mean': np.mean(mfcc_delta, axis=1),
                'mfcc_delta2_mean': np.mean(mfcc_delta2, axis=1),
                
                # Mel spectrogram
                'mel_mean': np.mean(mel_spec_db, axis=1),
                'mel_std': np.std(mel_spec_db, axis=1),
                
                # Energy
                'rms_mean': np.mean(rms),
                'rms_std': np.std(rms),
                'rms_skew': self._safe_skew(rms),
                
                # Pitch
                'pitch_mean': np.mean(pitch_mean) if pitch_mean else 0,
                'pitch_std': np.std(pitch_mean) if len(pitch_mean) > 1 else 0,
            }
            
            # Create comprehensive feature vector
            feature_vector = self._compile_feature_vector(feature_dict)
            
            # Extract segment-based features for temporal modeling
            segment_features = self._extract_segment_features(y, sr, segment_duration)
            
            return {
                'global_features': feature_vector,
                'segment_features': segment_features,
                'feature_dict': feature_dict,
                'tempo': tempo,
                'key': self._estimate_key(chroma_stft),
                'duration': len(y) / sr
            }
            
        except Exception as e:
            logger.error(f"Error extracting features from {audio_path}: {e}")
            return None
    
    def _safe_skew(self, data: np.ndarray) -> float:
        """Calculate skewness safely"""
        if len(data) < 3:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    def _compile_feature_vector(self, feature_dict: Dict) -> np.ndarray:
        """Compile all features into a single vector"""
        features = []
        
        # Add scalar features
        scalar_keys = ['spectral_centroid_mean', 'spectral_centroid_std', 'spectral_centroid_skew',
                      'spectral_rolloff_mean', 'spectral_rolloff_std', 'spectral_bandwidth_mean',
                      'spectral_bandwidth_std', 'spectral_flatness_mean', 'zero_crossing_rate_mean',
                      'zero_crossing_rate_std', 'tempo', 'beat_interval_mean', 'beat_interval_std',
                      'onset_rate', 'harmonic_ratio', 'rms_mean', 'rms_std', 'rms_skew',
                      'pitch_mean', 'pitch_std']
        
        for key in scalar_keys:
            if key in feature_dict:
                value = feature_dict[key]
                if isinstance(value, (int, float)):
                    features.append(value)
                else:
                    features.append(float(value) if value else 0.0)
        
        # Add vector features (ensure they're flattened properly)
        vector_keys = ['spectral_contrast_mean', 'spectral_contrast_std', 'chroma_mean',
                      'chroma_std', 'chroma_cqt_mean', 'tonnetz_mean', 'tonnetz_std',
                      'mfcc_mean', 'mfcc_std', 'mfcc_delta_mean', 'mfcc_delta2_mean']
        
        for key in vector_keys:
            if key in feature_dict:
                value = feature_dict[key]
                if isinstance(value, np.ndarray):
                    features.extend(value.flatten().tolist())
                elif isinstance(value, list):
                    features.extend(value)
                else:
                    # Handle unexpected types
                    logger.warning(f"Unexpected type for {key}: {type(value)}")
        
        return np.array(features, dtype=np.float32)
    
    def _extract_segment_features(self, y: np.ndarray, sr: int, 
                                segment_duration: int) -> List[np.ndarray]:
        """Extract features from audio segments for temporal modeling"""
        segment_samples = segment_duration * sr
        segments = []
        
        for i in range(0, len(y) - segment_samples, segment_samples // 2):
            segment = y[i:i + segment_samples]
            
            # Extract basic features for each segment
            segment_features = []
            
            # Spectral centroid
            sc = librosa.feature.spectral_centroid(y=segment, sr=sr)[0]
            segment_features.extend([np.mean(sc), np.std(sc)])
            
            # RMS energy
            rms = librosa.feature.rms(y=segment)[0]
            segment_features.extend([np.mean(rms), np.std(rms)])
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(segment)[0]
            segment_features.extend([np.mean(zcr), np.std(zcr)])
            
            # MFCC
            mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13)
            segment_features.extend(np.mean(mfcc, axis=1))
            
            segments.append(np.array(segment_features))
        
        return segments
    
    def _estimate_key(self, chroma: np.ndarray) -> str:
        """Estimate musical key using chroma features"""
        chroma_mean = np.mean(chroma, axis=1)
        
        # Major and minor key profiles
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        
        # Normalize profiles
        major_profile = major_profile / np.sum(major_profile)
        minor_profile = minor_profile / np.sum(minor_profile)
        
        # Calculate correlations for all keys
        key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        correlations = []
        
        for i in range(12):
            # Rotate chroma to key
            rotated_chroma = np.roll(chroma_mean, -i)
            
            # Calculate correlation with profiles
            major_corr = np.corrcoef(rotated_chroma, major_profile)[0, 1]
            minor_corr = np.corrcoef(rotated_chroma, minor_profile)[0, 1]
            
            correlations.append((f"{key_names[i]} major", major_corr))
            correlations.append((f"{key_names[i]} minor", minor_corr))
        
        # Return best matching key
        best_key = max(correlations, key=lambda x: x[1])
        return best_key[0]
    
    def create_deep_embedding(self, features: Dict) -> np.ndarray:
        """Create deep embedding using neural network"""
        try:
            with torch.no_grad():
                # Get global features
                global_features = features['global_features']
                
                # Normalize if scaler is fitted
                if hasattr(self, '_scaler_fitted'):
                    # Pad or truncate to expected size
                    expected_size = self.scaler.n_features_in_
                    if len(global_features) < expected_size:
                        global_features = np.pad(global_features, 
                                               (0, expected_size - len(global_features)))
                    elif len(global_features) > expected_size:
                        global_features = global_features[:expected_size]
                    
                    global_features = self.scaler.transform(global_features.reshape(1, -1))
                    global_features = self.pca.transform(global_features)
                else:
                    # First time - fit the scaler and PCA
                    self._fit_preprocessing(global_features)
                    global_features = self.scaler.transform(global_features.reshape(1, -1))
                    global_features = self.pca.transform(global_features)
                
                # Convert to tensor
                features_tensor = torch.FloatTensor(global_features).to(self.device)
                
                # Generate embedding
                embedding, attention = self.embedding_model(features_tensor)
                
                return embedding.cpu().numpy().squeeze()
                
        except Exception as e:
            logger.error(f"Error creating embedding: {e}")
            # Return random embedding as fallback
            return np.random.randn(64)
    
    def _fit_preprocessing(self, sample_features: np.ndarray):
        """Fit preprocessing on first sample"""
        # Generate more samples for robust fitting
        n_samples = 1000
        feature_dim = len(sample_features)
        
        # Create synthetic samples with noise
        samples = []
        for _ in range(n_samples):
            noise = np.random.randn(feature_dim) * 0.1
            sample = sample_features + noise
            samples.append(sample)
        
        samples = np.array(samples)
        
        # Fit scaler and PCA
        self.scaler.fit(samples)
        scaled_samples = self.scaler.transform(samples)
        self.pca.fit(scaled_samples)
        
        self._scaler_fitted = True
        
        # Save preprocessing models
        with open(self.dirs['models'] / 'scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        with open(self.dirs['models'] / 'pca.pkl', 'wb') as f:
            pickle.dump(self.pca, f)
    
    def scrape_jgb_index(self) -> List[Dict]:
        """Scrape the main JGB index page with setlist links"""
        try:
            url = "https://tela.sugarmegs.org/alpha/j.html"
            response = self.session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find all show entries in the table
            jgb_shows = []
            
            # Look for table rows with Jerry Garcia Band content
            for row in soup.find_all('tr'):
                cells = row.find_all('td')
                if len(cells) >= 4:  # Should have: Setlist | Download | Download | Stream
                    # Check if this row contains JGB content
                    row_text = row.get_text().lower()
                    if 'jerry garcia band' in row_text or 'jgb' in row_text or 'garcia' in row_text:
                        # Extract links
                        setlist_link = None
                        mp3_link = None
                        asx_link = None
                        
                        for cell in cells:
                            links = cell.find_all('a')
                            for link in links:
                                href = link.get('href', '')
                                text = link.get_text(strip=True).lower()
                                
                                if 'setlist' in text:
                                    setlist_link = urljoin(url, href)
                                elif text == 'mp3':
                                    mp3_link = href
                                elif '.asx' in href:
                                    asx_link = urljoin(url, href)
                        
                        if asx_link:
                            # Extract show info from ASX filename
                            asx_name = os.path.basename(asx_link).replace('.asx', '')
                            
                            # Extract date
                            date_match = re.search(r'(\d{4}[-/]\d{2}[-/]\d{2})', asx_name)
                            if date_match:
                                date = date_match.group(1).replace('/', '-')
                            else:
                                # Try alternative date format
                                date_match = re.search(r'(\d{4})-(\d{2})-(\d{2})', row_text)
                                if date_match:
                                    date = f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}"
                                else:
                                    continue
                            
                            show_info = {
                                'asx_name': asx_name,
                                'date': date,
                                'asx_url': asx_link,
                                'setlist_url': setlist_link,
                                'mp3_url': mp3_link,
                                'venue': self._extract_venue_from_asx(asx_name)
                            }
                            jgb_shows.append(show_info)
            
            logger.info(f"Found {len(jgb_shows)} Jerry Garcia Band shows with metadata")
            return jgb_shows
            
        except Exception as e:
            logger.error(f"Error scraping index: {e}")
            return []
    
    def _extract_venue(self, text: str) -> str:
        """Extract venue from show text"""
        # Remove date and common patterns
        venue = re.sub(r'\d{4}[-/]\d{2}[-/]\d{2}', '', text)
        venue = re.sub(r'jerry garcia band', '', venue, flags=re.IGNORECASE)
        venue = re.sub(r'jgb', '', venue, flags=re.IGNORECASE)
        venue = venue.strip(' -,')
        return venue if venue else "Unknown Venue"
    
    def _extract_venue_from_asx(self, asx_name: str) -> str:
        """Extract venue from ASX filename"""
        # Remove date patterns
        venue = re.sub(r'\d{4}[-/]\d{2}[-/]\d{2}', '', asx_name)
        venue = re.sub(r'^\d+[-_]', '', venue)  # Remove leading numbers
        venue = re.sub(r'[_-]', ' ', venue)  # Replace underscores/dashes with spaces
        venue = re.sub(r'\.asx$', '', venue)  # Remove extension
        venue = re.sub(r'garcia.*?saunders', 'Garcia Saunders', venue, flags=re.IGNORECASE)
        venue = re.sub(r'jgb', 'JGB', venue, flags=re.IGNORECASE)
        return venue.strip() if venue.strip() else "Unknown Venue"
    
    def scrape_show_page(self, show_info: Dict) -> List[Dict]:
        """Scrape individual show page for audio files and setlist"""
        try:
            audio_files = []
            setlist_tracks = []
            
            # Use provided URLs if available
            if 'mp3_url' in show_info and show_info['mp3_url']:
                audio_files.append({
                    'filename': f"{show_info['asx_name']}.mp3",
                    'url': show_info['mp3_url'],
                    'format': 'mp3',
                    'type': 'full_show'
                })
            else:
                # Fallback to archive.org pattern
                mp3_url = f"http://www.archive.org/serve/{show_info['asx_name']}/{show_info['asx_name']}.mp3"
                audio_files.append({
                    'filename': f"{show_info['asx_name']}.mp3",
                    'url': mp3_url,
                    'format': 'mp3',
                    'type': 'full_show'
                })
            
            # Get setlist if URL provided
            if 'setlist_url' in show_info and show_info['setlist_url']:
                try:
                    response = self.session.get(show_info['setlist_url'], timeout=10)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        
                        # Parse setlist - look for track listings
                        track_num = 1
                        
                        # Common patterns in setlist pages
                        for elem in soup.find_all(['p', 'div', 'li', 'br']):
                            text = elem.get_text(strip=True)
                            
                            # Skip empty or short text
                            if not text or len(text) < 3:
                                continue
                            
                            # Look for track indicators
                            if (re.match(r'^\d+[\.\)]?\s+', text) or  # Numbered tracks
                                re.match(r'^[A-Z]\d+[\.\)]?\s+', text) or  # Disc + track
                                any(song in text.lower() for song in [
                                    'deal', 'sugaree', 'fire', 'scarlet', 'tangled',
                                    'shining star', 'mission', 'cats', 'run for the roses',
                                    'knockin', 'positively', 'breadbox', 'midnight moonlight',
                                    'think', 'harder they come', 'sisters and brothers',
                                    'rhapsody in red', 'evangeline', 'gomorrah', 'reuben',
                                    'lonesome', 'russian lullaby', 'palm sunday'
                                ])):
                                
                                # Clean track name
                                track_name = re.sub(r'^\d+[\.\)]?\s+', '', text)
                                track_name = re.sub(r'^[A-Z]\d+[\.\)]?\s+', '', track_name)
                                track_name = track_name.strip()
                                
                                if track_name:
                                    setlist_tracks.append({
                                        'track_num': track_num,
                                        'track_name': track_name,
                                        'original_text': text
                                    })
                                    
                                    # Try individual track URL
                                    track_url = f"http://www.archive.org/serve/{show_info['asx_name']}/{show_info['asx_name']}_t{track_num:02d}.mp3"
                                    audio_files.append({
                                        'filename': f"{show_info['asx_name']}_t{track_num:02d}.mp3",
                                        'url': track_url,
                                        'format': 'mp3',
                                        'type': 'track',
                                        'track_name': track_name,
                                        'track_num': track_num
                                    })
                                    track_num += 1
                        
                        # Store parsed setlist
                        show_info['setlist'] = setlist_tracks
                        logger.info(f"Parsed {len(setlist_tracks)} tracks from setlist")
                        
                except Exception as e:
                    logger.debug(f"Error parsing setlist: {e}")
            
            return audio_files
            
        except Exception as e:
            logger.error(f"Error scraping show page: {e}")
            return []
    
    def download_audio_file(self, file_info: Dict, show_dir: Path) -> Optional[Path]:
        """Download individual audio file"""
        try:
            output_path = show_dir / file_info['filename']
            
            # Skip if already downloaded
            if output_path.exists() and output_path.stat().st_size > 1000000:  # > 1MB
                logger.info(f"Already exists: {file_info['filename']}")
                return output_path
            
            # First check if URL exists with HEAD request
            try:
                head_response = self.session.head(file_info['url'], timeout=5, allow_redirects=True)
                if head_response.status_code == 404:
                    logger.debug(f"File not found: {file_info['url']}")
                    return None
            except:
                pass
            
            # Download with streaming
            response = self.session.get(file_info['url'], stream=True, timeout=30, allow_redirects=True)
            
            # Check if we got a valid response
            if response.status_code != 200:
                logger.debug(f"HTTP {response.status_code} for {file_info['url']}")
                return None
            
            # Check content type
            content_type = response.headers.get('content-type', '')
            if 'audio' not in content_type and 'octet-stream' not in content_type:
                logger.debug(f"Not audio content: {content_type}")
                return None
            
            # Get file size
            total_size = int(response.headers.get('content-length', 0))
            if total_size < 1000000:  # Less than 1MB probably not a full audio file
                logger.debug(f"File too small: {total_size} bytes")
                return None
            
            # Download with progress
            with open(output_path, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Log progress for large files
                        if total_size > 0 and downloaded % (1024 * 1024 * 50) == 0:
                            progress = (downloaded / total_size) * 100
                            logger.info(f"Progress: {file_info['filename']} - {progress:.1f}%")
            
            # Verify download
            if output_path.stat().st_size < 1000000:
                output_path.unlink()  # Delete invalid file
                return None
            
            logger.info(f"✓ Downloaded: {file_info['filename']} ({output_path.stat().st_size / 1e6:.1f} MB)")
            return output_path
            
        except requests.exceptions.RequestException as e:
            logger.debug(f"Request error for {file_info['filename']}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error downloading {file_info['filename']}: {e}")
            return None
    
    def download_asx_file(self, show_info: Dict, show_dir: Path) -> Optional[Dict]:
        """Download and parse ASX playlist file"""
        try:
            asx_url = show_info.get('asx_url')
            if not asx_url:
                return None
            
            asx_path = show_dir / f"{show_info['asx_name']}.asx"
            
            # Download ASX file
            response = self.session.get(asx_url, timeout=10)
            if response.status_code == 200:
                with open(asx_path, 'w') as f:
                    f.write(response.text)
                
                # Parse ASX content for track information
                asx_data = self.parse_asx_content(response.text)
                logger.info(f"Downloaded ASX playlist with {len(asx_data.get('entries', []))} entries")
                return asx_data
        except Exception as e:
            logger.debug(f"Error downloading ASX: {e}")
        return None
    
    def parse_asx_content(self, asx_content: str) -> Dict:
        """Parse ASX playlist content for track information"""
        entries = []
        
        # ASX files are XML-like playlists
        import xml.etree.ElementTree as ET
        
        try:
            # Clean up content (ASX files often have issues)
            asx_content = asx_content.replace('&', '&amp;')
            
            # Try to parse as XML
            root = ET.fromstring(asx_content)
            
            for entry in root.findall('.//Entry'):
                entry_data = {}
                
                # Get title
                title_elem = entry.find('Title')
                if title_elem is not None and title_elem.text:
                    entry_data['title'] = title_elem.text.strip()
                
                # Get URL
                ref_elem = entry.find('Ref')
                if ref_elem is not None:
                    entry_data['url'] = ref_elem.get('href', '')
                
                # Get author/artist
                author_elem = entry.find('Author')
                if author_elem is not None and author_elem.text:
                    entry_data['author'] = author_elem.text.strip()
                
                if entry_data:
                    entries.append(entry_data)
        except:
            # Fallback: regex parsing if XML fails
            import re
            
            # Find all entries
            entry_pattern = r'<Entry>(.*?)</Entry>'
            title_pattern = r'<Title>(.*?)</Title>'
            ref_pattern = r'<Ref\s+href="([^"]+)"'
            
            for entry_match in re.finditer(entry_pattern, asx_content, re.IGNORECASE | re.DOTALL):
                entry_text = entry_match.group(1)
                entry_data = {}
                
                # Extract title
                title_match = re.search(title_pattern, entry_text, re.IGNORECASE)
                if title_match:
                    entry_data['title'] = title_match.group(1).strip()
                
                # Extract URL
                ref_match = re.search(ref_pattern, entry_text, re.IGNORECASE)
                if ref_match:
                    entry_data['url'] = ref_match.group(1)
                
                if entry_data:
                    entries.append(entry_data)
        
        return {
            'entries': entries,
            'total_tracks': len(entries)
        }
    
    def download_setlist_html(self, show_info: Dict, show_dir: Path) -> Optional[Dict]:
        """Download and parse the full setlist HTML page"""
        try:
            setlist_url = show_info.get('setlist_url')
            if not setlist_url:
                return None
            
            setlist_path = show_dir / "setlist.html"
            
            # Download setlist page
            response = self.session.get(setlist_url, timeout=10)
            if response.status_code == 200:
                with open(setlist_path, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                
                # Parse setlist for detailed information
                setlist_data = self.parse_detailed_setlist(response.text)
                logger.info(f"Downloaded setlist with {len(setlist_data.get('songs', []))} songs")
                return setlist_data
        except Exception as e:
            logger.debug(f"Error downloading setlist: {e}")
        return None
    
    def parse_detailed_setlist(self, html_content: str) -> Dict:
        """Parse detailed setlist information from HTML"""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        setlist_data = {
            'songs': [],
            'venue': None,
            'date': None,
            'notes': [],
            'band_members': []
        }
        
        # Extract all text content
        text_content = soup.get_text()
        lines = [line.strip() for line in text_content.split('\n') if line.strip()]
        
        # Common JGB songs to look for
        jgb_songs = {
            'deal', 'sugaree', 'mission in the rain', 'the way you do the things you do',
            'cats under the stars', 'rhapsody in red', 'run for the roses', 'gomorrah',
            'midnight moonlight', 'simple twist of fate', 'tangled up in blue',
            'positively 4th street', 'knockin on heaven\'s door', 'i shall be released',
            'the night they drove old dixie down', 'dont let go', 'how sweet it is',
            'let it rock', 'the harder they come', 'sitting in limbo', 'stop that train',
            'tore up over you', 'thats what love will make you do', 'evangeline',
            'ill take a melody', 'russian lullaby', 'love in the afternoon'
        }
        
        song_num = 1
        for line in lines:
            line_lower = line.lower()
            
            # Check if line contains a song
            for song in jgb_songs:
                if song in line_lower:
                    setlist_data['songs'].append({
                        'position': song_num,
                        'title': line,
                        'normalized_title': song
                    })
                    song_num += 1
                    break
            
            # Extract venue
            if 'venue:' in line_lower or any(word in line_lower for word in ['theater', 'theatre', 'auditorium', 'hall']):
                if not setlist_data['venue'] and len(line) > 5:
                    setlist_data['venue'] = line
            
            # Extract band members
            if 'jerry garcia' in line_lower and 'band' not in line_lower:
                setlist_data['band_members'].append(line)
        
        return setlist_data
    
    def process_show(self, show_info: Dict) -> Optional[Dict]:
        """Process a complete show: download and create embeddings"""
        try:
            # Create unique show ID
            if 'asx_url' in show_info:
                url_for_hash = show_info['asx_url']
            elif 'url' in show_info:
                url_for_hash = show_info['url']
            else:
                url_for_hash = show_info.get('asx_name', show_info['date'])
            
            show_id = f"{show_info['date']}_{hashlib.md5(url_for_hash.encode()).hexdigest()[:8]}"
            
            # Skip if already processed
            if show_id in self.processed_shows:
                logger.info(f"Skipping already processed show: {show_id}")
                return None
            
            logger.info(f"Processing show: {show_info['date']} - {show_info['venue']}")
            
            # Get audio files from show page
            audio_files = self.scrape_show_page(show_info)
            
            if not audio_files:
                logger.warning(f"No audio files found for {show_info['date']}")
                return None
            
            # Create show directory
            show_dir = self.dirs['audio'] / show_id
            show_dir.mkdir(exist_ok=True)
            
            # Download ASX playlist file
            asx_data = self.download_asx_file(show_info, show_dir)
            
            # Download full setlist HTML
            setlist_data = self.download_setlist_html(show_info, show_dir)
            
            # Download audio files
            downloaded_files = []
            for file_info in audio_files[:10]:  # Limit files per show
                file_path = self.download_audio_file(file_info, show_dir)
                if file_path:
                    downloaded_files.append(file_path)
                
                # Rate limiting
                time.sleep(0.5)
            
            if not downloaded_files:
                return None
            
            # Process audio and create embeddings with text context
            show_embeddings = []
            show_features = []
            
            for i, audio_file in enumerate(downloaded_files[:5]):  # Process first 5 tracks
                # Extract features
                features_data = self.extract_comprehensive_features(str(audio_file))
                
                if features_data:
                    # Create embedding
                    embedding = self.create_deep_embedding(features_data)
                    
                    # Find matching song info from setlist
                    song_info = None
                    if setlist_data and i < len(setlist_data.get('songs', [])):
                        song_info = setlist_data['songs'][i]
                    
                    # Find matching ASX entry
                    asx_entry = None
                    if asx_data and i < len(asx_data.get('entries', [])):
                        asx_entry = asx_data['entries'][i]
                    
                    # Create enhanced track data with text context
                    track_data = {
                        'file': str(audio_file.name),
                        'embedding': embedding,
                        'tempo': features_data['tempo'],
                        'key': features_data['key'],
                        'duration': features_data['duration'],
                        'features': features_data['feature_dict'],
                        # Text-based context for reinforcement
                        'text_context': {
                            'date': show_info['date'],
                            'venue': show_info.get('venue', setlist_data.get('venue') if setlist_data else None),
                            'song_title': song_info.get('normalized_title') if song_info else None,
                            'song_position': song_info.get('position') if song_info else i + 1,
                            'asx_title': asx_entry.get('title') if asx_entry else None,
                            'year': int(show_info['date'].split('-')[0]),
                            'era': self._determine_era(show_info['date'])
                        }
                    }
                    
                    show_embeddings.append(track_data)
                    show_features.append(features_data)
            
            if show_embeddings:
                # Save embeddings
                embeddings_file = self.dirs['embeddings'] / f"{show_id}_embeddings.h5"
                self._save_embeddings_hdf5(embeddings_file, show_embeddings)
                
                # Save detailed features
                features_file = self.dirs['features'] / f"{show_id}_features.pkl"
                with open(features_file, 'wb') as f:
                    pickle.dump(show_features, f)
                
                # Save comprehensive metadata
                metadata = {
                    'show_id': show_id,
                    'date': show_info['date'],
                    'venue': show_info.get('venue', setlist_data.get('venue') if setlist_data else 'Unknown'),
                    'url': show_info.get('asx_url', show_info.get('url', '')),
                    'audio_files': len(audio_files),
                    'downloaded_files': len(downloaded_files),
                    'processed_tracks': len(show_embeddings),
                    'timestamp': datetime.now().isoformat(),
                    'era': self._determine_era(show_info['date']),
                    'asx_data': asx_data,
                    'setlist': setlist_data,
                    'track_info': [
                        {
                            'file': track['file'],
                            'tempo': track['tempo'],
                            'key': track['key'],
                            'duration': track['duration'],
                            'song_title': track['text_context'].get('song_title'),
                            'asx_title': track['text_context'].get('asx_title')
                        }
                        for track in show_embeddings
                    ]
                }
                
                metadata_file = self.dirs['metadata'] / f"{show_id}_metadata.json"
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                # Mark as processed
                self.processed_shows.add(show_id)
                self._save_processed_shows()
                
                logger.info(f"✓ Completed {show_id}: {len(show_embeddings)} embeddings")
                return metadata
            
        except Exception as e:
            logger.error(f"Error processing show {show_info['date']}: {e}")
            return None
    
    def _determine_era(self, date_str: str) -> str:
        """Determine the musical era based on date"""
        try:
            year = int(date_str.split('-')[0])
            
            if year < 1970:
                return "early_garcia"
            elif 1970 <= year < 1975:
                return "early_jgb"
            elif 1975 <= year < 1980:
                return "classic_jgb"
            elif 1980 <= year < 1985:
                return "middle_jgb"
            elif 1985 <= year < 1990:
                return "late_jgb"
            elif 1990 <= year < 1996:
                return "final_jgb"
            else:
                return "post_garcia"
        except:
            return "unknown_era"
    
    def _save_embeddings_hdf5(self, filepath: Path, embeddings: List[Dict]):
        """Save embeddings with text context in HDF5 format"""
        with h5py.File(filepath, 'w') as f:
            for i, track_data in enumerate(embeddings):
                track_group = f.create_group(f"track_{i}")
                track_group.create_dataset('embedding', data=track_data['embedding'])
                track_group.attrs['file'] = track_data['file']
                track_group.attrs['tempo'] = track_data['tempo']
                track_group.attrs['key'] = track_data['key']
                track_group.attrs['duration'] = track_data['duration']
                
                # Save text context for reinforcement learning
                text_context = track_data.get('text_context', {})
                text_group = track_group.create_group('text_context')
                for key, value in text_context.items():
                    if value is not None:
                        text_group.attrs[key] = str(value)
    
    def batch_process_all_shows(self, limit: Optional[int] = None):
        """Process all JGB shows from sugarmegs"""
        # Get all shows
        shows = self.scrape_jgb_index()
        
        if limit:
            shows = shows[:limit]
        
        logger.info(f"Starting batch processing of {len(shows)} shows")
        
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
                
                # Progress update
                total_done = len(processed) + len(failed)
                if total_done % 10 == 0:
                    logger.info(f"Progress: {total_done}/{len(shows)} shows processed")
        
        # Generate final summary
        self._generate_final_summary(processed, failed, shows)
        
        return processed, failed
    
    def _generate_final_summary(self, processed: List[Dict], 
                               failed: List[Dict], all_shows: List[Dict]):
        """Generate comprehensive summary of scraping results"""
        total_tracks = sum(p.get('processed_tracks', 0) for p in processed)
        total_downloads = sum(p.get('downloaded_files', 0) for p in processed)
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'source': 'sugarmegs.org',
            'total_shows_found': len(all_shows),
            'total_processed': len(processed),
            'total_failed': len(failed),
            'total_tracks_processed': total_tracks,
            'total_files_downloaded': total_downloads,
            'embeddings_created': total_tracks,
            'model_info': {
                'embedding_dim': 64,
                'feature_dim': 256,
                'device': str(self.device)
            },
            'processed_shows': processed,
            'failed_shows': [{'date': s['date'], 'venue': s['venue']} for s in failed]
        }
        
        summary_file = self.output_dir / "scraping_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save model checkpoint
        model_file = self.dirs['models'] / 'embedding_model.pth'
        torch.save({
            'model_state_dict': self.embedding_model.state_dict(),
            'feature_dim': 256,
            'embedding_dim': 64
        }, model_file)
        
        logger.info(f"""
========================================
Sugarmegs Scraping Complete!
========================================
Total Shows Found: {len(all_shows)}
Successfully Processed: {len(processed)}
Failed: {len(failed)}
Total Tracks: {total_tracks}
Total Embeddings: {total_tracks}
Output Directory: {self.output_dir}
========================================
""")


class EmbeddingSimilarityEngine:
    """Engine for finding similar tracks using deep embeddings"""
    
    def __init__(self, embeddings_dir: str):
        self.embeddings_dir = Path(embeddings_dir)
        self.embeddings_cache = {}
        self.features_cache = {}
        self.metadata_cache = {}
        self._load_all_embeddings()
    
    def _load_all_embeddings(self):
        """Load all embeddings with text context from HDF5 files"""
        logger.info("Loading embeddings with text context...")
        
        for embedding_file in self.embeddings_dir.glob("*_embeddings.h5"):
            try:
                with h5py.File(embedding_file, 'r') as f:
                    for track_key in f.keys():
                        track_group = f[track_key]
                        
                        track_id = f"{embedding_file.stem}_{track_group.attrs['file']}"
                        
                        # Load text context if available
                        text_context = {}
                        if 'text_context' in track_group:
                            text_group = track_group['text_context']
                            for attr_key in text_group.attrs:
                                text_context[attr_key] = text_group.attrs[attr_key]
                        
                        self.embeddings_cache[track_id] = {
                            'embedding': track_group['embedding'][:],
                            'tempo': track_group.attrs['tempo'],
                            'key': track_group.attrs['key'],
                            'duration': track_group.attrs['duration'],
                            'show_id': embedding_file.stem.replace('_embeddings', ''),
                            'text_context': text_context
                        }
            except Exception as e:
                logger.error(f"Error loading {embedding_file}: {e}")
        
        # Load metadata files for additional context
        metadata_dir = self.embeddings_dir.parent / 'metadata'
        if metadata_dir.exists():
            for metadata_file in metadata_dir.glob("*_metadata.json"):
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                        self.metadata_cache[metadata['show_id']] = metadata
                except:
                    pass
        
        logger.info(f"Loaded {len(self.embeddings_cache)} track embeddings with context")
    
    def find_similar_tracks(self, reference_track: str, n_similar: int = 10,
                          tempo_weight: float = 0.2, key_weight: float = 0.1,
                          text_weight: float = 0.15) -> List[Tuple[str, float]]:
        """Find similar tracks using weighted similarity with text context"""
        if reference_track not in self.embeddings_cache:
            logger.error(f"Reference track not found: {reference_track}")
            return []
        
        ref_data = self.embeddings_cache[reference_track]
        ref_embedding = ref_data['embedding']
        ref_tempo = ref_data['tempo']
        ref_key = ref_data['key']
        ref_context = ref_data.get('text_context', {})
        
        similarities = []
        
        for track_id, track_data in self.embeddings_cache.items():
            if track_id != reference_track:
                # Embedding similarity (cosine)
                embedding = track_data['embedding']
                cos_sim = np.dot(ref_embedding, embedding) / (
                    np.linalg.norm(ref_embedding) * np.linalg.norm(embedding) + 1e-8
                )
                
                # Tempo similarity
                tempo_diff = abs(ref_tempo - track_data['tempo'])
                tempo_sim = np.exp(-tempo_diff / 20)  # Exponential decay
                
                # Key similarity
                key_sim = 1.0 if ref_key == track_data['key'] else 0.5
                
                # Text context similarity
                text_sim = self._calculate_text_similarity(ref_context, track_data.get('text_context', {}))
                
                # Weighted combination
                total_similarity = (
                    (1 - tempo_weight - key_weight - text_weight) * cos_sim +
                    tempo_weight * tempo_sim +
                    key_weight * key_sim +
                    text_weight * text_sim
                )
                
                similarities.append((track_id, total_similarity, {
                    'embedding_sim': cos_sim,
                    'tempo_sim': tempo_sim,
                    'key_sim': key_sim,
                    'text_sim': text_sim,
                    'tempo': track_data['tempo'],
                    'key': track_data['key'],
                    'context': track_data.get('text_context', {})
                }))
        
        # Sort by total similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:n_similar]
    
    def _calculate_text_similarity(self, ref_context: Dict, track_context: Dict) -> float:
        """Calculate similarity based on text context"""
        similarity_score = 0.0
        weights_sum = 0.0
        
        # Same song title = high similarity
        if ref_context.get('song_title') and track_context.get('song_title'):
            if ref_context['song_title'] == track_context['song_title']:
                similarity_score += 1.0 * 0.4
            weights_sum += 0.4
        
        # Same era = medium similarity
        if ref_context.get('era') and track_context.get('era'):
            if ref_context['era'] == track_context['era']:
                similarity_score += 1.0 * 0.2
            weights_sum += 0.2
        
        # Close dates = scaled similarity
        if ref_context.get('year') and track_context.get('year'):
            year_diff = abs(int(ref_context['year']) - int(track_context['year']))
            year_sim = np.exp(-year_diff / 5)  # 5 year scale
            similarity_score += year_sim * 0.2
            weights_sum += 0.2
        
        # Same venue = small boost
        if ref_context.get('venue') and track_context.get('venue'):
            if ref_context['venue'] == track_context['venue']:
                similarity_score += 1.0 * 0.1
            weights_sum += 0.1
        
        # Song position similarity (for setlist flow)
        if ref_context.get('song_position') and track_context.get('song_position'):
            pos_diff = abs(int(ref_context['song_position']) - int(track_context['song_position']))
            pos_sim = np.exp(-pos_diff / 3)  # 3 position scale
            similarity_score += pos_sim * 0.1
            weights_sum += 0.1
        
        return similarity_score / max(weights_sum, 0.1)
    
    def create_smart_remix_sequence(self, seed_track: str, target_length: int = 10,
                                  diversity_factor: float = 0.3) -> List[Dict]:
        """Create an intelligent remix sequence"""
        sequence = [{'track': seed_track, 'transition_type': 'start'}]
        used_tracks = {seed_track}
        current_track = seed_track
        
        for i in range(target_length - 1):
            # Find similar tracks
            similar_tracks = self.find_similar_tracks(current_track, n_similar=20)
            
            # Filter out used tracks
            candidates = [(t, s, d) for t, s, d in similar_tracks if t not in used_tracks]
            
            if not candidates:
                break
            
            # Select next track with controlled randomness
            if i < target_length - 2:  # Not the last track
                # Add diversity by sometimes picking less similar tracks
                weights = np.array([s for _, s, _ in candidates])
                weights = weights ** (1 / diversity_factor)  # Flatten distribution
                weights = weights / weights.sum()
                
                idx = np.random.choice(len(candidates), p=weights)
                next_track, similarity, details = candidates[idx]
            else:
                # For last track, pick most similar to create smooth ending
                next_track, similarity, details = candidates[0]
            
            # Determine transition type based on similarity
            if details['embedding_sim'] > 0.9:
                transition_type = 'seamless'
            elif details['tempo_sim'] > 0.8:
                transition_type = 'beat_matched'
            elif details['key_sim'] == 1.0:
                transition_type = 'harmonic'
            else:
                transition_type = 'crossfade'
            
            sequence.append({
                'track': next_track,
                'transition_type': transition_type,
                'similarity': similarity,
                'details': details
            })
            
            used_tracks.add(next_track)
            current_track = next_track
        
        return sequence


if __name__ == "__main__":
    # Initialize scraper
    scraper = SugarmegsScraper(
        output_dir="jgb_sugarmegs_complete",
        max_workers=6,
        use_gpu=torch.cuda.is_available()
    )
    
    logger.info("Starting complete Sugarmegs JGB scraping with deep learning embeddings...")
    logger.info("This will download and process all Jerry Garcia Band shows...")
    
    # Process all shows (set limit for testing)
    processed, failed = scraper.batch_process_all_shows(limit=None)  # Remove limit for full scrape
    
    logger.info("Scraping complete! Embeddings are ready for similarity-based remixing.")
    
    # Example: Using the similarity engine
    if processed:
        logger.info("\nTesting similarity engine...")
        similarity_engine = EmbeddingSimilarityEngine(scraper.dirs['embeddings'])
        
        # Get a random track
        sample_track = list(similarity_engine.embeddings_cache.keys())[0]
        logger.info(f"Finding tracks similar to: {sample_track}")
        
        similar = similarity_engine.find_similar_tracks(sample_track, n_similar=5)
        for track, sim, details in similar:
            logger.info(f"  - {track}: {sim:.3f} (tempo: {details['tempo']:.1f}, key: {details['key']})")