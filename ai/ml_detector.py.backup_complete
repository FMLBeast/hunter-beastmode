"""
Machine Learning Steganography Detector
GPU-Powered Deep Learning Analysis with Multiple Model Architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import cv2
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import time
import json
from PIL import Image
import librosa
import scipy.signal
from concurrent.futures import ThreadPoolExecutor
import pickle

# Try to import advanced models
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

try:
    from transformers import AutoFeatureExtractor, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class CNNStegoDetector(nn.Module):
    """Custom CNN for steganography detection"""
    
    def __init__(self, num_classes=2):
        super(CNNStegoDetector, self).__init__()
        
        # Convolutional layers designed to detect LSB modifications
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        
        # Pooling and normalization
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        
        # Adaptive pooling for variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Fully connected layers
        self.fc1 = nn.Linear(512 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return F.softmax(x, dim=1)

class NoiseDetector(nn.Module):
    """Noise-based steganography detector (Noiseprint-like)"""
    
    def __init__(self):
        super(NoiseDetector, self).__init__()
        
        # Encoder
        self.enc_conv1 = nn.Conv2d(3, 32, 5, padding=2)
        self.enc_conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.enc_conv3 = nn.Conv2d(64, 128, 5, padding=2)
        
        # Decoder  
        self.dec_conv1 = nn.Conv2d(128, 64, 5, padding=2)
        self.dec_conv2 = nn.Conv2d(64, 32, 5, padding=2)
        self.dec_conv3 = nn.Conv2d(32, 1, 5, padding=2)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(32)
        
    def forward(self, x):
        # Encoder
        x = F.relu(self.bn1(self.enc_conv1(x)))
        x = F.relu(self.bn2(self.enc_conv2(x)))
        x = F.relu(self.bn3(self.enc_conv3(x)))
        
        # Decoder
        x = F.relu(self.bn4(self.dec_conv1(x)))
        x = F.relu(self.bn5(self.dec_conv2(x)))
        noise_map = torch.tanh(self.dec_conv3(x))
        
        return noise_map

class AudioStegDetector(nn.Module):
    """CNN for audio steganography detection"""
    
    def __init__(self, num_classes=2):
        super(AudioStegDetector, self).__init__()
        
        # 1D convolutions for audio spectrograms
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))
        
        self.fc1 = nn.Linear(256 * 2 * 2, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return F.softmax(x, dim=1)

class MLStegDetector:
    def __init__(self, config):
        self.config = config.ml
        self.logger = logging.getLogger(__name__)
        self.device = self._setup_device()
        
        # Model cache
        self.models = {}
        self.ensemble_models = []
        
        # Preprocessing
        self.image_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize models
        asyncio.create_task(self._load_models())
        
        # Thread pool for CPU preprocessing
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def _setup_device(self):
        """Setup GPU device with memory management"""
        if not self.config.gpu_enabled or not torch.cuda.is_available():
            self.logger.info("Using CPU for ML inference")
            return torch.device('cpu')
        
        device = torch.device('cuda:0')
        
        # Set memory fraction if specified
        if self.config.gpu_memory_limit:
            try:
                torch.cuda.set_per_process_memory_fraction(
                    self.config.gpu_memory_limit / torch.cuda.get_device_properties(0).total_memory
                )
            except Exception as e:
                self.logger.warning(f"Could not set GPU memory limit: {e}")
        
        self.logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return device
    
    async def _load_models(self):
        """Load all ML models"""
        model_dir = Path(self.config.model_cache_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Load CNN steganography detector
        try:
            self.models['cnn_detector'] = self._load_cnn_detector()
            self.logger.info("Loaded CNN steganography detector")
        except Exception as e:
            self.logger.error(f"Failed to load CNN detector: {e}")
        
        # Load noise detector
        try:
            self.models['noise_detector'] = self._load_noise_detector()
            self.logger.info("Loaded noise detector")
        except Exception as e:
            self.logger.error(f"Failed to load noise detector: {e}")
        
        # Load audio detector
        try:
            self.models['audio_detector'] = self._load_audio_detector()
            self.logger.info("Loaded audio steganography detector")
        except Exception as e:
            self.logger.error(f"Failed to load audio detector: {e}")
        
        # Load pre-trained models if available
        if TIMM_AVAILABLE:
            try:
                self.models['efficientnet'] = self._load_efficientnet()
                self.logger.info("Loaded EfficientNet model")
            except Exception as e:
                self.logger.error(f"Failed to load EfficientNet: {e}")
        
        # Setup ensemble
        if self.config.use_ensemble:
            self.ensemble_models = [
                model for model in self.models.values() 
                if hasattr(model, 'forward')
            ]
    
    def _load_cnn_detector(self):
        """Load or create CNN detector"""
        model_path = Path(self.config.model_cache_dir) / "cnn_steg_detector.pth"
        
        model = CNNStegoDetector()
        
        if model_path.exists():
            try:
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.logger.info(f"Loaded trained CNN detector from {model_path}")
            except Exception as e:
                self.logger.warning(f"Could not load trained model: {e}, using random weights")
        else:
            self.logger.info("Using randomly initialized CNN detector")
        
        model.to(self.device)
        model.eval()
        return model
    
    def _load_noise_detector(self):
        """Load or create noise detector"""
        model_path = Path(self.config.model_cache_dir) / "noise_detector.pth"
        
        model = NoiseDetector()
        
        if model_path.exists():
            try:
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.logger.info(f"Loaded trained noise detector from {model_path}")
            except Exception as e:
                self.logger.warning(f"Could not load trained model: {e}, using random weights")
        else:
            self.logger.info("Using randomly initialized noise detector")
        
        model.to(self.device)
        model.eval()
        return model
    
    def _load_audio_detector(self):
        """Load or create audio detector"""
        model_path = Path(self.config.model_cache_dir) / "audio_steg_detector.pth"
        
        model = AudioStegDetector()
        
        if model_path.exists():
            try:
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.logger.info(f"Loaded trained audio detector from {model_path}")
            except Exception as e:
                self.logger.warning(f"Could not load trained model: {e}, using random weights")
        else:
            self.logger.info("Using randomly initialized audio detector")
        
        model.to(self.device)
        model.eval()
        return model
    
    def _load_efficientnet(self):
        """Load EfficientNet for transfer learning"""
        model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=2)
        model.to(self.device)
        model.eval()
        return model
    
    def execute_method(self, method: str, file_path: Path) -> List[Dict[str, Any]]:
        """Execute ML method synchronously"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.execute_method_async(method, file_path, 0))
        finally:
            loop.close()
    
    async def execute_method_async(self, method: str, file_path: Path, gpu_id: int = 0) -> List[Dict[str, Any]]:
        """Execute ML method asynchronously"""
        method_map = {
            'cnn_steg_detection': self._detect_image_steganography,
            'noiseprint': self._detect_noise_patterns,
            'audio_steg_detection': self._detect_audio_steganography,
            'anomaly_detection': self._detect_anomalies,
            'ensemble_detection': self._ensemble_detection
        }
        
        if method not in method_map:
            raise ValueError(f"Unknown ML method: {method}")
        
        try:
            return await method_map[method](file_path)
        except Exception as e:
            self.logger.error(f"ML method {method} failed for {file_path}: {e}")
            return [{
                "type": "error",
                "method": method,
                "tool_name": "ml_detector",
                "confidence": 0.0,
                "details": f"ML analysis failed: {str(e)}",
                "file_path": str(file_path)
            }]
    
    async def _detect_image_steganography(self, file_path: Path) -> List[Dict[str, Any]]:
        """Detect steganography in images using CNN"""
        if 'cnn_detector' not in self.models:
            return []
        
        # Check if file is an image
        try:
            image = Image.open(file_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
        except Exception:
            return []  # Not an image
        
        results = []
        
        try:
            # Preprocess image
            input_tensor = self.image_transform(image).unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                output = self.models['cnn_detector'](input_tensor)
                probabilities = output[0].cpu().numpy()
                
                stego_confidence = float(probabilities[1])  # Assuming class 1 is steganography
                
                if stego_confidence > self.config.confidence_threshold:
                    results.append({
                        "type": "steganography_detection",
                        "method": "cnn_steg_detection",
                        "tool_name": "ml_detector",
                        "confidence": stego_confidence,
                        "details": f"CNN detected steganography with {stego_confidence:.3f} confidence",
                        "probabilities": {
                            "clean": float(probabilities[0]),
                            "steganography": float(probabilities[1])
                        },
                        "file_path": str(file_path)
                    })
            
            # Additional analysis: check LSB planes
            lsb_analysis = await self._analyze_lsb_planes(image)
            if lsb_analysis:
                results.extend(lsb_analysis)
                
        except Exception as e:
            self.logger.error(f"CNN detection failed for {file_path}: {e}")
        
        return results
    
    async def _analyze_lsb_planes(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Analyze LSB planes for steganography"""
        results = []
        
        try:
            # Convert to numpy array
            img_array = np.array(image)
            
            # Extract LSB planes for each channel
            for channel in range(3):
                lsb_plane = img_array[:, :, channel] & 1
                
                # Calculate entropy of LSB plane
                entropy = self._calculate_entropy(lsb_plane.flatten())
                
                # High entropy in LSB plane suggests steganography
                if entropy > 0.9:  # Threshold for high entropy
                    results.append({
                        "type": "lsb_anomaly",
                        "method": "lsb_entropy_analysis",
                        "tool_name": "ml_detector",
                        "confidence": min(entropy, 0.95),
                        "details": f"High entropy in LSB plane of channel {channel}: {entropy:.3f}",
                        "channel": channel,
                        "entropy": entropy,
                        "file_path": str(image.filename) if hasattr(image, 'filename') else "unknown"
                    })
                
                # Check for visual patterns in LSB plane
                visual_score = self._check_lsb_visual_patterns(lsb_plane)
                if visual_score > 0.7:
                    results.append({
                        "type": "lsb_visual_pattern",
                        "method": "lsb_visual_analysis",
                        "tool_name": "ml_detector",
                        "confidence": visual_score,
                        "details": f"Suspicious visual patterns in LSB plane of channel {channel}",
                        "channel": channel,
                        "visual_score": visual_score,
                        "file_path": str(image.filename) if hasattr(image, 'filename') else "unknown"
                    })
                    
        except Exception as e:
            self.logger.error(f"LSB analysis failed: {e}")
        
        return results
    
    def _calculate_entropy(self, data):
        """Calculate Shannon entropy of data"""
        if len(data) == 0:
            return 0
        
        # Get frequency of each value
        values, counts = np.unique(data, return_counts=True)
        probabilities = counts / len(data)
        
        # Calculate entropy
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return entropy / np.log2(len(values)) if len(values) > 1 else 0
    
    def _check_lsb_visual_patterns(self, lsb_plane):
        """Check for visual patterns in LSB plane that suggest steganography"""
        # Scale LSB plane to 0-255 for analysis
        visual_plane = lsb_plane * 255
        
        # Calculate gradient magnitude
        grad_x = cv2.Sobel(visual_plane.astype(np.uint8), cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(visual_plane.astype(np.uint8), cv2.CV_64F, 0, 1, ksize=3)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # High gradient suggests structured data (not random noise)
        mean_gradient = np.mean(grad_magnitude)
        
        # Normalize score
        visual_score = min(mean_gradient / 50.0, 1.0)  # Arbitrary normalization
        
        return visual_score
    
    async def _detect_noise_patterns(self, file_path: Path) -> List[Dict[str, Any]]:
        """Detect noise patterns using noise detector"""
        if 'noise_detector' not in self.models:
            return []
        
        try:
            image = Image.open(file_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
        except Exception:
            return []
        
        results = []
        
        try:
            # Preprocess image
            input_tensor = self.image_transform(image).unsqueeze(0).to(self.device)
            
            # Generate noise map
            with torch.no_grad():
                noise_map = self.models['noise_detector'](input_tensor)
                noise_map = noise_map[0, 0].cpu().numpy()  # Remove batch and channel dims
            
            # Analyze noise characteristics
            noise_std = np.std(noise_map)
            noise_mean = np.mean(np.abs(noise_map))
            
            # Look for suspicious noise patterns
            if noise_std > 0.1 or noise_mean > 0.05:
                # Check for structured patterns in noise
                pattern_score = self._analyze_noise_structure(noise_map)
                
                confidence = min(pattern_score * 2, 0.9)
                
                if confidence > self.config.confidence_threshold:
                    results.append({
                        "type": "noise_pattern_detection",
                        "method": "noiseprint",
                        "tool_name": "ml_detector",
                        "confidence": confidence,
                        "details": f"Suspicious noise patterns detected (std: {noise_std:.3f})",
                        "noise_statistics": {
                            "std": float(noise_std),
                            "mean": float(noise_mean),
                            "pattern_score": float(pattern_score)
                        },
                        "file_path": str(file_path)
                    })
                    
        except Exception as e:
            self.logger.error(f"Noise detection failed for {file_path}: {e}")
        
        return results
    
    def _analyze_noise_structure(self, noise_map):
        """Analyze structure in noise map"""
        # Apply FFT to detect periodic patterns
        fft = np.fft.fft2(noise_map)
        fft_magnitude = np.abs(fft)
        
        # Look for strong frequency components (indicating structure)
        # Exclude DC component
        fft_magnitude[0, 0] = 0
        
        # Calculate ratio of peak to mean
        peak_value = np.max(fft_magnitude)
        mean_value = np.mean(fft_magnitude)
        
        if mean_value > 0:
            pattern_score = peak_value / mean_value / 100  # Normalize
        else:
            pattern_score = 0
        
        return min(pattern_score, 1.0)
    
    async def _detect_audio_steganography(self, file_path: Path) -> List[Dict[str, Any]]:
        """Detect steganography in audio files"""
        if 'audio_detector' not in self.models:
            return []
        
        results = []
        
        try:
            # Load audio file
            audio, sr = librosa.load(str(file_path), sr=22050)
            
            # Generate spectrogram
            spectrogram = librosa.stft(audio, n_fft=2048, hop_length=512)
            magnitude = np.abs(spectrogram)
            
            # Convert to log scale
            log_magnitude = librosa.amplitude_to_db(magnitude)
            
            # Resize to fixed size for model input
            log_magnitude_resized = cv2.resize(log_magnitude, (128, 128))
            
            # Prepare tensor
            input_tensor = torch.FloatTensor(log_magnitude_resized).unsqueeze(0).unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                output = self.models['audio_detector'](input_tensor)
                probabilities = output[0].cpu().numpy()
                
                stego_confidence = float(probabilities[1])
                
                if stego_confidence > self.config.confidence_threshold:
                    results.append({
                        "type": "audio_steganography_detection",
                        "method": "audio_steg_detection",
                        "tool_name": "ml_detector",
                        "confidence": stego_confidence,
                        "details": f"Audio CNN detected steganography with {stego_confidence:.3f} confidence",
                        "probabilities": {
                            "clean": float(probabilities[0]),
                            "steganography": float(probabilities[1])
                        },
                        "file_path": str(file_path)
                    })
            
            # Additional spectral analysis
            spectral_analysis = await self._analyze_audio_spectrum(magnitude, sr)
            if spectral_analysis:
                results.extend(spectral_analysis)
                
        except Exception as e:
            self.logger.error(f"Audio detection failed for {file_path}: {e}")
        
        return results
    
    async def _analyze_audio_spectrum(self, magnitude, sr) -> List[Dict[str, Any]]:
        """Analyze audio spectrum for steganography indicators"""
        results = []
        
        try:
            # Look for unusual frequency distributions
            freq_bins = magnitude.shape[0]
            freq_energies = np.mean(magnitude, axis=1)
            
            # Check for anomalous energy distribution
            # High-frequency emphasis might indicate data hiding
            high_freq_energy = np.mean(freq_energies[freq_bins//2:])
            low_freq_energy = np.mean(freq_energies[:freq_bins//2])
            
            if high_freq_energy > 0 and low_freq_energy > 0:
                hf_ratio = high_freq_energy / low_freq_energy
                
                if hf_ratio > 0.3:  # Unusually high high-frequency content
                    results.append({
                        "type": "spectral_anomaly",
                        "method": "spectral_analysis",
                        "tool_name": "ml_detector",
                        "confidence": min(hf_ratio, 0.8),
                        "details": f"Unusual high-frequency energy ratio: {hf_ratio:.3f}",
                        "hf_ratio": float(hf_ratio),
                        "file_path": "audio_file"
                    })
            
            # Check for periodic patterns in spectrum
            autocorr = np.correlate(freq_energies, freq_energies, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            
            # Look for strong periodic components
            if len(autocorr) > 10:
                peak_idx = np.argmax(autocorr[5:]) + 5  # Skip the main peak
                peak_value = autocorr[peak_idx]
                
                if peak_value > 0.5 * autocorr[0]:  # Strong periodic component
                    results.append({
                        "type": "periodic_pattern",
                        "method": "autocorrelation_analysis",
                        "tool_name": "ml_detector",
                        "confidence": 0.6,
                        "details": f"Strong periodic pattern detected in spectrum",
                        "peak_ratio": float(peak_value / autocorr[0]),
                        "period": peak_idx,
                        "file_path": "audio_file"
                    })
                    
        except Exception as e:
            self.logger.error(f"Spectral analysis failed: {e}")
        
        return results
    
    async def _detect_anomalies(self, file_path: Path) -> List[Dict[str, Any]]:
        """General anomaly detection using multiple methods"""
        results = []
        
        # Try to determine file type and apply appropriate anomaly detection
        try:
            with open(file_path, 'rb') as f:
                header = f.read(512)
            
            # Statistical anomaly detection on file bytes
            byte_frequencies = np.bincount(header, minlength=256)
            byte_entropy = self._calculate_entropy(header)
            
            # High entropy might indicate encrypted/compressed data
            if byte_entropy > 7.5:
                results.append({
                    "type": "high_entropy_anomaly",
                    "method": "entropy_analysis",
                    "tool_name": "ml_detector",
                    "confidence": min((byte_entropy - 7.0) * 2, 0.9),
                    "details": f"High byte entropy detected: {byte_entropy:.3f}",
                    "entropy": float(byte_entropy),
                    "file_path": str(file_path)
                })
            
            # Check for unusual byte distribution
            chi_square = self._chi_square_test(byte_frequencies)
            if chi_square > 1000:  # High chi-square suggests non-random distribution
                results.append({
                    "type": "byte_distribution_anomaly",
                    "method": "chi_square_test",
                    "tool_name": "ml_detector",
                    "confidence": min(chi_square / 5000, 0.8),
                    "details": f"Unusual byte distribution (χ²: {chi_square:.1f})",
                    "chi_square": float(chi_square),
                    "file_path": str(file_path)
                })
                
        except Exception as e:
            self.logger.error(f"Anomaly detection failed for {file_path}: {e}")
        
        return results
    
    def _chi_square_test(self, observed_frequencies):
        """Perform chi-square test for randomness"""
        total_bytes = np.sum(observed_frequencies)
        expected_frequency = total_bytes / 256
        
        # Calculate chi-square statistic
        chi_square = np.sum((observed_frequencies - expected_frequency)**2 / expected_frequency)
        
        return chi_square
    
    async def _ensemble_detection(self, file_path: Path) -> List[Dict[str, Any]]:
        """Run ensemble of multiple models for robust detection"""
        if not self.config.use_ensemble or not self.ensemble_models:
            return []
        
        results = []
        
        # Run all available models
        model_results = {}
        
        # Image models
        if str(file_path).lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
            if 'cnn_detector' in self.models:
                cnn_results = await self._detect_image_steganography(file_path)
                model_results['cnn'] = cnn_results
            
            if 'noise_detector' in self.models:
                noise_results = await self._detect_noise_patterns(file_path)
                model_results['noise'] = noise_results
        
        # Audio models
        elif str(file_path).lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
            if 'audio_detector' in self.models:
                audio_results = await self._detect_audio_steganography(file_path)
                model_results['audio'] = audio_results
        
        # Combine results using voting
        if len(model_results) >= 2:
            confidence_scores = []
            detection_count = 0
            
            for model_name, model_results_list in model_results.items():
                for result in model_results_list:
                    if result['type'] in ['steganography_detection', 'steganography']:
                        confidence_scores.append(result['confidence'])
                        detection_count += 1
            
            if confidence_scores:
                ensemble_confidence = np.mean(confidence_scores)
                agreement_score = detection_count / len(model_results)
                
                final_confidence = ensemble_confidence * agreement_score
                
                if final_confidence > self.config.confidence_threshold:
                    results.append({
                        "type": "ensemble_steganography_detection",
                        "method": "ensemble_detection",
                        "tool_name": "ml_detector",
                        "confidence": final_confidence,
                        "details": f"Ensemble of {len(model_results)} models detected steganography",
                        "individual_confidences": confidence_scores,
                        "agreement_score": agreement_score,
                        "detection_count": detection_count,
                        "file_path": str(file_path)
                    })
        
        return results
    
    def cleanup(self):
        """Cleanup GPU memory and resources"""
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Clear models
        self.models.clear()
        self.ensemble_models.clear()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        self.logger.info("ML detector cleaned up")