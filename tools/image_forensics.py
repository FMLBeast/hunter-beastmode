"""
Image Forensics Tools - Advanced Image Steganography Detection
Includes stegdetect, LSB analysis, noise analysis, error level analysis, and advanced forensics
"""

import asyncio
import logging
import numpy as np
import cv2
from PIL import Image, ImageStat, ExifTags
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import tempfile
import subprocess
import json
import hashlib
import struct
from concurrent.futures import ThreadPoolExecutor
import base64

# Advanced image processing
try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False

try:
    from skimage import measure, morphology, filters, feature, segmentation
    from skimage.metrics import structural_similarity as ssim
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

try:
    import scipy.ndimage
    from scipy.stats import chi2_contingency, entropy
    from scipy.spatial.distance import pdist
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False

class ImageForensicsTools:
    def __init__(self, config):
        self.config = config.image_forensics
        self.logger = logging.getLogger(__name__)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="image_forensics_"))
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Tool availability
        self.tool_availability = self._check_tool_availability()
        
        # Common image formats
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    
    def _check_tool_availability(self) -> Dict[str, bool]:
        """Check availability of external tools"""
        tools = {}
        
        # Check for stegdetect
        try:
            result = subprocess.run(['stegdetect', '--help'], capture_output=True, timeout=10)
            tools['stegdetect'] = result.returncode in [0, 1]  # Some tools return 1 for help
        except:
            tools['stegdetect'] = False
        
        # Check for jphide/jpseek
        try:
            result = subprocess.run(['jphide'], capture_output=True, timeout=10)
            tools['jphide'] = result.returncode in [0, 1]
        except:
            tools['jphide'] = False
        
        return tools
    
    def execute_method(self, method: str, file_path: Path) -> List[Dict[str, Any]]:
        """Execute image forensics method"""
        method_map = {
            'stegdetect': self._run_stegdetect,
            'lsb_analysis': self._lsb_analysis,
            'noise_analysis': self._noise_analysis,
            'error_level_analysis': self._error_level_analysis,
            'jpeg_analysis': self._jpeg_analysis,
            'metadata_analysis': self._metadata_analysis,
            'histogram_analysis': self._histogram_analysis,
            'frequency_analysis': self._frequency_analysis,
            'pixel_analysis': self._pixel_analysis,
            'compression_analysis': self._compression_analysis,
            'resampling_detection': self._resampling_detection,
            'copy_move_detection': self._copy_move_detection,
            'splice_detection': self._splice_detection,
            'chrominance_analysis': self._chrominance_analysis,
            'quantization_analysis': self._quantization_analysis
        }
        
        if method not in method_map:
            raise ValueError(f"Unknown image forensics method: {method}")
        
        # Check if file is an image
        if not self._is_image_file(file_path):
            return []
        
        try:
            return method_map[method](file_path)
        except Exception as e:
            self.logger.error(f"Image forensics method {method} failed for {file_path}: {e}")
            return [{
                "type": "error",
                "method": method,
                "tool_name": "image_forensics",
                "confidence": 0.0,
                "details": f"Image analysis failed: {str(e)}",
                "file_path": str(file_path)
            }]
    
    def _is_image_file(self, file_path: Path) -> bool:
        """Check if file is a supported image file"""
        return file_path.suffix.lower() in self.supported_formats
    
    def _load_image(self, file_path: Path) -> Optional[np.ndarray]:
        """Load image as numpy array"""
        try:
            # Try PIL first
            with Image.open(file_path) as img:
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                elif img.mode == 'P':
                    img = img.convert('RGB')
                return np.array(img)
        except Exception as e:
            self.logger.debug(f"PIL failed, trying OpenCV: {e}")
            
            try:
                # Try OpenCV
                img = cv2.imread(str(file_path))
                if img is not None:
                    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except Exception as e:
                self.logger.debug(f"OpenCV failed: {e}")
        
        return None
    
    def _run_stegdetect(self, file_path: Path) -> List[Dict[str, Any]]:
        """Run stegdetect tool"""
        if not self.tool_availability.get('stegdetect', False):
            return []
        
        results = []
        
        try:
            cmd = ['stegdetect', str(file_path)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.stdout:
                # Parse stegdetect output
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if ':' in line:
                        filename, detection = line.split(':', 1)
                        detection = detection.strip()
                        
                        if detection != 'negative':
                            # Map stegdetect outputs to confidence levels
                            confidence_map = {
                                'jphide': 0.8,
                                'jsteg': 0.8,
                                'outguess': 0.7,
                                'invisible secrets': 0.6,
                                'f5': 0.9
                            }
                            
                            detected_method = detection.lower()
                            confidence = 0.5
                            
                            for method, conf in confidence_map.items():
                                if method in detected_method:
                                    confidence = conf
                                    break
                            
                            results.append({
                                "type": "steganography_detection",
                                "method": "stegdetect",
                                "tool_name": "stegdetect",
                                "confidence": confidence,
                                "details": f"Stegdetect found: {detection}",
                                "detected_method": detection,
                                "file_path": str(file_path)
                            })
                            
        except subprocess.TimeoutExpired:
            self.logger.warning(f"Stegdetect timeout for {file_path}")
        except Exception as e:
            self.logger.error(f"Stegdetect error for {file_path}: {e}")
        
        return results
    
    def _lsb_analysis(self, file_path: Path) -> List[Dict[str, Any]]:
        """Comprehensive LSB steganography analysis"""
        results = []
        
        image = self._load_image(file_path)
        if image is None:
            return results
        
        # Analyze each color channel
        for channel in range(min(3, image.shape[2] if len(image.shape) > 2 else 1)):
            if len(image.shape) == 3:
                channel_data = image[:, :, channel]
            else:
                channel_data = image
            
            # LSB plane extraction and analysis
            lsb_analysis = self._analyze_lsb_plane(channel_data, channel)
            if lsb_analysis:
                results.extend(lsb_analysis)
            
            # Adjacent pixel analysis
            adjacency_analysis = self._analyze_pixel_adjacency(channel_data, channel)
            if adjacency_analysis:
                results.extend(adjacency_analysis)
            
            # Pairs analysis
            pairs_analysis = self._pairs_analysis(channel_data, channel)
            if pairs_analysis:
                results.extend(pairs_analysis)
        
        return results
    
    def _analyze_lsb_plane(self, channel_data: np.ndarray, channel: int) -> List[Dict[str, Any]]:
        """Analyze LSB plane of image channel"""
        results = []
        
        # Extract LSB plane
        lsb_plane = channel_data & 1
        
        # Visual analysis of LSB plane
        visual_analysis = self._analyze_lsb_visual_patterns(lsb_plane, channel)
        if visual_analysis:
            results.extend(visual_analysis)
        
        # Statistical analysis
        statistical_analysis = self._analyze_lsb_statistics(lsb_plane, channel)
        if statistical_analysis:
            results.extend(statistical_analysis)
        
        # Entropy analysis
        entropy_analysis = self._analyze_lsb_entropy(lsb_plane, channel)
        if entropy_analysis:
            results.extend(entropy_analysis)
        
        return results
    
    def _analyze_lsb_visual_patterns(self, lsb_plane: np.ndarray, channel: int) -> List[Dict[str, Any]]:
        """Analyze visual patterns in LSB plane"""
        results = []
        
        # Save LSB plane as image for visual inspection
        lsb_image = (lsb_plane * 255).astype(np.uint8)
        
        # Calculate gradient magnitude to detect structure
        if SKIMAGE_AVAILABLE:
            gradient = filters.sobel(lsb_image)
            structure_score = np.mean(gradient)
            
            if structure_score > 50:  # Threshold for structured content
                results.append({
                    "type": "lsb_structure",
                    "method": "lsb_visual_analysis",
                    "tool_name": "image_forensics",
                    "confidence": min(structure_score / 100, 0.8),
                    "details": f"Structured patterns in LSB plane (channel {channel})",
                    "channel": channel,
                    "structure_score": float(structure_score),
                    "file_path": str(self.current_file_path) if hasattr(self, 'current_file_path') else "unknown"
                })
        
        # Check for text-like patterns using OCR
        if PYTESSERACT_AVAILABLE:
            try:
                text = pytesseract.image_to_string(lsb_image, config='--psm 6')
                if len(text.strip()) > 10:  # Found readable text
                    results.append({
                        "type": "lsb_text",
                        "method": "lsb_ocr_analysis",
                        "tool_name": "image_forensics",
                        "confidence": 0.9,
                        "details": f"Text found in LSB plane (channel {channel}): {text[:50]}...",
                        "channel": channel,
                        "extracted_text": text[:200],
                        "file_path": str(self.current_file_path) if hasattr(self, 'current_file_path') else "unknown"
                    })
            except Exception as e:
                self.logger.debug(f"OCR analysis failed: {e}")
        
        return results
    
    def _analyze_lsb_statistics(self, lsb_plane: np.ndarray, channel: int) -> List[Dict[str, Any]]:
        """Statistical analysis of LSB plane"""
        results = []
        
        # Chi-square test for randomness
        flat_lsb = lsb_plane.flatten()
        zeros = np.sum(flat_lsb == 0)
        ones = np.sum(flat_lsb == 1)
        total = len(flat_lsb)
        
        expected = total / 2
        chi_square = ((zeros - expected) ** 2 / expected + 
                     (ones - expected) ** 2 / expected)
        
        # Critical value for 95% confidence with 1 DOF is 3.84
        if chi_square > 3.84:
            p_value = 1 - chi2_contingency([[zeros, ones], [expected, expected]])[1]
            
            results.append({
                "type": "lsb_randomness_test",
                "method": "lsb_chi_square",
                "tool_name": "image_forensics",
                "confidence": min(chi_square / 20, 0.9),
                "details": f"LSB fails randomness test (channel {channel}, χ²: {chi_square:.2f})",
                "channel": channel,
                "chi_square": float(chi_square),
                "p_value": float(p_value),
                "zeros_ratio": float(zeros / total),
                "file_path": str(self.current_file_path) if hasattr(self, 'current_file_path') else "unknown"
            })
        
        # Run-length analysis
        runs = self._calculate_run_lengths(flat_lsb)
        if runs:
            avg_run_length = np.mean(runs)
            if avg_run_length > 3:  # Long runs suggest structure
                results.append({
                    "type": "lsb_run_length",
                    "method": "lsb_run_analysis",
                    "tool_name": "image_forensics",
                    "confidence": min(avg_run_length / 10, 0.7),
                    "details": f"Long runs in LSB plane (channel {channel}, avg: {avg_run_length:.1f})",
                    "channel": channel,
                    "avg_run_length": float(avg_run_length),
                    "max_run_length": int(max(runs)),
                    "file_path": str(self.current_file_path) if hasattr(self, 'current_file_path') else "unknown"
                })
        
        return results
    
    def _calculate_run_lengths(self, sequence: np.ndarray) -> List[int]:
        """Calculate run lengths in binary sequence"""
        if len(sequence) == 0:
            return []
        
        runs = []
        current_run = 1
        current_value = sequence[0]
        
        for i in range(1, len(sequence)):
            if sequence[i] == current_value:
                current_run += 1
            else:
                runs.append(current_run)
                current_run = 1
                current_value = sequence[i]
        
        runs.append(current_run)
        return runs
    
    def _analyze_lsb_entropy(self, lsb_plane: np.ndarray, channel: int) -> List[Dict[str, Any]]:
        """Entropy analysis of LSB plane"""
        results = []
        
        # Calculate local entropy using sliding window
        window_size = 8
        h, w = lsb_plane.shape
        entropy_map = np.zeros((h - window_size + 1, w - window_size + 1))
        
        for i in range(h - window_size + 1):
            for j in range(w - window_size + 1):
                window = lsb_plane[i:i+window_size, j:j+window_size]
                entropy_map[i, j] = self._calculate_local_entropy(window)
        
        avg_entropy = np.mean(entropy_map)
        entropy_variance = np.var(entropy_map)
        
        # High entropy suggests embedded data
        if avg_entropy > 0.9:
            results.append({
                "type": "lsb_high_entropy",
                "method": "lsb_entropy_analysis",
                "tool_name": "image_forensics",
                "confidence": min(avg_entropy, 0.95),
                "details": f"High entropy in LSB plane (channel {channel}): {avg_entropy:.3f}",
                "channel": channel,
                "avg_entropy": float(avg_entropy),
                "entropy_variance": float(entropy_variance),
                "file_path": str(self.current_file_path) if hasattr(self, 'current_file_path') else "unknown"
            })
        
        # Low entropy variance suggests uniform distribution (possible steganography)
        if entropy_variance < 0.01 and avg_entropy > 0.8:
            results.append({
                "type": "lsb_uniform_entropy",
                "method": "lsb_entropy_variance",
                "tool_name": "image_forensics",
                "confidence": 0.7,
                "details": f"Uniform high entropy in LSB plane (channel {channel})",
                "channel": channel,
                "avg_entropy": float(avg_entropy),
                "entropy_variance": float(entropy_variance),
                "file_path": str(self.current_file_path) if hasattr(self, 'current_file_path') else "unknown"
            })
        
        return results
    
    def _calculate_local_entropy(self, window: np.ndarray) -> float:
        """Calculate entropy of a local window"""
        values, counts = np.unique(window, return_counts=True)
        probabilities = counts / np.sum(counts)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return entropy / np.log2(len(values)) if len(values) > 1 else 0
    
    def _analyze_pixel_adjacency(self, channel_data: np.ndarray, channel: int) -> List[Dict[str, Any]]:
        """Analyze adjacent pixel relationships for steganography"""
        results = []
        
        # Calculate adjacent pixel differences
        diff_horizontal = np.diff(channel_data, axis=1)
        diff_vertical = np.diff(channel_data, axis=0)
        
        # Check LSB correlation between adjacent pixels
        lsb_h = (channel_data[:, :-1] & 1) ^ (channel_data[:, 1:] & 1)
        lsb_v = (channel_data[:-1, :] & 1) ^ (channel_data[1:, :] & 1)
        
        lsb_h_ratio = np.mean(lsb_h)
        lsb_v_ratio = np.mean(lsb_v)
        
        # High XOR ratio suggests LSB steganography
        if lsb_h_ratio > 0.6 or lsb_v_ratio > 0.6:
            results.append({
                "type": "lsb_adjacency_anomaly",
                "method": "pixel_adjacency_analysis",
                "tool_name": "image_forensics",
                "confidence": max(lsb_h_ratio, lsb_v_ratio) - 0.1,
                "details": f"High LSB XOR ratio in adjacent pixels (channel {channel})",
                "channel": channel,
                "horizontal_xor_ratio": float(lsb_h_ratio),
                "vertical_xor_ratio": float(lsb_v_ratio),
                "file_path": str(self.current_file_path) if hasattr(self, 'current_file_path') else "unknown"
            })
        
        return results
    
    def _pairs_analysis(self, channel_data: np.ndarray, channel: int) -> List[Dict[str, Any]]:
        """Pairs analysis for LSB steganography detection"""
        results = []
        
        # Sample analysis (simplified pairs analysis)
        flat_data = channel_data.flatten()
        
        # Group pixels into pairs (even/odd values)
        even_pixels = flat_data[flat_data % 2 == 0]
        odd_pixels = flat_data[flat_data % 2 == 1]
        
        # Calculate expected vs actual ratios
        total_pixels = len(flat_data)
        even_ratio = len(even_pixels) / total_pixels
        odd_ratio = len(odd_pixels) / total_pixels
        
        # For natural images, expect roughly balanced distribution
        balance_score = abs(even_ratio - 0.5)
        
        if balance_score > 0.1:  # Significant imbalance
            results.append({
                "type": "lsb_pairs_imbalance",
                "method": "pairs_analysis",
                "tool_name": "image_forensics",
                "confidence": min(balance_score * 5, 0.8),
                "details": f"LSB pairs imbalance detected (channel {channel})",
                "channel": channel,
                "even_ratio": float(even_ratio),
                "odd_ratio": float(odd_ratio),
                "balance_score": float(balance_score),
                "file_path": str(self.current_file_path) if hasattr(self, 'current_file_path') else "unknown"
            })
        
        return results
    
    def _noise_analysis(self, file_path: Path) -> List[Dict[str, Any]]:
        """Comprehensive noise analysis for steganography detection"""
        results = []
        
        image = self._load_image(file_path)
        if image is None:
            return results
        
        # Convert to grayscale for noise analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Noise estimation using multiple methods
        noise_analysis = self._estimate_noise_levels(gray)
        if noise_analysis:
            results.extend(noise_analysis)
        
        # Noise pattern analysis
        pattern_analysis = self._analyze_noise_patterns(gray)
        if pattern_analysis:
            results.extend(pattern_analysis)
        
        # High-frequency analysis
        freq_analysis = self._analyze_high_frequency_content(gray)
        if freq_analysis:
            results.extend(freq_analysis)
        
        return results
    
    def _estimate_noise_levels(self, gray_image: np.ndarray) -> List[Dict[str, Any]]:
        """Estimate noise levels using multiple methods"""
        results = []
        
        # Method 1: Laplacian variance (blur detection)
        laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()
        
        # Method 2: Noise estimation using median filtering
        median_filtered = cv2.medianBlur(gray_image, 5)
        noise_estimate = np.mean(np.abs(gray_image.astype(float) - median_filtered.astype(float)))
        
        # Method 3: High-frequency energy
        if SCIPY_AVAILABLE:
            # Apply high-pass filter
            kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
            high_freq = scipy.ndimage.convolve(gray_image.astype(float), kernel)
            hf_energy = np.mean(np.abs(high_freq))
        else:
            hf_energy = 0
        
        # Analyze noise characteristics
        if noise_estimate > 10:  # High noise level
            results.append({
                "type": "high_noise_level",
                "method": "noise_estimation",
                "tool_name": "image_forensics",
                "confidence": min(noise_estimate / 50, 0.8),
                "details": f"High noise level detected: {noise_estimate:.2f}",
                "noise_estimate": float(noise_estimate),
                "laplacian_variance": float(laplacian_var),
                "hf_energy": float(hf_energy),
                "file_path": str(file_path)
            })
        
        if hf_energy > 20:  # High high-frequency content
            results.append({
                "type": "high_frequency_anomaly",
                "method": "frequency_analysis",
                "tool_name": "image_forensics",
                "confidence": min(hf_energy / 100, 0.7),
                "details": f"High frequency content detected: {hf_energy:.2f}",
                "hf_energy": float(hf_energy),
                "file_path": str(file_path)
            })
        
        return results
    
    def _analyze_noise_patterns(self, gray_image: np.ndarray) -> List[Dict[str, Any]]:
        """Analyze patterns in image noise"""
        results = []
        
        # Estimate noise using Wiener filtering approach
        # Apply Gaussian blur and subtract from original
        blurred = cv2.GaussianBlur(gray_image, (5, 5), 1.0)
        noise_residual = gray_image.astype(float) - blurred.astype(float)
        
        # Analyze noise statistics
        noise_std = np.std(noise_residual)
        noise_skewness = self._calculate_skewness(noise_residual)
        noise_kurtosis = self._calculate_kurtosis(noise_residual)
        
        # Natural noise should be approximately Gaussian (skewness~0, kurtosis~3)
        if abs(noise_skewness) > 0.5:
            results.append({
                "type": "noise_skewness_anomaly",
                "method": "noise_pattern_analysis",
                "tool_name": "image_forensics",
                "confidence": min(abs(noise_skewness), 0.8),
                "details": f"Unusual noise skewness: {noise_skewness:.3f}",
                "noise_skewness": float(noise_skewness),
                "noise_std": float(noise_std),
                "file_path": str(self.current_file_path) if hasattr(self, 'current_file_path') else "unknown"
            })
        
        if abs(noise_kurtosis - 3) > 2:  # Kurtosis far from Gaussian
            results.append({
                "type": "noise_kurtosis_anomaly",
                "method": "noise_pattern_analysis",
                "tool_name": "image_forensics",
                "confidence": min(abs(noise_kurtosis - 3) / 5, 0.8),
                "details": f"Unusual noise kurtosis: {noise_kurtosis:.3f}",
                "noise_kurtosis": float(noise_kurtosis),
                "noise_std": float(noise_std),
                "file_path": str(self.current_file_path) if hasattr(self, 'current_file_path') else "unknown"
            })
        
        return results
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4)
    
    def _analyze_high_frequency_content(self, gray_image: np.ndarray) -> List[Dict[str, Any]]:
        """Analyze high-frequency content for anomalies"""
        results = []
        
        # Apply DCT to analyze frequency content
        # Divide image into 8x8 blocks like JPEG
        h, w = gray_image.shape
        block_size = 8
        
        high_freq_energy = []
        
        for i in range(0, h - block_size + 1, block_size):
            for j in range(0, w - block_size + 1, block_size):
                block = gray_image[i:i+block_size, j:j+block_size]
                
                # Apply DCT
                dct_block = cv2.dct(block.astype(np.float32))
                
                # Calculate high-frequency energy (bottom-right coefficients)
                hf_coeffs = dct_block[4:, 4:]  # High-frequency region
                hf_energy = np.sum(np.abs(hf_coeffs))
                high_freq_energy.append(hf_energy)
        
        avg_hf_energy = np.mean(high_freq_energy)
        hf_energy_std = np.std(high_freq_energy)
        
        # Unusual high-frequency distribution might indicate steganography
        if avg_hf_energy > 100:  # Threshold for high HF energy
            results.append({
                "type": "high_frequency_energy",
                "method": "dct_frequency_analysis",
                "tool_name": "image_forensics",
                "confidence": min(avg_hf_energy / 500, 0.8),
                "details": f"High frequency energy detected: {avg_hf_energy:.2f}",
                "avg_hf_energy": float(avg_hf_energy),
                "hf_energy_std": float(hf_energy_std),
                "file_path": str(self.current_file_path) if hasattr(self, 'current_file_path') else "unknown"
            })
        
        return results
    
    def _error_level_analysis(self, file_path: Path) -> List[Dict[str, Any]]:
        """Error Level Analysis (ELA) for detecting image manipulation"""
        results = []
        
        # Only works for JPEG images
        if not str(file_path).lower().endswith(('.jpg', '.jpeg')):
            return results
        
        try:
            # Load original image
            original = Image.open(file_path)
            if original.mode != 'RGB':
                original = original.convert('RGB')
            
            # Save at quality 90 and reload
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                original.save(temp_file.name, 'JPEG', quality=90)
                resaved = Image.open(temp_file.name)
                Path(temp_file.name).unlink()
            
            # Calculate difference
            original_array = np.array(original)
            resaved_array = np.array(resaved)
            
            diff = np.abs(original_array.astype(float) - resaved_array.astype(float))
            
            # Enhance the difference
            diff_enhanced = diff * 10  # Amplify differences
            diff_enhanced = np.clip(diff_enhanced, 0, 255)
            
            # Analyze the error level distribution
            ela_analysis = self._analyze_ela_distribution(diff_enhanced)
            if ela_analysis:
                results.extend(ela_analysis)
                
        except Exception as e:
            self.logger.error(f"ELA analysis failed: {e}")
        
        return results
    
    def _analyze_ela_distribution(self, ela_image: np.ndarray) -> List[Dict[str, Any]]:
        """Analyze ELA image for manipulation indicators"""
        results = []
        
        # Convert to grayscale for analysis
        if len(ela_image.shape) == 3:
            ela_gray = cv2.cvtColor(ela_image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            ela_gray = ela_image.astype(np.uint8)
        
        # Calculate statistics
        mean_error = np.mean(ela_gray)
        max_error = np.max(ela_gray)
        error_std = np.std(ela_gray)
        
        # Find bright regions (potential manipulation areas)
        bright_threshold = np.percentile(ela_gray, 95)
        bright_areas = ela_gray > bright_threshold
        bright_percentage = np.mean(bright_areas) * 100
        
        if mean_error > 15:  # High average error level
            results.append({
                "type": "high_error_level",
                "method": "error_level_analysis",
                "tool_name": "image_forensics",
                "confidence": min(mean_error / 50, 0.8),
                "details": f"High error level detected: {mean_error:.2f}",
                "mean_error": float(mean_error),
                "max_error": float(max_error),
                "error_std": float(error_std),
                "bright_percentage": float(bright_percentage),
                "file_path": str(self.current_file_path) if hasattr(self, 'current_file_path') else "unknown"
            })
        
        if bright_percentage > 5:  # Significant bright areas
            results.append({
                "type": "manipulation_indicators",
                "method": "ela_bright_regions",
                "tool_name": "image_forensics",
                "confidence": min(bright_percentage / 20, 0.7),
                "details": f"Potential manipulation areas: {bright_percentage:.1f}% of image",
                "bright_percentage": float(bright_percentage),
                "bright_threshold": float(bright_threshold),
                "file_path": str(self.current_file_path) if hasattr(self, 'current_file_path') else "unknown"
            })
        
        return results
    
    def _jpeg_analysis(self, file_path: Path) -> List[Dict[str, Any]]:
        """Comprehensive JPEG analysis"""
        results = []
        
        # Only analyze JPEG files
        if not str(file_path).lower().endswith(('.jpg', '.jpeg')):
            return results
        
        try:
            # Extract JPEG metadata and structure
            jpeg_info = self._extract_jpeg_info(file_path)
            if jpeg_info:
                results.extend(jpeg_info)
            
            # Quantization table analysis
            quant_analysis = self._analyze_quantization_tables(file_path)
            if quant_analysis:
                results.extend(quant_analysis)
            
            # DCT coefficient analysis
            dct_analysis = self._analyze_dct_coefficients(file_path)
            if dct_analysis:
                results.extend(dct_analysis)
                
        except Exception as e:
            self.logger.error(f"JPEG analysis failed: {e}")
        
        return results
    
    def _extract_jpeg_info(self, file_path: Path) -> List[Dict[str, Any]]:
        """Extract JPEG-specific information"""
        results = []
        
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
            
            # Look for multiple JPEG markers (possible steganography)
            jpeg_markers = []
            i = 0
            while i < len(data) - 1:
                if data[i] == 0xFF and data[i+1] != 0xFF and data[i+1] != 0x00:
                    marker = data[i+1]
                    jpeg_markers.append(marker)
                i += 1
            
            # Analyze marker frequency
            marker_counts = {}
            for marker in jpeg_markers:
                marker_counts[marker] = marker_counts.get(marker, 0) + 1
            
            # Look for unusual markers or frequencies
            if len(marker_counts) > 20:  # Many different markers
                results.append({
                    "type": "jpeg_marker_anomaly",
                    "method": "jpeg_structure_analysis",
                    "tool_name": "image_forensics",
                    "confidence": 0.6,
                    "details": f"Unusual number of JPEG markers: {len(marker_counts)}",
                    "marker_count": len(marker_counts),
                    "total_markers": len(jpeg_markers),
                    "file_path": str(file_path)
                })
            
            # Check for application-specific markers (APP1-APP15)
            app_markers = [marker for marker in jpeg_markers if 0xE1 <= marker <= 0xEF]
            if len(app_markers) > 10:
                results.append({
                    "type": "excessive_app_markers",
                    "method": "jpeg_app_marker_analysis",
                    "tool_name": "image_forensics",
                    "confidence": 0.5,
                    "details": f"Many application markers found: {len(app_markers)}",
                    "app_marker_count": len(app_markers),
                    "file_path": str(file_path)
                })
                
        except Exception as e:
            self.logger.debug(f"JPEG info extraction failed: {e}")
        
        return results
    
    def _analyze_quantization_tables(self, file_path: Path) -> List[Dict[str, Any]]:
        """Analyze JPEG quantization tables for anomalies"""
        results = []
        
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
            
            # Find quantization tables (DQT markers)
            dqt_tables = []
            i = 0
            while i < len(data) - 3:
                if data[i] == 0xFF and data[i+1] == 0xDB:  # DQT marker
                    length = (data[i+2] << 8) | data[i+3]
                    table_data = data[i+4:i+2+length]
                    dqt_tables.append(table_data)
                    i += 2 + length
                else:
                    i += 1
            
            if len(dqt_tables) > 4:  # More than typical number of quantization tables
                results.append({
                    "type": "excessive_quantization_tables",
                    "method": "quantization_table_analysis",
                    "tool_name": "image_forensics",
                    "confidence": 0.6,
                    "details": f"Unusual number of quantization tables: {len(dqt_tables)}",
                    "table_count": len(dqt_tables),
                    "file_path": str(file_path)
                })
            
            # Analyze table values for anomalies
            for i, table in enumerate(dqt_tables):
                if len(table) >= 64:  # Standard 8x8 quantization table
                    qt_values = list(table[:64])
                    
                    # Check for unusual patterns
                    if min(qt_values) == max(qt_values):  # All values the same
                        results.append({
                            "type": "uniform_quantization_table",
                            "method": "quantization_analysis",
                            "tool_name": "image_forensics",
                            "confidence": 0.8,
                            "details": f"Uniform quantization table {i}: all values = {qt_values[0]}",
                            "table_index": i,
                            "uniform_value": qt_values[0],
                            "file_path": str(file_path)
                        })
                    
                    # Check for values of 1 (minimal quantization)
                    ones_count = qt_values.count(1)
                    if ones_count > 32:  # More than half are 1s
                        results.append({
                            "type": "minimal_quantization",
                            "method": "quantization_analysis",
                            "tool_name": "image_forensics",
                            "confidence": 0.7,
                            "details": f"Minimal quantization detected in table {i}: {ones_count}/64 values are 1",
                            "table_index": i,
                            "ones_count": ones_count,
                            "file_path": str(file_path)
                        })
                        
        except Exception as e:
            self.logger.debug(f"Quantization table analysis failed: {e}")
        
        return results
    
    def _analyze_dct_coefficients(self, file_path: Path) -> List[Dict[str, Any]]:
        """Analyze DCT coefficients for steganography indicators"""
        results = []
        
        try:
            # Load image and convert to YUV
            image = self._load_image(file_path)
            if image is None:
                return results
            
            # Convert to YUV (JPEG uses Y component primarily)
            yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            y_channel = yuv[:, :, 0]
            
            # Apply DCT to 8x8 blocks
            h, w = y_channel.shape
            block_size = 8
            
            dct_anomalies = []
            
            for i in range(0, h - block_size + 1, block_size):
                for j in range(0, w - block_size + 1, block_size):
                    block = y_channel[i:i+block_size, j:j+block_size]
                    
                    # Apply DCT
                    dct_block = cv2.dct(block.astype(np.float32))
                    
                    # Analyze coefficient distribution
                    anomaly_score = self._analyze_dct_block(dct_block)
                    if anomaly_score > 0.5:
                        dct_anomalies.append(anomaly_score)
            
            if dct_anomalies:
                avg_anomaly = np.mean(dct_anomalies)
                anomaly_percentage = len(dct_anomalies) / ((h // block_size) * (w // block_size)) * 100
                
                if anomaly_percentage > 5:  # More than 5% of blocks are anomalous
                    results.append({
                        "type": "dct_coefficient_anomaly",
                        "method": "dct_analysis",
                        "tool_name": "image_forensics",
                        "confidence": min(anomaly_percentage / 20, 0.8),
                        "details": f"DCT anomalies in {anomaly_percentage:.1f}% of blocks",
                        "anomaly_percentage": float(anomaly_percentage),
                        "avg_anomaly_score": float(avg_anomaly),
                        "anomalous_blocks": len(dct_anomalies),
                        "file_path": str(file_path)
                    })
                    
        except Exception as e:
            self.logger.debug(f"DCT coefficient analysis failed: {e}")
        
        return results
    
    def _analyze_dct_block(self, dct_block: np.ndarray) -> float:
        """Analyze a single DCT block for anomalies"""
        # Check for unusual coefficient patterns
        
        # 1. Check high-frequency coefficients
        hf_coeffs = dct_block[4:, 4:]  # High-frequency region
        hf_energy = np.sum(np.abs(hf_coeffs))
        
        # 2. Check for regularity in coefficients
        flat_coeffs = dct_block.flatten()[1:]  # Skip DC component
        coeff_std = np.std(flat_coeffs)
        
        # 3. Check for specific patterns that might indicate steganography
        # F5 steganography affects specific DCT coefficients
        suspicious_coeffs = np.abs(flat_coeffs) == 1  # Coefficients with absolute value 1
        suspicious_ratio = np.mean(suspicious_coeffs)
        
        # Combine scores
        anomaly_score = 0
        
        if hf_energy > 50:  # High high-frequency energy
            anomaly_score += 0.3
        
        if coeff_std > 20:  # High coefficient variance
            anomaly_score += 0.3
        
        if suspicious_ratio > 0.2:  # Many coefficients with value ±1
            anomaly_score += 0.4
        
        return min(anomaly_score, 1.0)
    
    def _metadata_analysis(self, file_path: Path) -> List[Dict[str, Any]]:
        """Comprehensive metadata analysis"""
        results = []
        
        try:
            with Image.open(file_path) as img:
                # Extract EXIF data
                exif_data = img._getexif()
                if exif_data:
                    exif_analysis = self._analyze_exif_data(exif_data)
                    if exif_analysis:
                        results.extend(exif_analysis)
                
                # Check for XMP data
                xmp_data = img.info.get('XML:com.adobe.xmp')
                if xmp_data:
                    results.append({
                        "type": "xmp_metadata_present",
                        "method": "metadata_analysis",
                        "tool_name": "image_forensics",
                        "confidence": 0.3,
                        "details": f"XMP metadata found ({len(xmp_data)} bytes)",
                        "xmp_size": len(xmp_data),
                        "file_path": str(file_path)
                    })
                
                # Check for ICC profile
                icc_profile = img.info.get('icc_profile')
                if icc_profile:
                    results.append({
                        "type": "icc_profile_present",
                        "method": "metadata_analysis",
                        "tool_name": "image_forensics",
                        "confidence": 0.2,
                        "details": f"ICC profile found ({len(icc_profile)} bytes)",
                        "icc_size": len(icc_profile),
                        "file_path": str(file_path)
                    })
                    
        except Exception as e:
            self.logger.debug(f"Metadata analysis failed: {e}")
        
        return results
    
    def _analyze_exif_data(self, exif_data: dict) -> List[Dict[str, Any]]:
        """Analyze EXIF data for anomalies"""
        results = []
        
        # Convert numeric tags to names
        exif_dict = {}
        for tag_id, value in exif_data.items():
            tag_name = ExifTags.TAGS.get(tag_id, tag_id)
            exif_dict[tag_name] = value
        
        # Check for suspicious EXIF entries
        suspicious_fields = ['UserComment', 'ImageDescription', 'Software', 'Artist']
        
        for field in suspicious_fields:
            if field in exif_dict:
                value = exif_dict[field]
                if isinstance(value, (str, bytes)):
                    if isinstance(value, bytes):
                        try:
                            value = value.decode('utf-8', errors='ignore')
                        except:
                            continue
                    
                    # Check for unusual content
                    if len(value) > 100:  # Very long field
                        results.append({
                            "type": "suspicious_exif_field",
                            "method": "exif_analysis",
                            "tool_name": "image_forensics",
                            "confidence": 0.6,
                            "details": f"Unusually long {field}: {len(value)} characters",
                            "field_name": field,
                            "field_length": len(value),
                            "field_content": value[:100] + "..." if len(value) > 100 else value,
                            "file_path": str(self.current_file_path) if hasattr(self, 'current_file_path') else "unknown"
                        })
                    
                    # Check for base64-like content
                    if len(value) > 20 and self._looks_like_base64(value):
                        results.append({
                            "type": "base64_in_exif",
                            "method": "exif_base64_detection",
                            "tool_name": "image_forensics",
                            "confidence": 0.8,
                            "details": f"Possible base64 data in {field}",
                            "field_name": field,
                            "field_content": value[:100],
                            "file_path": str(self.current_file_path) if hasattr(self, 'current_file_path') else "unknown"
                        })
        
        # Check for unusual number of EXIF fields
        if len(exif_dict) > 50:
            results.append({
                "type": "excessive_exif_fields",
                "method": "exif_count_analysis",
                "tool_name": "image_forensics",
                "confidence": 0.5,
                "details": f"Unusually many EXIF fields: {len(exif_dict)}",
                "exif_field_count": len(exif_dict),
                "file_path": str(self.current_file_path) if hasattr(self, 'current_file_path') else "unknown"
            })
        
        return results
    
    def _looks_like_base64(self, text: str) -> bool:
        """Check if text looks like base64 encoding"""
        # Remove whitespace
        text = ''.join(text.split())
        
        # Check length (base64 length should be multiple of 4)
        if len(text) % 4 != 0:
            return False
        
        # Check character set
        import string
        base64_chars = string.ascii_letters + string.digits + '+/='
        
        # Allow up to 10% non-base64 characters
        non_base64_chars = sum(1 for c in text if c not in base64_chars)
        if non_base64_chars / len(text) > 0.1:
            return False
        
        return True
    
    def _histogram_analysis(self, file_path: Path) -> List[Dict[str, Any]]:
        """Analyze image histograms for steganography indicators"""
        results = []
        
        image = self._load_image(file_path)
        if image is None:
            return results
        
        # Analyze each color channel
        for channel in range(min(3, image.shape[2] if len(image.shape) > 2 else 1)):
            if len(image.shape) == 3:
                channel_data = image[:, :, channel]
            else:
                channel_data = image
            
            # Calculate histogram
            hist, bins = np.histogram(channel_data, bins=256, range=(0, 256))
            
            # Analyze histogram characteristics
            hist_analysis = self._analyze_histogram_characteristics(hist, channel)
            if hist_analysis:
                results.extend(hist_analysis)
        
        return results
    
    def _analyze_histogram_characteristics(self, histogram: np.ndarray, channel: int) -> List[Dict[str, Any]]:
        """Analyze histogram for steganography indicators"""
        results = []
        
        # 1. Check for pairs of values (LSB steganography indicator)
        pair_anomalies = 0
        for i in range(0, 254, 2):  # Check pairs (0,1), (2,3), etc.
            if histogram[i] > 0 and histogram[i+1] > 0:
                ratio = min(histogram[i], histogram[i+1]) / max(histogram[i], histogram[i+1])
                if ratio > 0.9:  # Very similar values in pair
                    pair_anomalies += 1
        
        if pair_anomalies > 10:  # Many similar pairs
            results.append({
                "type": "histogram_pairs_anomaly",
                "method": "histogram_pairs_analysis",
                "tool_name": "image_forensics",
                "confidence": min(pair_anomalies / 50, 0.8),
                "details": f"Histogram pairs anomaly in channel {channel}: {pair_anomalies} similar pairs",
                "channel": channel,
                "similar_pairs": pair_anomalies,
                "file_path": str(self.current_file_path) if hasattr(self, 'current_file_path') else "unknown"
            })
        
        # 2. Check for unusual histogram shape
        hist_entropy = entropy(histogram + 1)  # Add 1 to avoid log(0)
        
        if hist_entropy > 7.5:  # Very flat histogram
            results.append({
                "type": "flat_histogram",
                "method": "histogram_entropy_analysis", 
                "tool_name": "image_forensics",
                "confidence": min(hist_entropy / 8, 0.7),
                "details": f"Unusually flat histogram in channel {channel}: entropy = {hist_entropy:.2f}",
                "channel": channel,
                "histogram_entropy": float(hist_entropy),
                "file_path": str(self.current_file_path) if hasattr(self, 'current_file_path') else "unknown"
            })
        
        # 3. Check for spikes or gaps
        zero_bins = np.sum(histogram == 0)
        if zero_bins > 50:  # Many empty bins
            results.append({
                "type": "histogram_gaps",
                "method": "histogram_gap_analysis",
                "tool_name": "image_forensics",
                "confidence": min(zero_bins / 100, 0.6),
                "details": f"Many gaps in histogram (channel {channel}): {zero_bins} empty bins",
                "channel": channel,
                "empty_bins": zero_bins,
                "file_path": str(self.current_file_path) if hasattr(self, 'current_file_path') else "unknown"
            })
        
        return results
    
    def _frequency_analysis(self, file_path: Path) -> List[Dict[str, Any]]:
        """Frequency domain analysis using FFT"""
        results = []
        
        image = self._load_image(file_path)
        if image is None:
            return results
        
        # Convert to grayscale for frequency analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Apply 2D FFT
        fft = np.fft.fft2(gray)
        fft_shifted = np.fft.fftshift(fft)
        magnitude_spectrum = np.log(np.abs(fft_shifted) + 1)
        
        # Analyze frequency characteristics
        freq_analysis = self._analyze_frequency_spectrum(magnitude_spectrum)
        if freq_analysis:
            results.extend(freq_analysis)
        
        return results
    
    def _analyze_frequency_spectrum(self, magnitude_spectrum: np.ndarray) -> List[Dict[str, Any]]:
        """Analyze frequency spectrum for anomalies"""
        results = []
        
        h, w = magnitude_spectrum.shape
        center_h, center_w = h // 2, w // 2
        
        # Analyze radial frequency distribution
        y, x = np.ogrid[:h, :w]
        center = [center_h, center_w]
        
        # Calculate distance from center for each pixel
        distances = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        
        # Create radial profile
        max_radius = min(center_h, center_w)
        radial_profile = []
        
        for r in range(1, max_radius):
            mask = (distances >= r-0.5) & (distances < r+0.5)
            if np.any(mask):
                radial_profile.append(np.mean(magnitude_spectrum[mask]))
        
        if len(radial_profile) > 10:
            # Check for unusual frequency distribution
            # High frequencies should generally decrease
            high_freq_energy = np.mean(radial_profile[-len(radial_profile)//4:])  # Last quarter
            low_freq_energy = np.mean(radial_profile[:len(radial_profile)//4])    # First quarter
            
            if high_freq_energy > 0 and low_freq_energy > 0:
                hf_ratio = high_freq_energy / low_freq_energy
                
                if hf_ratio > 0.3:  # Unusually high high-frequency content
                    results.append({
                        "type": "frequency_anomaly",
                        "method": "frequency_spectrum_analysis",
                        "tool_name": "image_forensics",
                        "confidence": min(hf_ratio, 0.8),
                        "details": f"Unusual high-frequency content: ratio = {hf_ratio:.3f}",
                        "hf_lf_ratio": float(hf_ratio),
                        "high_freq_energy": float(high_freq_energy),
                        "low_freq_energy": float(low_freq_energy),
                        "file_path": str(self.current_file_path) if hasattr(self, 'current_file_path') else "unknown"
                    })
        
        return results
    
    def _pixel_analysis(self, file_path: Path) -> List[Dict[str, Any]]:
        """Comprehensive pixel-level analysis"""
        results = []
        
        image = self._load_image(file_path)
        if image is None:
            return results
        
        # Analyze pixel value distributions
        pixel_analysis = self._analyze_pixel_distributions(image)
        if pixel_analysis:
            results.extend(pixel_analysis)
        
        # Analyze pixel correlations
        correlation_analysis = self._analyze_pixel_correlations(image)
        if correlation_analysis:
            results.extend(correlation_analysis)
        
        return results
    
    def _analyze_pixel_distributions(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Analyze pixel value distributions"""
        results = []
        
        # Check for unusual pixel value patterns
        for channel in range(min(3, image.shape[2] if len(image.shape) > 2 else 1)):
            if len(image.shape) == 3:
                channel_data = image[:, :, channel]
            else:
                channel_data = image
            
            flat_data = channel_data.flatten()
            
            # Check for quantization effects
            unique_values = np.unique(flat_data)
            
            # Natural images typically have most pixel values
            if len(unique_values) < 100:  # Very few unique values
                results.append({
                    "type": "limited_pixel_values",
                    "method": "pixel_distribution_analysis",
                    "tool_name": "image_forensics",
                    "confidence": 0.6,
                    "details": f"Limited pixel values in channel {channel}: {len(unique_values)}",
                    "channel": channel,
                    "unique_values": len(unique_values),
                    "file_path": str(self.current_file_path) if hasattr(self, 'current_file_path') else "unknown"
                })
            
            # Check for gaps in pixel values
            if len(unique_values) > 10:
                value_gaps = np.diff(np.sort(unique_values))
                large_gaps = np.sum(value_gaps > 5)  # Gaps larger than 5
                
                if large_gaps > len(unique_values) * 0.1:  # More than 10% are large gaps
                    results.append({
                        "type": "pixel_value_gaps",
                        "method": "pixel_gap_analysis",
                        "tool_name": "image_forensics",
                        "confidence": 0.5,
                        "details": f"Gaps in pixel values (channel {channel}): {large_gaps} large gaps",
                        "channel": channel,
                        "large_gaps": large_gaps,
                        "total_unique": len(unique_values),
                        "file_path": str(self.current_file_path) if hasattr(self, 'current_file_path') else "unknown"
                    })
        
        return results
    
    def _analyze_pixel_correlations(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Analyze correlations between pixel channels"""
        results = []
        
        if len(image.shape) != 3 or image.shape[2] < 3:
            return results  # Need RGB image
        
        # Calculate correlations between channels
        r_channel = image[:, :, 0].flatten()
        g_channel = image[:, :, 1].flatten()
        b_channel = image[:, :, 2].flatten()
        
        # Calculate correlation coefficients
        rg_corr = np.corrcoef(r_channel, g_channel)[0, 1]
        rb_corr = np.corrcoef(r_channel, b_channel)[0, 1]
        gb_corr = np.corrcoef(g_channel, b_channel)[0, 1]
        
        # Check for unusual correlations
        correlations = [rg_corr, rb_corr, gb_corr]
        channel_pairs = ['R-G', 'R-B', 'G-B']
        
        for corr, pair in zip(correlations, channel_pairs):
            if not np.isnan(corr):
                if corr > 0.98:  # Very high correlation
                    results.append({
                        "type": "high_channel_correlation",
                        "method": "channel_correlation_analysis",
                        "tool_name": "image_forensics",
                        "confidence": min((corr - 0.95) * 20, 0.8),
                        "details": f"Very high correlation between {pair}: {corr:.4f}",
                        "channel_pair": pair,
                        "correlation": float(corr),
                        "file_path": str(self.current_file_path) if hasattr(self, 'current_file_path') else "unknown"
                    })
                elif corr < -0.5:  # Unusual negative correlation
                    results.append({
                        "type": "negative_channel_correlation",
                        "method": "channel_correlation_analysis",
                        "tool_name": "image_forensics",
                        "confidence": min(abs(corr), 0.7),
                        "details": f"Unusual negative correlation between {pair}: {corr:.4f}",
                        "channel_pair": pair,
                        "correlation": float(corr),
                        "file_path": str(self.current_file_path) if hasattr(self, 'current_file_path') else "unknown"
                    })
        
        return results
    
    def _compression_analysis(self, file_path: Path) -> List[Dict[str, Any]]:
        """Analyze compression artifacts and characteristics"""
        results = []
        
        # This is a placeholder for compression analysis
        # Real implementation would analyze JPEG compression artifacts,
        # blocking artifacts, quantization effects, etc.
        
        return results
    
    def _resampling_detection(self, file_path: Path) -> List[Dict[str, Any]]:
        """Detect image resampling/interpolation"""
        results = []
        
        image = self._load_image(file_path)
        if image is None:
            return results
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Look for periodic patterns that indicate resampling
        # This is a simplified approach
        
        return results
    
    def _copy_move_detection(self, file_path: Path) -> List[Dict[str, Any]]:
        """Detect copy-move forgery"""
        results = []
        
        # This would implement copy-move detection algorithms
        # Such as block-matching approaches
        
        return results
    
    def _splice_detection(self, file_path: Path) -> List[Dict[str, Any]]:
        """Detect image splicing"""
        results = []
        
        # This would implement splicing detection
        # Using techniques like CFA pattern analysis, lighting inconsistencies, etc.
        
        return results
    
    def _chrominance_analysis(self, file_path: Path) -> List[Dict[str, Any]]:
        """Analyze chrominance components for anomalies"""
        results = []
        
        image = self._load_image(file_path)
        if image is None or len(image.shape) != 3:
            return results
        
        # Convert to YUV
        yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        u_channel = yuv[:, :, 1]
        v_channel = yuv[:, :, 2]
        
        # Analyze chrominance statistics
        u_std = np.std(u_channel)
        v_std = np.std(v_channel)
        
        # Low chrominance variation might indicate steganography
        if u_std < 5 or v_std < 5:
            results.append({
                "type": "low_chrominance_variation",
                "method": "chrominance_analysis",
                "tool_name": "image_forensics",
                "confidence": 0.5,
                "details": f"Low chrominance variation: U_std={u_std:.2f}, V_std={v_std:.2f}",
                "u_std": float(u_std),
                "v_std": float(v_std),
                "file_path": str(file_path)
            })
        
        return results
    
    def _quantization_analysis(self, file_path: Path) -> List[Dict[str, Any]]:
        """Analyze quantization effects"""
        results = []
        
        # This would analyze quantization artifacts in compressed images
        # Look for double quantization, mismatched quality factors, etc.
        
        return results
    
    def cleanup(self):
        """Cleanup resources"""
        if self.temp_dir.exists():
            import shutil
            shutil.rmtree(self.temp_dir)
        
        self.executor.shutdown(wait=True)