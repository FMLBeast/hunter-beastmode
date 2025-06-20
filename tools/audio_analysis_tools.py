"""
Audio Analysis Tools - Advanced Audio Steganography Detection
Supports DeepSpeech, spectral analysis, LSB detection, echo hiding, and more
"""

import asyncio
import logging
import numpy as np
import librosa
import scipy.signal
import scipy.stats
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import tempfile
import subprocess
import json
import wave
import struct
import math
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Advanced audio processing
try:
    import soundfile as sf
    import aubio
    AUBIO_AVAILABLE = True
except ImportError:
    AUBIO_AVAILABLE = False

try:
    import deepspeech
    DEEPSPEECH_AVAILABLE = True
except ImportError:
    DEEPSPEECH_AVAILABLE = False

try:
    import torch
    import torchaudio
    TORCH_AUDIO_AVAILABLE = True
except ImportError:
    TORCH_AUDIO_AVAILABLE = False

try:
    from scipy.fft import fft, fftfreq, stft
    SCIPY_FFT_AVAILABLE = True
except ImportError:
    from scipy.fftpack import fft, fftfreq
    SCIPY_FFT_AVAILABLE = False

class AudioAnalysisTools:
    def __init__(self, config):
        self.config = config.audio_analysis
        self.logger = logging.getLogger(__name__)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="audio_steg_"))
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize DeepSpeech model
        self.deepspeech_model = None
        if self.config.deep_speech_enabled and DEEPSPEECH_AVAILABLE:
            asyncio.create_task(self._load_deepspeech_model())
        
        # Audio processing parameters
        self.default_sr = 22050
        self.frame_length = 2048
        self.hop_length = 512
        
    async def _load_deepspeech_model(self):
        """Load DeepSpeech model for speech-to-text"""
        try:
            model_path = Path(self.config.deep_speech_model)
            if model_path.exists():
                self.deepspeech_model = deepspeech.Model(str(model_path))
                self.logger.info("DeepSpeech model loaded successfully")
            else:
                self.logger.warning(f"DeepSpeech model not found: {model_path}")
        except Exception as e:
            self.logger.error(f"Failed to load DeepSpeech model: {e}")
    
    def execute_method(self, method: str, file_path: Path) -> List[Dict[str, Any]]:
        """Execute audio analysis method"""
        method_map = {
            'spectral_analysis': self._spectral_analysis,
            'lsb_audio': self._lsb_audio_analysis,
            'echo_hiding': self._echo_hiding_detection,
            'phase_coding': self._phase_coding_detection,
            'spread_spectrum': self._spread_spectrum_detection,
            'deep_speech': self._deep_speech_analysis,
            'frequency_masking': self._frequency_masking_detection,
            'amplitude_modulation': self._amplitude_modulation_detection,
            'silence_analysis': self._silence_analysis,
            'harmonic_analysis': self._harmonic_analysis,
            'noise_analysis': self._noise_analysis,
            'temporal_analysis': self._temporal_analysis
        }
        
        if method not in method_map:
            raise ValueError(f"Unknown audio analysis method: {method}")
        
        # Check if file is audio
        if not self._is_audio_file(file_path):
            return []
        
        try:
            return method_map[method](file_path)
        except Exception as e:
            self.logger.error(f"Audio method {method} failed for {file_path}: {e}")
            return [{
                "type": "error",
                "method": method,
                "tool_name": "audio_analysis",
                "confidence": 0.0,
                "details": f"Audio analysis failed: {str(e)}",
                "file_path": str(file_path)
            }]
    
    def _is_audio_file(self, file_path: Path) -> bool:
        """Check if file is an audio file"""
        audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma'}
        return file_path.suffix.lower() in audio_extensions
    
    def _load_audio(self, file_path: Path) -> Tuple[np.ndarray, int]:
        """Load audio file and return samples and sample rate"""
        try:
            # Try librosa first
            audio, sr = librosa.load(str(file_path), sr=None)
            return audio, sr
        except Exception as e:
            self.logger.debug(f"Librosa failed, trying soundfile: {e}")
            
            try:
                # Try soundfile
                if sf:
                    audio, sr = sf.read(str(file_path))
                    if len(audio.shape) > 1:  # Convert stereo to mono
                        audio = np.mean(audio, axis=1)
                    return audio, sr
            except Exception as e:
                self.logger.debug(f"Soundfile failed: {e}")
            
            # Fallback: try subprocess with ffmpeg
            return self._load_audio_ffmpeg(file_path)
    
    def _load_audio_ffmpeg(self, file_path: Path) -> Tuple[np.ndarray, int]:
        """Load audio using ffmpeg as fallback"""
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                # Convert to WAV using ffmpeg
                cmd = [
                    'ffmpeg', '-i', str(file_path), 
                    '-ac', '1',  # Mono
                    '-ar', '22050',  # Sample rate
                    '-y', temp_file.name
                ]
                
                result = subprocess.run(cmd, capture_output=True, timeout=60)
                
                if result.returncode == 0:
                    audio, sr = librosa.load(temp_file.name, sr=None)
                    Path(temp_file.name).unlink()
                    return audio, sr
                else:
                    raise RuntimeError(f"FFmpeg failed: {result.stderr}")
                    
        except Exception as e:
            raise RuntimeError(f"Could not load audio file: {e}")
    
    def _spectral_analysis(self, file_path: Path) -> List[Dict[str, Any]]:
        """Comprehensive spectral analysis for steganography detection"""
        results = []
        
        try:
            audio, sr = self._load_audio(file_path)
            
            # Compute spectrogram
            D = librosa.stft(audio, n_fft=self.frame_length, hop_length=self.hop_length)
            magnitude = np.abs(D)
            phase = np.angle(D)
            
            # 1. Spectral entropy analysis
            spectral_entropy = self._compute_spectral_entropy(magnitude)
            if spectral_entropy > 0.8:  # High entropy threshold
                results.append({
                    "type": "spectral_anomaly",
                    "method": "spectral_entropy",
                    "tool_name": "audio_analysis",
                    "confidence": min(spectral_entropy, 0.9),
                    "details": f"High spectral entropy detected: {spectral_entropy:.3f}",
                    "spectral_entropy": float(spectral_entropy),
                    "file_path": str(file_path)
                })
            
            # 2. Frequency band analysis
            band_analysis = self._analyze_frequency_bands(magnitude, sr)
            if band_analysis:
                results.extend(band_analysis)
            
            # 3. Phase discontinuity detection
            phase_anomalies = self._detect_phase_discontinuities(phase)
            if phase_anomalies:
                results.extend(phase_anomalies)
            
            # 4. Spectral peaks analysis
            peak_analysis = self._analyze_spectral_peaks(magnitude, sr)
            if peak_analysis:
                results.extend(peak_analysis)
            
            # 5. Spectral rolloff analysis
            rolloff_analysis = self._analyze_spectral_rolloff(audio, sr)
            if rolloff_analysis:
                results.extend(rolloff_analysis)
                
        except Exception as e:
            self.logger.error(f"Spectral analysis failed: {e}")
        
        return results
    
    def _compute_spectral_entropy(self, magnitude: np.ndarray) -> float:
        """Compute spectral entropy"""
        # Normalize magnitude spectrum
        power_spectrum = magnitude ** 2
        power_spectrum = power_spectrum / np.sum(power_spectrum, axis=0, keepdims=True)
        
        # Compute entropy for each time frame
        entropies = []
        for t in range(power_spectrum.shape[1]):
            frame = power_spectrum[:, t]
            frame = frame[frame > 0]  # Remove zeros
            if len(frame) > 0:
                entropy = -np.sum(frame * np.log2(frame + 1e-10))
                entropies.append(entropy)
        
        return np.mean(entropies) if entropies else 0.0
    
    def _analyze_frequency_bands(self, magnitude: np.ndarray, sr: int) -> List[Dict[str, Any]]:
        """Analyze frequency bands for anomalies"""
        results = []
        
        # Define frequency bands
        nyquist = sr // 2
        bands = {
            'low': (0, nyquist // 4),
            'mid': (nyquist // 4, nyquist // 2),
            'high': (nyquist // 2, nyquist)
        }
        
        freq_bins = np.linspace(0, nyquist, magnitude.shape[0])
        
        for band_name, (low_freq, high_freq) in bands.items():
            # Find frequency bin indices
            low_idx = np.argmin(np.abs(freq_bins - low_freq))
            high_idx = np.argmin(np.abs(freq_bins - high_freq))
            
            # Extract band energy
            band_energy = np.mean(magnitude[low_idx:high_idx, :])
            total_energy = np.mean(magnitude)
            
            energy_ratio = band_energy / total_energy if total_energy > 0 else 0
            
            # Check for unusual energy distribution
            if band_name == 'high' and energy_ratio > 0.3:
                results.append({
                    "type": "frequency_band_anomaly",
                    "method": "frequency_band_analysis",
                    "tool_name": "audio_analysis",
                    "confidence": min(energy_ratio * 2, 0.8),
                    "details": f"Unusual {band_name}-frequency energy: {energy_ratio:.3f}",
                    "band": band_name,
                    "energy_ratio": float(energy_ratio),
                    "frequency_range": f"{low_freq}-{high_freq} Hz",
                    "file_path": str(file_path) if hasattr(self, 'current_file_path') else "unknown"
                })
        
        return results
    
    def _detect_phase_discontinuities(self, phase: np.ndarray) -> List[Dict[str, Any]]:
        """Detect sudden phase changes that might indicate steganography"""
        results = []
        
        # Compute phase derivatives
        phase_diff = np.diff(phase, axis=1)
        
        # Unwrap phase to handle 2π discontinuities
        phase_unwrapped = np.unwrap(phase, axis=1)
        phase_diff_unwrapped = np.diff(phase_unwrapped, axis=1)
        
        # Find large phase jumps
        threshold = np.pi / 2  # 90 degrees
        large_jumps = np.abs(phase_diff_unwrapped) > threshold
        
        jump_percentage = np.mean(large_jumps)
        
        if jump_percentage > 0.1:  # More than 10% of time-frequency bins have large jumps
            results.append({
                "type": "phase_discontinuity",
                "method": "phase_analysis",
                "tool_name": "audio_analysis",
                "confidence": min(jump_percentage * 5, 0.8),
                "details": f"Phase discontinuities detected in {jump_percentage:.1%} of spectrum",
                "discontinuity_percentage": float(jump_percentage),
                "file_path": str(self.current_file_path) if hasattr(self, 'current_file_path') else "unknown"
            })
        
        return results
    
    def _analyze_spectral_peaks(self, magnitude: np.ndarray, sr: int) -> List[Dict[str, Any]]:
        """Analyze spectral peaks for hidden signals"""
        results = []
        
        # Average magnitude spectrum
        avg_spectrum = np.mean(magnitude, axis=1)
        
        # Find peaks
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(avg_spectrum, height=np.max(avg_spectrum) * 0.1)
        
        if len(peaks) > 20:  # Unusually many peaks
            # Convert peak indices to frequencies
            freqs = librosa.fft_frequencies(sr=sr, n_fft=self.frame_length)
            peak_freqs = freqs[peaks]
            
            results.append({
                "type": "spectral_peaks_anomaly",
                "method": "peak_analysis",
                "tool_name": "audio_analysis",
                "confidence": 0.6,
                "details": f"Unusual number of spectral peaks: {len(peaks)}",
                "peak_count": len(peaks),
                "peak_frequencies": peak_freqs[:10].tolist(),  # First 10 peaks
                "file_path": str(self.current_file_path) if hasattr(self, 'current_file_path') else "unknown"
            })
        
        return results
    
    def _analyze_spectral_rolloff(self, audio: np.ndarray, sr: int) -> List[Dict[str, Any]]:
        """Analyze spectral rolloff for content analysis"""
        results = []
        
        try:
            # Compute spectral rolloff
            rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, roll_percent=0.85)[0]
            
            # Analyze rolloff statistics
            mean_rolloff = np.mean(rolloff)
            std_rolloff = np.std(rolloff)
            
            # Normalize by Nyquist frequency
            nyquist = sr / 2
            mean_rolloff_norm = mean_rolloff / nyquist
            
            # Check for unusual rolloff characteristics
            if mean_rolloff_norm > 0.7:  # High rolloff suggests broadband content
                results.append({
                    "type": "spectral_rolloff_anomaly",
                    "method": "rolloff_analysis",
                    "tool_name": "audio_analysis",
                    "confidence": 0.5,
                    "details": f"High spectral rolloff: {mean_rolloff_norm:.3f}",
                    "mean_rolloff": float(mean_rolloff),
                    "rolloff_variability": float(std_rolloff),
                    "file_path": str(self.current_file_path) if hasattr(self, 'current_file_path') else "unknown"
                })
                
        except Exception as e:
            self.logger.debug(f"Spectral rolloff analysis failed: {e}")
        
        return results
    
    def _lsb_audio_analysis(self, file_path: Path) -> List[Dict[str, Any]]:
        """Detect LSB steganography in audio files"""
        results = []
        
        try:
            audio, sr = self._load_audio(file_path)
            
            # Convert to integer representation
            if audio.dtype == np.float32 or audio.dtype == np.float64:
                # Assume audio is normalized to [-1, 1]
                audio_int = (audio * 32767).astype(np.int16)
            else:
                audio_int = audio.astype(np.int16)
            
            # Extract LSBs
            lsb_sequence = audio_int & 1
            
            # Statistical analysis of LSB sequence
            lsb_analysis = self._analyze_lsb_sequence(lsb_sequence)
            if lsb_analysis:
                results.extend(lsb_analysis)
            
            # Check for patterns in LSBs
            pattern_analysis = self._detect_lsb_patterns(lsb_sequence)
            if pattern_analysis:
                results.extend(pattern_analysis)
                
        except Exception as e:
            self.logger.error(f"LSB audio analysis failed: {e}")
        
        return results
    
    def _analyze_lsb_sequence(self, lsb_sequence: np.ndarray) -> List[Dict[str, Any]]:
        """Analyze LSB sequence for randomness"""
        results = []
        
        # 1. Chi-square test for randomness
        expected_zeros = len(lsb_sequence) / 2
        actual_zeros = np.sum(lsb_sequence == 0)
        actual_ones = len(lsb_sequence) - actual_zeros
        
        chi_square = ((actual_zeros - expected_zeros) ** 2 / expected_zeros + 
                     (actual_ones - expected_zeros) ** 2 / expected_zeros)
        
        # Chi-square critical value for 95% confidence with 1 DOF is 3.84
        if chi_square > 3.84:
            results.append({
                "type": "lsb_randomness_test",
                "method": "chi_square_test",
                "tool_name": "audio_analysis",
                "confidence": min(chi_square / 20, 0.9),  # Normalize
                "details": f"LSB sequence fails randomness test (χ²: {chi_square:.2f})",
                "chi_square": float(chi_square),
                "zeros_ratio": float(actual_zeros / len(lsb_sequence)),
                "file_path": str(self.current_file_path) if hasattr(self, 'current_file_path') else "unknown"
            })
        
        # 2. Autocorrelation test
        autocorr = np.correlate(lsb_sequence, lsb_sequence, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        
        # Normalize
        autocorr = autocorr / autocorr[0]
        
        # Check for significant autocorrelation
        if len(autocorr) > 100:
            max_autocorr = np.max(np.abs(autocorr[10:100]))  # Skip first few lags
            if max_autocorr > 0.3:
                results.append({
                    "type": "lsb_autocorrelation",
                    "method": "autocorrelation_test",
                    "tool_name": "audio_analysis",
                    "confidence": min(max_autocorr * 2, 0.8),
                    "details": f"Significant autocorrelation in LSB sequence: {max_autocorr:.3f}",
                    "max_autocorrelation": float(max_autocorr),
                    "file_path": str(self.current_file_path) if hasattr(self, 'current_file_path') else "unknown"
                })
        
        return results
    
    def _detect_lsb_patterns(self, lsb_sequence: np.ndarray) -> List[Dict[str, Any]]:
        """Detect patterns in LSB sequence"""
        results = []
        
        # Look for repeating patterns
        sequence_str = ''.join(map(str, lsb_sequence[:1000]))  # First 1000 bits
        
        # Check for common patterns
        patterns = ['01010101', '00110011', '11110000', '10101010']
        for pattern in patterns:
            count = sequence_str.count(pattern)
            if count > 5:  # Pattern appears more than 5 times
                results.append({
                    "type": "lsb_pattern",
                    "method": "pattern_detection",
                    "tool_name": "audio_analysis",
                    "confidence": min(count / 10, 0.7),
                    "details": f"Repeating pattern '{pattern}' found {count} times in LSB",
                    "pattern": pattern,
                    "occurrences": count,
                    "file_path": str(self.current_file_path) if hasattr(self, 'current_file_path') else "unknown"
                })
        
        return results
    
    def _echo_hiding_detection(self, file_path: Path) -> List[Dict[str, Any]]:
        """Detect echo hiding steganography"""
        results = []
        
        try:
            audio, sr = self._load_audio(file_path)
            
            # Compute autocorrelation to detect echoes
            autocorr = np.correlate(audio, audio, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            
            # Normalize
            autocorr = autocorr / autocorr[0]
            
            # Look for significant peaks (potential echoes)
            from scipy.signal import find_peaks
            peaks, properties = find_peaks(autocorr[100:], height=0.1)  # Skip first 100 samples
            
            if len(peaks) > 0:
                # Convert peak positions to time delays
                peak_delays = (peaks + 100) / sr  # Add back the 100 sample offset
                
                # Check for typical echo delays (1-50ms)
                echo_delays = peak_delays[(peak_delays > 0.001) & (peak_delays < 0.05)]
                
                if len(echo_delays) > 0:
                    results.append({
                        "type": "echo_hiding",
                        "method": "echo_detection",
                        "tool_name": "audio_analysis",
                        "confidence": 0.7,
                        "details": f"Potential echo hiding detected with {len(echo_delays)} echoes",
                        "echo_delays": echo_delays.tolist(),
                        "echo_strengths": autocorr[peaks + 100].tolist(),
                        "file_path": str(file_path)
                    })
                    
        except Exception as e:
            self.logger.error(f"Echo hiding detection failed: {e}")
        
        return results
    
    def _phase_coding_detection(self, file_path: Path) -> List[Dict[str, Any]]:
        """Detect phase coding steganography"""
        results = []
        
        try:
            audio, sr = self._load_audio(file_path)
            
            # Compute STFT
            D = librosa.stft(audio, n_fft=self.frame_length, hop_length=self.hop_length)
            phase = np.angle(D)
            
            # Analyze phase characteristics
            # 1. Phase coherence analysis
            phase_coherence = self._analyze_phase_coherence(phase)
            if phase_coherence < 0.5:  # Low coherence might indicate manipulation
                results.append({
                    "type": "phase_coding",
                    "method": "phase_coherence",
                    "tool_name": "audio_analysis",
                    "confidence": 0.6,
                    "details": f"Low phase coherence detected: {phase_coherence:.3f}",
                    "phase_coherence": float(phase_coherence),
                    "file_path": str(file_path)
                })
            
            # 2. Phase derivative analysis
            phase_derivatives = self._analyze_phase_derivatives(phase)
            if phase_derivatives:
                results.extend(phase_derivatives)
                
        except Exception as e:
            self.logger.error(f"Phase coding detection failed: {e}")
        
        return results
    
    def _analyze_phase_coherence(self, phase: np.ndarray) -> float:
        """Analyze phase coherence across frequency bins"""
        # Compute phase differences between adjacent frequency bins
        phase_diff = np.diff(phase, axis=0)
        
        # Compute coherence as consistency of phase relationships
        coherence_values = []
        for t in range(phase_diff.shape[1]):
            frame_diff = phase_diff[:, t]
            # Compute circular variance (1 - |mean(exp(i*phase_diff))|)
            complex_sum = np.mean(np.exp(1j * frame_diff))
            coherence = np.abs(complex_sum)
            coherence_values.append(coherence)
        
        return np.mean(coherence_values)
    
    def _analyze_phase_derivatives(self, phase: np.ndarray) -> List[Dict[str, Any]]:
        """Analyze phase derivatives for discontinuities"""
        results = []
        
        # Compute temporal phase derivatives
        phase_dt = np.diff(phase, axis=1)
        
        # Look for abrupt changes
        std_phase_dt = np.std(phase_dt, axis=1)
        mean_std = np.mean(std_phase_dt)
        
        if mean_std > 1.0:  # High phase variability
            results.append({
                "type": "phase_derivative_anomaly",
                "method": "phase_derivative_analysis",
                "tool_name": "audio_analysis",
                "confidence": min(mean_std / 2, 0.7),
                "details": f"High phase derivative variability: {mean_std:.3f}",
                "phase_variability": float(mean_std),
                "file_path": str(self.current_file_path) if hasattr(self, 'current_file_path') else "unknown"
            })
        
        return results
    
    def _spread_spectrum_detection(self, file_path: Path) -> List[Dict[str, Any]]:
        """Detect spread spectrum steganography"""
        results = []
        
        try:
            audio, sr = self._load_audio(file_path)
            
            # Compute power spectral density
            freqs, psd = scipy.signal.welch(audio, sr, nperseg=1024)
            
            # Analyze spectral characteristics
            # 1. Spectral flatness (Wiener entropy)
            spectral_flatness = scipy.stats.gmean(psd) / np.mean(psd)
            
            if spectral_flatness > 0.5:  # High spectral flatness suggests spread spectrum
                results.append({
                    "type": "spread_spectrum",
                    "method": "spectral_flatness",
                    "tool_name": "audio_analysis",
                    "confidence": min(spectral_flatness, 0.8),
                    "details": f"High spectral flatness detected: {spectral_flatness:.3f}",
                    "spectral_flatness": float(spectral_flatness),
                    "file_path": str(file_path)
                })
            
            # 2. Look for noise-like characteristics
            noise_analysis = self._analyze_noise_characteristics(audio, sr)
            if noise_analysis:
                results.extend(noise_analysis)
                
        except Exception as e:
            self.logger.error(f"Spread spectrum detection failed: {e}")
        
        return results
    
    def _analyze_noise_characteristics(self, audio: np.ndarray, sr: int) -> List[Dict[str, Any]]:
        """Analyze audio for noise-like characteristics"""
        results = []
        
        # Compute zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        mean_zcr = np.mean(zcr)
        
        # High ZCR suggests noise-like content
        if mean_zcr > 0.3:
            results.append({
                "type": "noise_characteristics",
                "method": "zero_crossing_rate",
                "tool_name": "audio_analysis",
                "confidence": min(mean_zcr, 0.7),
                "details": f"High zero crossing rate: {mean_zcr:.3f}",
                "zero_crossing_rate": float(mean_zcr),
                "file_path": str(self.current_file_path) if hasattr(self, 'current_file_path') else "unknown"
            })
        
        return results
    
    def _deep_speech_analysis(self, file_path: Path) -> List[Dict[str, Any]]:
        """Use DeepSpeech for speech-to-text analysis"""
        results = []
        
        if not self.deepspeech_model:
            return results
        
        try:
            # Load and preprocess audio for DeepSpeech
            audio, sr = self._load_audio(file_path)
            
            # DeepSpeech expects 16kHz mono audio
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                sr = 16000
            
            # Convert to 16-bit integers
            audio_int16 = (audio * 32767).astype(np.int16)
            
            # Run speech recognition
            text = self.deepspeech_model.stt(audio_int16)
            
            if text.strip():
                # Analyze transcribed text for steganographic content
                text_analysis = self._analyze_transcribed_text(text)
                if text_analysis:
                    results.extend(text_analysis)
                
                results.append({
                    "type": "speech_transcription",
                    "method": "deep_speech",
                    "tool_name": "audio_analysis",
                    "confidence": 0.8,
                    "details": f"Transcribed speech: {text[:100]}...",
                    "transcribed_text": text,
                    "text_length": len(text),
                    "file_path": str(file_path)
                })
                
        except Exception as e:
            self.logger.error(f"DeepSpeech analysis failed: {e}")
        
        return results
    
    def _analyze_transcribed_text(self, text: str) -> List[Dict[str, Any]]:
        """Analyze transcribed text for hidden messages"""
        results = []
        
        # Look for suspicious patterns
        patterns = [
            (r'[A-Za-z0-9+/]{20,}={0,2}', 'base64'),  # Base64
            (r'[0-9a-fA-F]{32,}', 'hexadecimal'),     # Hex
            (r'flag\{.*?\}', 'ctf_flag'),             # CTF flags
            (r'password.*?[:=]\s*\S+', 'password'),    # Passwords
        ]
        
        for pattern, pattern_type in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                results.append({
                    "type": "transcription_pattern",
                    "method": "text_pattern_analysis",
                    "tool_name": "audio_analysis",
                    "confidence": 0.7,
                    "details": f"Found {pattern_type} pattern in transcription",
                    "pattern_type": pattern_type,
                    "matches": matches[:5],  # First 5 matches
                    "file_path": str(self.current_file_path) if hasattr(self, 'current_file_path') else "unknown"
                })
        
        return results
    
    def _frequency_masking_detection(self, file_path: Path) -> List[Dict[str, Any]]:
        """Detect frequency masking steganography"""
        results = []
        
        try:
            audio, sr = self._load_audio(file_path)
            
            # Compute mel-frequency cepstral coefficients
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            
            # Analyze MFCC characteristics
            mfcc_analysis = self._analyze_mfcc_characteristics(mfccs)
            if mfcc_analysis:
                results.extend(mfcc_analysis)
                
        except Exception as e:
            self.logger.error(f"Frequency masking detection failed: {e}")
        
        return results
    
    def _analyze_mfcc_characteristics(self, mfccs: np.ndarray) -> List[Dict[str, Any]]:
        """Analyze MFCC characteristics for anomalies"""
        results = []
        
        # Compute MFCC statistics
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        
        # Check for unusual MFCC patterns
        # High-order MFCCs should generally be small
        high_order_energy = np.mean(np.abs(mfcc_mean[8:]))  # MFCCs 9-13
        
        if high_order_energy > 5.0:  # Threshold for high-order MFCC energy
            results.append({
                "type": "mfcc_anomaly",
                "method": "mfcc_analysis",
                "tool_name": "audio_analysis",
                "confidence": 0.5,
                "details": f"Unusual high-order MFCC energy: {high_order_energy:.2f}",
                "high_order_mfcc_energy": float(high_order_energy),
                "file_path": str(self.current_file_path) if hasattr(self, 'current_file_path') else "unknown"
            })
        
        return results
    
    def _amplitude_modulation_detection(self, file_path: Path) -> List[Dict[str, Any]]:
        """Detect amplitude modulation steganography"""
        results = []
        
        try:
            audio, sr = self._load_audio(file_path)
            
            # Compute envelope
            envelope = np.abs(scipy.signal.hilbert(audio))
            
            # Analyze envelope characteristics
            envelope_analysis = self._analyze_envelope(envelope, sr)
            if envelope_analysis:
                results.extend(envelope_analysis)
                
        except Exception as e:
            self.logger.error(f"Amplitude modulation detection failed: {e}")
        
        return results
    
    def _analyze_envelope(self, envelope: np.ndarray, sr: int) -> List[Dict[str, Any]]:
        """Analyze amplitude envelope for modulation"""
        results = []
        
        # Compute envelope spectrum
        envelope_fft = np.abs(fft(envelope))
        freqs = fftfreq(len(envelope), 1/sr)
        
        # Look for peaks in envelope spectrum (indicating modulation)
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(envelope_fft[:len(envelope_fft)//2], height=np.max(envelope_fft) * 0.1)
        
        if len(peaks) > 5:  # Multiple modulation frequencies
            modulation_freqs = freqs[peaks]
            
            results.append({
                "type": "amplitude_modulation",
                "method": "envelope_analysis",
                "tool_name": "audio_analysis",
                "confidence": 0.6,
                "details": f"Multiple modulation frequencies detected: {len(peaks)}",
                "modulation_frequencies": modulation_freqs[:10].tolist(),
                "peak_count": len(peaks),
                "file_path": str(self.current_file_path) if hasattr(self, 'current_file_path') else "unknown"
            })
        
        return results
    
    def _silence_analysis(self, file_path: Path) -> List[Dict[str, Any]]:
        """Analyze silence periods for hidden data"""
        results = []
        
        try:
            audio, sr = self._load_audio(file_path)
            
            # Detect silence periods
            silence_threshold = 0.01  # Amplitude threshold for silence
            silence_mask = np.abs(audio) < silence_threshold
            
            # Find continuous silence regions
            silence_regions = self._find_continuous_regions(silence_mask)
            
            if silence_regions:
                silence_analysis = self._analyze_silence_regions(silence_regions, audio, sr)
                if silence_analysis:
                    results.extend(silence_analysis)
                    
        except Exception as e:
            self.logger.error(f"Silence analysis failed: {e}")
        
        return results
    
    def _find_continuous_regions(self, mask: np.ndarray) -> List[Tuple[int, int]]:
        """Find continuous regions where mask is True"""
        regions = []
        start = None
        
        for i, value in enumerate(mask):
            if value and start is None:
                start = i
            elif not value and start is not None:
                regions.append((start, i))
                start = None
        
        if start is not None:
            regions.append((start, len(mask)))
        
        return regions
    
    def _analyze_silence_regions(self, silence_regions: List[Tuple[int, int]], audio: np.ndarray, sr: int) -> List[Dict[str, Any]]:
        """Analyze silence regions for hidden data"""
        results = []
        
        # Check if silence regions contain very low-level signals
        for start, end in silence_regions:
            if end - start > sr * 0.1:  # Silence longer than 100ms
                silence_audio = audio[start:end]
                
                # Check for very low amplitude signals in "silence"
                silence_energy = np.mean(silence_audio ** 2)
                
                if silence_energy > 1e-8:  # Very small but non-zero energy
                    results.append({
                        "type": "silence_hiding",
                        "method": "silence_analysis",
                        "tool_name": "audio_analysis",
                        "confidence": 0.4,
                        "details": f"Low-level signal detected in silence region",
                        "silence_duration": (end - start) / sr,
                        "silence_energy": float(silence_energy),
                        "file_path": str(self.current_file_path) if hasattr(self, 'current_file_path') else "unknown"
                    })
        
        return results
    
    def _harmonic_analysis(self, file_path: Path) -> List[Dict[str, Any]]:
        """Analyze harmonic content for steganography"""
        results = []
        
        try:
            audio, sr = self._load_audio(file_path)
            
            # Separate harmonic and percussive components
            harmonic, percussive = librosa.effects.hpss(audio)
            
            # Analyze harmonic-to-percussive ratio
            harmonic_energy = np.mean(harmonic ** 2)
            percussive_energy = np.mean(percussive ** 2)
            
            if percussive_energy > 0:
                hp_ratio = harmonic_energy / percussive_energy
                
                # Unusual ratios might indicate manipulation
                if hp_ratio > 10 or hp_ratio < 0.1:
                    results.append({
                        "type": "harmonic_anomaly",
                        "method": "harmonic_analysis",
                        "tool_name": "audio_analysis",
                        "confidence": 0.5,
                        "details": f"Unusual harmonic-to-percussive ratio: {hp_ratio:.3f}",
                        "hp_ratio": float(hp_ratio),
                        "file_path": str(file_path)
                    })
                    
        except Exception as e:
            self.logger.error(f"Harmonic analysis failed: {e}")
        
        return results
    
    def _noise_analysis(self, file_path: Path) -> List[Dict[str, Any]]:
        """Comprehensive noise analysis"""
        results = []
        
        try:
            audio, sr = self._load_audio(file_path)
            
            # Estimate noise floor
            noise_floor = self._estimate_noise_floor(audio)
            
            # Analyze noise characteristics
            noise_analysis = self._analyze_noise_floor(audio, noise_floor)
            if noise_analysis:
                results.extend(noise_analysis)
                
        except Exception as e:
            self.logger.error(f"Noise analysis failed: {e}")
        
        return results
    
    def _estimate_noise_floor(self, audio: np.ndarray) -> float:
        """Estimate the noise floor of the audio"""
        # Use the 10th percentile of absolute amplitudes
        return np.percentile(np.abs(audio), 10)
    
    def _analyze_noise_floor(self, audio: np.ndarray, noise_floor: float) -> List[Dict[str, Any]]:
        """Analyze noise floor characteristics"""
        results = []
        
        # Extract likely noise segments (below noise floor)
        noise_mask = np.abs(audio) <= noise_floor * 2
        noise_segments = audio[noise_mask]
        
        if len(noise_segments) > 1000:  # Enough noise samples
            # Analyze noise statistics
            noise_entropy = self._calculate_entropy(noise_segments)
            
            if noise_entropy > 0.9:  # High entropy in noise
                results.append({
                    "type": "noise_entropy_anomaly",
                    "method": "noise_floor_analysis",
                    "tool_name": "audio_analysis",
                    "confidence": min(noise_entropy, 0.7),
                    "details": f"High entropy in noise floor: {noise_entropy:.3f}",
                    "noise_entropy": float(noise_entropy),
                    "noise_floor": float(noise_floor),
                    "file_path": str(self.current_file_path) if hasattr(self, 'current_file_path') else "unknown"
                })
        
        return results
    
    def _calculate_entropy(self, data: np.ndarray) -> float:
        """Calculate Shannon entropy of data"""
        if len(data) == 0:
            return 0
        
        # Quantize data for entropy calculation
        data_quantized = np.round(data * 1000).astype(int)
        values, counts = np.unique(data_quantized, return_counts=True)
        probabilities = counts / len(data_quantized)
        
        # Calculate entropy
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return entropy / np.log2(len(values)) if len(values) > 1 else 0
    
    def _temporal_analysis(self, file_path: Path) -> List[Dict[str, Any]]:
        """Analyze temporal characteristics"""
        results = []
        
        try:
            audio, sr = self._load_audio(file_path)
            
            # Analyze temporal features
            tempo_analysis = self._analyze_tempo_characteristics(audio, sr)
            if tempo_analysis:
                results.extend(tempo_analysis)
                
        except Exception as e:
            self.logger.error(f"Temporal analysis failed: {e}")
        
        return results
    
    def _analyze_tempo_characteristics(self, audio: np.ndarray, sr: int) -> List[Dict[str, Any]]:
        """Analyze tempo and rhythm characteristics"""
        results = []
        
        try:
            # Compute onset strength
            onset_envelope = librosa.onset.onset_strength(y=audio, sr=sr)
            
            # Estimate tempo
            tempo, beats = librosa.beat.beat_track(onset_envelope=onset_envelope, sr=sr)
            
            # Analyze beat consistency
            if len(beats) > 10:
                beat_intervals = np.diff(beats) / sr
                beat_consistency = 1.0 - (np.std(beat_intervals) / np.mean(beat_intervals))
                
                if beat_consistency < 0.5:  # Inconsistent beats
                    results.append({
                        "type": "tempo_inconsistency",
                        "method": "tempo_analysis",
                        "tool_name": "audio_analysis",
                        "confidence": 0.4,
                        "details": f"Inconsistent tempo detected (consistency: {beat_consistency:.3f})",
                        "tempo": float(tempo),
                        "beat_consistency": float(beat_consistency),
                        "file_path": str(self.current_file_path) if hasattr(self, 'current_file_path') else "unknown"
                    })
                    
        except Exception as e:
            self.logger.debug(f"Tempo analysis failed: {e}")
        
        return results
    
    def cleanup(self):
        """Cleanup resources"""
        if self.temp_dir.exists():
            import shutil
            shutil.rmtree(self.temp_dir)
        
        self.executor.shutdown(wait=True)
        
        if self.deepspeech_model:
            del self.deepspeech_model