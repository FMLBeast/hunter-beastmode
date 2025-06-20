"""
Crypto Analysis Tools - Advanced Cryptographic Analysis and Pattern Detection
Supports entropy analysis, pattern detection, key search, frequency analysis, and more
"""

import asyncio
import logging
import numpy as np
import re
import struct
import hashlib
import base64
import binascii
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import json
from concurrent.futures import ThreadPoolExecutor
import itertools
import string
from collections import Counter, defaultdict
import math

# Advanced crypto libraries
try:
    from Crypto.Cipher import AES, DES, Blowfish, RC4
    from Crypto.Util import Counter
    PYCRYPTO_AVAILABLE = True
except ImportError:
    try:
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        from cryptography.hazmat.backends import default_backend
        CRYPTOGRAPHY_AVAILABLE = True
    except ImportError:
        PYCRYPTO_AVAILABLE = False
        CRYPTOGRAPHY_AVAILABLE = False

# Statistical analysis
try:
    from scipy import stats
    from scipy.fft import fft, fftfreq
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Machine learning for pattern detection
try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

class CryptoAnalysisTools:
    def __init__(self, config):
        self.config = config.crypto
        self.logger = logging.getLogger(__name__)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="crypto_analysis_"))
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Common encryption patterns and signatures
        self.crypto_patterns = self._load_crypto_patterns()
        
        # Tool availability
        self.tool_availability = self._check_tool_availability()
        
        # Wordlists for key cracking
        self.wordlists_dir = Path(self.config.wordlists_dir)
        
        # Statistical constants
        self.english_freq = self._load_english_frequencies()
        
    def _load_crypto_patterns(self) -> Dict[str, Dict]:
        """Load patterns for detecting various cryptographic formats"""
        return {
            # Base64 patterns
            'base64': {
                'pattern': re.compile(r'[A-Za-z0-9+/]{4,}={0,2}'),
                'min_length': 20,
                'description': 'Base64 encoded data'
            },
            
            # Hexadecimal patterns
            'hex': {
                'pattern': re.compile(r'[0-9A-Fa-f]{8,}'),
                'min_length': 16,
                'description': 'Hexadecimal encoded data'
            },
            
            # PEM format
            'pem': {
                'pattern': re.compile(r'-----BEGIN [A-Z ]+-----.*?-----END [A-Z ]+-----', re.DOTALL),
                'min_length': 50,
                'description': 'PEM encoded certificate/key'
            },
            
            # Common hash formats
            'md5': {
                'pattern': re.compile(r'\b[a-f0-9]{32}\b'),
                'min_length': 32,
                'description': 'MD5 hash'
            },
            
            'sha1': {
                'pattern': re.compile(r'\b[a-f0-9]{40}\b'),
                'min_length': 40,
                'description': 'SHA-1 hash'
            },
            
            'sha256': {
                'pattern': re.compile(r'\b[a-f0-9]{64}\b'),
                'min_length': 64,
                'description': 'SHA-256 hash'
            },
            
            # Bitcoin addresses
            'bitcoin': {
                'pattern': re.compile(r'\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b'),
                'min_length': 26,
                'description': 'Bitcoin address'
            },
            
            # Common cipher identifiers
            'cipher_names': {
                'pattern': re.compile(r'\b(AES|DES|3DES|Blowfish|RC4|RSA|DSA|ECDSA)\b', re.IGNORECASE),
                'min_length': 3,
                'description': 'Cipher algorithm names'
            }
        }
    
    def _load_english_frequencies(self) -> Dict[str, float]:
        """Load English language character frequencies for analysis"""
        return {
            'E': 12.70, 'T': 9.06, 'A': 8.17, 'O': 7.51, 'I': 6.97, 'N': 6.75,
            'S': 6.33, 'H': 6.09, 'R': 5.99, 'D': 4.25, 'L': 4.03, 'C': 2.78,
            'U': 2.76, 'M': 2.41, 'W': 2.36, 'F': 2.23, 'G': 2.02, 'Y': 1.97,
            'P': 1.93, 'B': 1.29, 'V': 0.98, 'K': 0.77, 'J': 0.15, 'X': 0.15,
            'Q': 0.10, 'Z': 0.07
        }
    
    def _check_tool_availability(self) -> Dict[str, bool]:
        """Check availability of external cryptographic tools"""
        tools = {}
        
        tool_checks = {
            'hashcat': 'hashcat --version',
            'john': 'john --version',
            'openssl': 'openssl version',
            'gpg': 'gpg --version',
            'hash-identifier': 'hash-identifier --help',
            'findmyhash': 'findmyhash --help'
        }
        
        for tool, check_cmd in tool_checks.items():
            try:
                result = subprocess.run(
                    check_cmd.split(),
                    capture_output=True,
                    timeout=10
                )
                tools[tool] = result.returncode == 0
            except:
                tools[tool] = False
        
        return tools
    
    def execute_method(self, method: str, file_path: Path) -> List[Dict[str, Any]]:
        """Execute cryptographic analysis method"""
        method_map = {
            'entropy_analysis': self._entropy_analysis,
            'pattern_detection': self._pattern_detection,
            'frequency_analysis': self._frequency_analysis,
            'key_search': self._key_search,
            'cipher_detection': self._cipher_detection,
            'hash_analysis': self._hash_analysis,
            'xor_analysis': self._xor_analysis,
            'substitution_analysis': self._substitution_analysis,
            'block_cipher_analysis': self._block_cipher_analysis,
            'stream_cipher_analysis': self._stream_cipher_analysis,
            'compression_detection': self._compression_detection,
            'randomness_tests': self._randomness_tests,
            'correlation_analysis': self._correlation_analysis,
            'autocorrelation_analysis': self._autocorrelation_analysis,
            'bit_distribution_analysis': self._bit_distribution_analysis,
            'ml_pattern_detection': self._ml_pattern_detection
        }
        
        if method not in method_map:
            raise ValueError(f"Unknown crypto analysis method: {method}")
        
        try:
            return method_map[method](file_path)
        except Exception as e:
            self.logger.error(f"Crypto analysis method {method} failed for {file_path}: {e}")
            return [{
                "type": "error",
                "method": method,
                "tool_name": "crypto_analysis",
                "confidence": 0.0,
                "details": f"Cryptographic analysis failed: {str(e)}",
                "file_path": str(file_path)
            }]
    
    def _entropy_analysis(self, file_path: Path) -> List[Dict[str, Any]]:
        """Comprehensive entropy analysis"""
        results = []
        
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
            
            if len(data) == 0:
                return results
            
            # Calculate overall entropy
            overall_entropy = self._calculate_shannon_entropy(data)
            
            # Calculate block entropies
            block_size = 256
            block_entropies = []
            
            for i in range(0, len(data), block_size):
                block = data[i:i + block_size]
                if len(block) >= 32:  # Minimum block size for meaningful entropy
                    block_entropy = self._calculate_shannon_entropy(block)
                    block_entropies.append(block_entropy)
            
            # Analyze entropy characteristics
            if overall_entropy > 7.5:  # High entropy
                confidence = min((overall_entropy - 7.0) * 2, 0.9)
                
                results.append({
                    "type": "high_entropy",
                    "method": "entropy_analysis",
                    "tool_name": "crypto_analysis",
                    "confidence": confidence,
                    "details": f"High entropy detected: {overall_entropy:.3f}",
                    "overall_entropy": float(overall_entropy),
                    "interpretation": "Possible encrypted/compressed data",
                    "file_path": str(file_path)
                })
            
            # Analyze entropy distribution
            if block_entropies:
                entropy_mean = np.mean(block_entropies)
                entropy_std = np.std(block_entropies)
                entropy_variance = np.var(block_entropies)
                
                # Low variance with high mean entropy suggests encryption
                if entropy_mean > 7.0 and entropy_variance < 0.5:
                    results.append({
                        "type": "uniform_high_entropy",
                        "method": "entropy_distribution_analysis",
                        "tool_name": "crypto_analysis",
                        "confidence": 0.8,
                        "details": f"Uniform high entropy distribution (mean: {entropy_mean:.3f}, var: {entropy_variance:.3f})",
                        "entropy_mean": float(entropy_mean),
                        "entropy_variance": float(entropy_variance),
                        "entropy_std": float(entropy_std),
                        "interpretation": "Likely encrypted data",
                        "file_path": str(file_path)
                    })
                
                # High variance suggests mixed content
                elif entropy_variance > 2.0:
                    results.append({
                        "type": "variable_entropy",
                        "method": "entropy_distribution_analysis",
                        "tool_name": "crypto_analysis",
                        "confidence": 0.6,
                        "details": f"Highly variable entropy (variance: {entropy_variance:.3f})",
                        "entropy_variance": float(entropy_variance),
                        "interpretation": "Mixed encrypted/plaintext content",
                        "file_path": str(file_path)
                    })
            
            # Byte frequency analysis
            byte_analysis = self._analyze_byte_frequencies(data)
            if byte_analysis:
                results.extend(byte_analysis)
                
        except Exception as e:
            self.logger.error(f"Entropy analysis failed: {e}")
        
        return results
    
    def _calculate_shannon_entropy(self, data: bytes) -> float:
        """Calculate Shannon entropy of byte data"""
        if len(data) == 0:
            return 0.0
        
        # Count byte frequencies
        byte_counts = [0] * 256
        for byte in data:
            byte_counts[byte] += 1
        
        # Calculate entropy
        entropy = 0.0
        length = len(data)
        
        for count in byte_counts:
            if count > 0:
                probability = count / length
                entropy -= probability * math.log2(probability)
        
        return entropy
    
    def _analyze_byte_frequencies(self, data: bytes) -> List[Dict[str, Any]]:
        """Analyze byte frequency distribution"""
        results = []
        
        if len(data) == 0:
            return results
        
        # Count byte frequencies
        byte_counts = Counter(data)
        
        # Calculate statistics
        frequencies = list(byte_counts.values())
        freq_mean = np.mean(frequencies)
        freq_std = np.std(frequencies)
        
        # Expected frequency for uniform distribution
        expected_freq = len(data) / 256
        
        # Chi-square test for uniformity
        chi_square = sum((freq - expected_freq) ** 2 / expected_freq for freq in frequencies)
        
        # Analyze distribution characteristics
        unique_bytes = len(byte_counts)
        
        if unique_bytes < 50:  # Very few unique bytes
            results.append({
                "type": "limited_byte_alphabet",
                "method": "byte_frequency_analysis",
                "tool_name": "crypto_analysis",
                "confidence": 0.7,
                "details": f"Limited byte alphabet: only {unique_bytes} unique values",
                "unique_bytes": unique_bytes,
                "interpretation": "Possible encoded/obfuscated data",
                "file_path": str(self.current_file_path) if hasattr(self, 'current_file_path') else "unknown"
            })
        
        elif chi_square < 100:  # Very uniform distribution
            results.append({
                "type": "uniform_byte_distribution",
                "method": "byte_frequency_analysis",
                "tool_name": "crypto_analysis",
                "confidence": 0.8,
                "details": f"Highly uniform byte distribution (χ²: {chi_square:.2f})",
                "chi_square": float(chi_square),
                "unique_bytes": unique_bytes,
                "interpretation": "Likely encrypted/compressed data",
                "file_path": str(self.current_file_path) if hasattr(self, 'current_file_path') else "unknown"
            })
        
        elif chi_square > 5000:  # Very non-uniform distribution
            results.append({
                "type": "skewed_byte_distribution",
                "method": "byte_frequency_analysis",
                "tool_name": "crypto_analysis",
                "confidence": 0.6,
                "details": f"Highly skewed byte distribution (χ²: {chi_square:.2f})",
                "chi_square": float(chi_square),
                "interpretation": "Possible structured/encoded data",
                "file_path": str(self.current_file_path) if hasattr(self, 'current_file_path') else "unknown"
            })
        
        return results
    
    def _pattern_detection(self, file_path: Path) -> List[Dict[str, Any]]:
        """Detect cryptographic patterns in file content"""
        results = []
        
        try:
            # Read file content
            with open(file_path, 'rb') as f:
                binary_data = f.read()
            
            # Try to decode as text for pattern matching
            try:
                text_data = binary_data.decode('utf-8', errors='ignore')
            except:
                text_data = binary_data.decode('latin-1', errors='ignore')
            
            # Check each pattern type
            for pattern_name, pattern_info in self.crypto_patterns.items():
                matches = self._find_pattern_matches(text_data, pattern_info, pattern_name)
                if matches:
                    results.extend(matches)
            
            # Binary pattern analysis
            binary_patterns = self._analyze_binary_patterns(binary_data)
            if binary_patterns:
                results.extend(binary_patterns)
            
            # Custom pattern analysis
            custom_patterns = self._analyze_custom_patterns(binary_data, text_data)
            if custom_patterns:
                results.extend(custom_patterns)
                
        except Exception as e:
            self.logger.error(f"Pattern detection failed: {e}")
        
        return results
    
    def _find_pattern_matches(self, text_data: str, pattern_info: Dict, pattern_name: str) -> List[Dict[str, Any]]:
        """Find matches for a specific pattern"""
        results = []
        
        pattern = pattern_info['pattern']
        min_length = pattern_info['min_length']
        description = pattern_info['description']
        
        matches = pattern.findall(text_data)
        
        # Filter matches by minimum length
        valid_matches = [match for match in matches if len(match) >= min_length]
        
        if valid_matches:
            # Calculate confidence based on number and quality of matches
            confidence = min(len(valid_matches) * 0.1, 0.8)
            
            # Increase confidence for certain pattern types
            if pattern_name in ['pem', 'bitcoin']:
                confidence = min(confidence + 0.2, 0.9)
            
            results.append({
                "type": f"crypto_pattern_{pattern_name}",
                "method": "pattern_detection",
                "tool_name": "crypto_analysis",
                "confidence": confidence,
                "details": f"Found {len(valid_matches)} {description} patterns",
                "pattern_type": pattern_name,
                "pattern_description": description,
                "match_count": len(valid_matches),
                "samples": valid_matches[:5],  # First 5 matches
                "file_path": str(self.current_file_path) if hasattr(self, 'current_file_path') else "unknown"
            })
        
        return results
    
    def _analyze_binary_patterns(self, data: bytes) -> List[Dict[str, Any]]:
        """Analyze binary patterns that might indicate cryptographic content"""
        results = []
        
        if len(data) < 16:
            return results
        
        # Look for repeating byte sequences
        repeating_patterns = self._find_repeating_sequences(data)
        if repeating_patterns:
            results.extend(repeating_patterns)
        
        # Look for block boundaries (common in block ciphers)
        block_patterns = self._detect_block_boundaries(data)
        if block_patterns:
            results.extend(block_patterns)
        
        # Look for padding patterns
        padding_patterns = self._detect_padding_patterns(data)
        if padding_patterns:
            results.extend(padding_patterns)
        
        return results
    
    def _find_repeating_sequences(self, data: bytes) -> List[Dict[str, Any]]:
        """Find repeating byte sequences"""
        results = []
        
        # Check for 4-byte and 8-byte repeating patterns
        for seq_len in [4, 8, 16]:
            if len(data) < seq_len * 3:  # Need at least 3 repetitions
                continue
            
            pattern_counts = defaultdict(int)
            
            for i in range(0, len(data) - seq_len + 1, seq_len):
                pattern = data[i:i + seq_len]
                pattern_counts[pattern] += 1
            
            # Find patterns that repeat significantly
            frequent_patterns = {pattern: count for pattern, count in pattern_counts.items() 
                               if count >= 3 and pattern != b'\x00' * seq_len}
            
            if frequent_patterns:
                results.append({
                    "type": "repeating_sequences",
                    "method": "binary_pattern_analysis",
                    "tool_name": "crypto_analysis",
                    "confidence": 0.6,
                    "details": f"Found {len(frequent_patterns)} repeating {seq_len}-byte sequences",
                    "sequence_length": seq_len,
                    "pattern_count": len(frequent_patterns),
                    "top_patterns": [{"pattern": pattern.hex(), "count": count} 
                                   for pattern, count in sorted(frequent_patterns.items(), 
                                                              key=lambda x: x[1], reverse=True)[:3]],
                    "interpretation": "Possible encrypted blocks or structured data",
                    "file_path": str(self.current_file_path) if hasattr(self, 'current_file_path') else "unknown"
                })
        
        return results
    
    def _detect_block_boundaries(self, data: bytes) -> List[Dict[str, Any]]:
        """Detect potential block cipher boundaries"""
        results = []
        
        # Common block sizes
        block_sizes = [8, 16, 32, 64]  # bytes
        
        for block_size in block_sizes:
            if len(data) % block_size == 0 and len(data) >= block_size * 4:
                # File size is multiple of block size
                
                # Analyze boundaries between blocks
                boundary_analysis = self._analyze_block_boundaries(data, block_size)
                
                if boundary_analysis['significant_boundaries'] > 0:
                    confidence = min(boundary_analysis['significant_boundaries'] * 0.1, 0.7)
                    
                    results.append({
                        "type": "block_cipher_indication",
                        "method": "block_boundary_analysis",
                        "tool_name": "crypto_analysis",
                        "confidence": confidence,
                        "details": f"File size multiple of {block_size} bytes with boundary patterns",
                        "block_size": block_size,
                        "total_blocks": len(data) // block_size,
                        "significant_boundaries": boundary_analysis['significant_boundaries'],
                        "interpretation": f"Possible {block_size * 8}-bit block cipher",
                        "file_path": str(self.current_file_path) if hasattr(self, 'current_file_path') else "unknown"
                    })
        
        return results
    
    def _analyze_block_boundaries(self, data: bytes, block_size: int) -> Dict[str, int]:
        """Analyze characteristics at block boundaries"""
        significant_boundaries = 0
        
        for i in range(block_size, len(data), block_size):
            if i >= len(data):
                break
            
            # Check for patterns at block boundaries
            prev_block_end = data[i - 4:i]
            curr_block_start = data[i:i + 4]
            
            # Look for patterns that suggest block cipher usage
            # (This is a simplified heuristic)
            if prev_block_end != curr_block_start:  # Blocks are different
                # Calculate byte-wise XOR to detect patterns
                xor_result = bytes(a ^ b for a, b in zip(prev_block_end, curr_block_start))
                
                # If XOR result has low entropy, might indicate cipher patterns
                if self._calculate_shannon_entropy(xor_result) < 2.0:
                    significant_boundaries += 1
        
        return {"significant_boundaries": significant_boundaries}
    
    def _detect_padding_patterns(self, data: bytes) -> List[Dict[str, Any]]:
        """Detect padding patterns common in cryptographic systems"""
        results = []
        
        if len(data) < 16:
            return results
        
        # Check end of file for PKCS padding
        last_16_bytes = data[-16:]
        
        # PKCS#7 padding check
        last_byte = last_16_bytes[-1]
        if 1 <= last_byte <= 16:
            # Check if the last N bytes are all the same value N
            padding_bytes = last_16_bytes[-last_byte:]
            if len(padding_bytes) == last_byte and all(b == last_byte for b in padding_bytes):
                results.append({
                    "type": "pkcs7_padding",
                    "method": "padding_detection",
                    "tool_name": "crypto_analysis",
                    "confidence": 0.8,
                    "details": f"PKCS#7 padding detected: {last_byte} bytes",
                    "padding_length": last_byte,
                    "interpretation": "Block cipher with PKCS#7 padding",
                    "file_path": str(self.current_file_path) if hasattr(self, 'current_file_path') else "unknown"
                })
        
        # Check for null padding
        trailing_nulls = 0
        for i in range(len(data) - 1, -1, -1):
            if data[i] == 0:
                trailing_nulls += 1
            else:
                break
        
        if trailing_nulls > 8:  # Significant null padding
            results.append({
                "type": "null_padding",
                "method": "padding_detection",
                "tool_name": "crypto_analysis",
                "confidence": 0.6,
                "details": f"Null padding detected: {trailing_nulls} bytes",
                "padding_length": trailing_nulls,
                "interpretation": "Possible block cipher with null padding",
                "file_path": str(self.current_file_path) if hasattr(self, 'current_file_path') else "unknown"
            })
        
        return results
    
    def _analyze_custom_patterns(self, binary_data: bytes, text_data: str) -> List[Dict[str, Any]]:
        """Analyze custom cryptographic patterns from configuration"""
        results = []
        
        if not self.config.custom_patterns:
            return results
        
        for pattern_str in self.config.custom_patterns:
            try:
                pattern = re.compile(pattern_str)
                matches = pattern.findall(text_data)
                
                if matches:
                    results.append({
                        "type": "custom_crypto_pattern",
                        "method": "custom_pattern_detection",
                        "tool_name": "crypto_analysis",
                        "confidence": 0.7,
                        "details": f"Custom pattern matched: {pattern_str}",
                        "pattern": pattern_str,
                        "match_count": len(matches),
                        "samples": matches[:3],
                        "file_path": str(self.current_file_path) if hasattr(self, 'current_file_path') else "unknown"
                    })
                    
            except re.error as e:
                self.logger.warning(f"Invalid custom pattern {pattern_str}: {e}")
        
        return results
    
    def _frequency_analysis(self, file_path: Path) -> List[Dict[str, Any]]:
        """Comprehensive frequency analysis for cipher detection"""
        results = []
        
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
            
            if len(data) == 0:
                return results
            
            # Byte frequency analysis
            byte_freq_analysis = self._analyze_byte_frequency_patterns(data)
            if byte_freq_analysis:
                results.extend(byte_freq_analysis)
            
            # Text frequency analysis (if data contains text)
            try:
                text_data = data.decode('utf-8', errors='ignore')
                if len(text_data) > 100 and self._is_likely_text(text_data):
                    text_freq_analysis = self._analyze_text_frequencies(text_data)
                    if text_freq_analysis:
                        results.extend(text_freq_analysis)
            except:
                pass
            
            # N-gram analysis
            ngram_analysis = self._analyze_ngrams(data)
            if ngram_analysis:
                results.extend(ngram_analysis)
                
        except Exception as e:
            self.logger.error(f"Frequency analysis failed: {e}")
        
        return results
    
    def _analyze_byte_frequency_patterns(self, data: bytes) -> List[Dict[str, Any]]:
        """Analyze byte frequency patterns"""
        results = []
        
        byte_counts = Counter(data)
        
        # Calculate frequency distribution metrics
        frequencies = list(byte_counts.values())
        total_bytes = len(data)
        
        # Index of Coincidence (IC)
        ic = sum(f * (f - 1) for f in frequencies) / (total_bytes * (total_bytes - 1))
        
        # Expected IC for random data is ~1/256 ≈ 0.0039
        # Expected IC for English text is ~0.067
        
        if ic > 0.05:  # High IC suggests non-random structure
            results.append({
                "type": "high_index_of_coincidence",
                "method": "frequency_analysis",
                "tool_name": "crypto_analysis",
                "confidence": min(ic * 10, 0.8),
                "details": f"High index of coincidence: {ic:.4f}",
                "index_of_coincidence": float(ic),
                "interpretation": "Likely plaintext or simple substitution cipher",
                "file_path": str(self.current_file_path) if hasattr(self, 'current_file_path') else "unknown"
            })
        
        elif ic < 0.001:  # Very low IC suggests encryption/compression
            results.append({
                "type": "low_index_of_coincidence",
                "method": "frequency_analysis",
                "tool_name": "crypto_analysis",
                "confidence": 0.7,
                "details": f"Very low index of coincidence: {ic:.4f}",
                "index_of_coincidence": float(ic),
                "interpretation": "Likely encrypted or compressed data",
                "file_path": str(self.current_file_path) if hasattr(self, 'current_file_path') else "unknown"
            })
        
        return results
    
    def _is_likely_text(self, text_data: str) -> bool:
        """Determine if data is likely to be text"""
        # Check for reasonable proportion of printable characters
        printable_chars = sum(1 for c in text_data if c.isprintable())
        printable_ratio = printable_chars / len(text_data)
        
        return printable_ratio > 0.7
    
    def _analyze_text_frequencies(self, text_data: str) -> List[Dict[str, Any]]:
        """Analyze character frequencies in text data"""
        results = []
        
        # Convert to uppercase for analysis
        text_upper = text_data.upper()
        
        # Count letter frequencies
        letter_counts = Counter(c for c in text_upper if c.isalpha())
        total_letters = sum(letter_counts.values())
        
        if total_letters < 100:  # Need sufficient text for analysis
            return results
        
        # Calculate letter frequencies as percentages
        observed_frequencies = {}
        for letter in string.ascii_uppercase:
            count = letter_counts.get(letter, 0)
            observed_frequencies[letter] = (count / total_letters) * 100
        
        # Calculate chi-square statistic against English frequencies
        chi_square = 0
        for letter in string.ascii_uppercase:
            observed = observed_frequencies[letter]
            expected = self.english_freq.get(letter, 0)
            if expected > 0:
                chi_square += ((observed - expected) ** 2) / expected
        
        # Analyze frequency patterns
        if chi_square > 100:  # High chi-square suggests non-English or cipher
            results.append({
                "type": "unusual_letter_frequencies",
                "method": "text_frequency_analysis",
                "tool_name": "crypto_analysis",
                "confidence": min(chi_square / 500, 0.8),
                "details": f"Letter frequencies differ from English (χ²: {chi_square:.2f})",
                "chi_square": float(chi_square),
                "letter_frequencies": observed_frequencies,
                "interpretation": "Possible substitution cipher or non-English text",
                "file_path": str(self.current_file_path) if hasattr(self, 'current_file_path') else "unknown"
            })
        
        # Check for flat frequency distribution (possible cipher)
        freq_values = list(observed_frequencies.values())
        freq_std = np.std(freq_values)
        
        if freq_std < 1.0:  # Very uniform distribution
            results.append({
                "type": "flat_letter_distribution",
                "method": "text_frequency_analysis",
                "tool_name": "crypto_analysis",
                "confidence": 0.7,
                "details": f"Unusually flat letter distribution (std: {freq_std:.3f})",
                "frequency_std": float(freq_std),
                "interpretation": "Possible polyalphabetic or transposition cipher",
                "file_path": str(self.current_file_path) if hasattr(self, 'current_file_path') else "unknown"
            })
        
        return results
    
    def _analyze_ngrams(self, data: bytes) -> List[Dict[str, Any]]:
        """Analyze n-gram patterns in data"""
        results = []
        
        # Analyze 2-gram and 3-gram frequencies
        for n in [2, 3]:
            if len(data) < n * 10:  # Need sufficient data
                continue
            
            ngram_counts = defaultdict(int)
            total_ngrams = 0
            
            for i in range(len(data) - n + 1):
                ngram = data[i:i + n]
                ngram_counts[ngram] += 1
                total_ngrams += 1
            
            if total_ngrams == 0:
                continue
            
            # Calculate n-gram statistics
            frequencies = list(ngram_counts.values())
            
            # Most common n-grams
            most_common = sorted(ngram_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # Check for highly repetitive n-grams
            if most_common and most_common[0][1] > total_ngrams * 0.05:  # Top n-gram > 5%
                results.append({
                    "type": f"repetitive_{n}grams",
                    "method": "ngram_analysis",
                    "tool_name": "crypto_analysis",
                    "confidence": 0.6,
                    "details": f"Highly repetitive {n}-grams detected",
                    "ngram_size": n,
                    "top_ngram_frequency": float(most_common[0][1] / total_ngrams),
                    "top_ngrams": [{"ngram": ngram.hex(), "count": count, "frequency": count / total_ngrams} 
                                 for ngram, count in most_common[:5]],
                    "interpretation": "Possible structured or encrypted data with patterns",
                    "file_path": str(self.current_file_path) if hasattr(self, 'current_file_path') else "unknown"
                })
        
        return results
    
    def _key_search(self, file_path: Path) -> List[Dict[str, Any]]:
        """Search for potential cryptographic keys or passwords"""
        results = []
        
        try:
            # Basic key search in file content
            with open(file_path, 'rb') as f:
                data = f.read()
            
            # Try to decode as text
            try:
                text_data = data.decode('utf-8', errors='ignore')
            except:
                text_data = data.decode('latin-1', errors='ignore')
            
            # Search for key-like patterns
            key_patterns = self._search_key_patterns(text_data)
            if key_patterns:
                results.extend(key_patterns)
            
            # Dictionary-based key search
            if self.config.key_search_enabled:
                dict_search = self._dictionary_key_search(data, file_path)
                if dict_search:
                    results.extend(dict_search)
                    
        except Exception as e:
            self.logger.error(f"Key search failed: {e}")
        
        return results
    
    def _search_key_patterns(self, text_data: str) -> List[Dict[str, Any]]:
        """Search for patterns that might be cryptographic keys"""
        results = []
        
        # Key-related keywords
        key_keywords = [
            r'key\s*[:=]\s*([A-Za-z0-9+/=]{10,})',
            r'password\s*[:=]\s*([A-Za-z0-9!@#$%^&*()_+-=]{4,})',
            r'secret\s*[:=]\s*([A-Za-z0-9+/=]{10,})',
            r'token\s*[:=]\s*([A-Za-z0-9+/=]{10,})',
            r'api[_-]?key\s*[:=]\s*([A-Za-z0-9+/=]{10,})',
        ]
        
        for pattern_str in key_keywords:
            pattern = re.compile(pattern_str, re.IGNORECASE)
            matches = pattern.findall(text_data)
            
            if matches:
                # Analyze potential keys
                for match in matches[:5]:  # Limit to first 5 matches
                    key_analysis = self._analyze_potential_key(match)
                    
                    results.append({
                        "type": "potential_cryptographic_key",
                        "method": "key_pattern_search",
                        "tool_name": "crypto_analysis",
                        "confidence": key_analysis['confidence'],
                        "details": f"Potential key found: {match[:20]}...",
                        "key_preview": match[:20],
                        "key_length": len(match),
                        "key_analysis": key_analysis,
                        "file_path": str(self.current_file_path) if hasattr(self, 'current_file_path') else "unknown"
                    })
        
        return results
    
    def _analyze_potential_key(self, key_candidate: str) -> Dict[str, Any]:
        """Analyze a potential cryptographic key"""
        analysis = {
            'length': len(key_candidate),
            'character_set': 'unknown',
            'entropy': 0.0,
            'confidence': 0.3  # Base confidence
        }
        
        # Determine character set
        if re.match(r'^[A-Za-z0-9+/=]+$', key_candidate):
            analysis['character_set'] = 'base64'
            analysis['confidence'] += 0.3
        elif re.match(r'^[0-9A-Fa-f]+$', key_candidate):
            analysis['character_set'] = 'hexadecimal'
            analysis['confidence'] += 0.2
        elif re.match(r'^[A-Za-z0-9]+$', key_candidate):
            analysis['character_set'] = 'alphanumeric'
            analysis['confidence'] += 0.1
        
        # Calculate entropy
        if len(key_candidate) > 0:
            char_counts = Counter(key_candidate)
            char_probs = [count / len(key_candidate) for count in char_counts.values()]
            entropy = -sum(p * math.log2(p) for p in char_probs)
            analysis['entropy'] = entropy
            
            # Higher entropy generally indicates better key quality
            if entropy > 4.0:
                analysis['confidence'] += 0.2
        
        # Length-based confidence
        if 16 <= len(key_candidate) <= 64:  # Reasonable key length
            analysis['confidence'] += 0.2
        elif len(key_candidate) >= 32:  # Long key
            analysis['confidence'] += 0.1
        
        analysis['confidence'] = min(analysis['confidence'], 0.9)
        
        return analysis
    
    def _dictionary_key_search(self, data: bytes, file_path: Path) -> List[Dict[str, Any]]:
        """Perform dictionary-based key search using external tools"""
        results = []
        
        # This would use tools like hashcat or john for key cracking
        # Implementation depends on specific tools and file formats
        
        return results
    
    def _cipher_detection(self, file_path: Path) -> List[Dict[str, Any]]:
        """Detect specific cipher types based on patterns and characteristics"""
        results = []
        
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
            
            # Caesar cipher detection
            caesar_results = self._detect_caesar_cipher(data)
            if caesar_results:
                results.extend(caesar_results)
            
            # XOR cipher detection
            xor_results = self._detect_simple_xor(data)
            if xor_results:
                results.extend(xor_results)
            
            # Base64 detection and analysis
            base64_results = self._detect_base64_content(data)
            if base64_results:
                results.extend(base64_results)
                
        except Exception as e:
            self.logger.error(f"Cipher detection failed: {e}")
        
        return results
    
    def _detect_caesar_cipher(self, data: bytes) -> List[Dict[str, Any]]:
        """Detect Caesar cipher by trying all possible shifts"""
        results = []
        
        try:
            text_data = data.decode('utf-8', errors='ignore')
        except:
            return results
        
        # Only analyze if data looks like it could be text
        if not self._is_likely_text(text_data):
            return results
        
        # Try all possible Caesar shifts
        best_shift = 0
        best_score = 0
        
        for shift in range(1, 26):
            shifted_text = self._caesar_shift(text_data, shift)
            score = self._calculate_english_score(shifted_text)
            
            if score > best_score:
                best_score = score
                best_shift = shift
        
        # If we found a significantly better score, it might be Caesar cipher
        if best_score > 0.5:  # Threshold for English-like text
            shifted_text = self._caesar_shift(text_data, best_shift)
            
            results.append({
                "type": "possible_caesar_cipher",
                "method": "caesar_cipher_detection",
                "tool_name": "crypto_analysis",
                "confidence": min(best_score, 0.8),
                "details": f"Possible Caesar cipher with shift {best_shift}",
                "shift": best_shift,
                "english_score": float(best_score),
                "decrypted_preview": shifted_text[:100],
                "file_path": str(self.current_file_path) if hasattr(self, 'current_file_path') else "unknown"
            })
        
        return results
    
    def _caesar_shift(self, text: str, shift: int) -> str:
        """Apply Caesar cipher shift to text"""
        result = []
        for char in text:
            if char.isalpha():
                base = ord('A') if char.isupper() else ord('a')
                shifted = (ord(char) - base + shift) % 26 + base
                result.append(chr(shifted))
            else:
                result.append(char)
        return ''.join(result)
    
    def _calculate_english_score(self, text: str) -> float:
        """Calculate how English-like a text is"""
        # Simple scoring based on common English letter frequencies
        text_upper = text.upper()
        letter_counts = Counter(c for c in text_upper if c.isalpha())
        total_letters = sum(letter_counts.values())
        
        if total_letters == 0:
            return 0.0
        
        score = 0.0
        for letter, expected_freq in self.english_freq.items():
            observed_count = letter_counts.get(letter, 0)
            observed_freq = (observed_count / total_letters) * 100
            
            # Score based on how close to expected frequency
            diff = abs(observed_freq - expected_freq)
            score += max(0, expected_freq - diff) / expected_freq
        
        return score / len(self.english_freq)
    
    def _detect_simple_xor(self, data: bytes) -> List[Dict[str, Any]]:
        """Detect simple XOR encryption"""
        results = []
        
        if len(data) < 100:  # Need sufficient data
            return results
        
        # Try single-byte XOR keys
        for key in range(1, 256):
            xored_data = bytes(b ^ key for b in data)
            
            try:
                # Try to decode as text
                decoded_text = xored_data.decode('utf-8', errors='ignore')
                
                if self._is_likely_text(decoded_text):
                    english_score = self._calculate_english_score(decoded_text)
                    
                    if english_score > 0.3:  # Threshold for possible English
                        results.append({
                            "type": "possible_xor_cipher",
                            "method": "xor_cipher_detection",
                            "tool_name": "crypto_analysis",
                            "confidence": min(english_score * 1.5, 0.8),
                            "details": f"Possible single-byte XOR with key 0x{key:02x}",
                            "xor_key": f"0x{key:02x}",
                            "key_decimal": key,
                            "english_score": float(english_score),
                            "decrypted_preview": decoded_text[:100],
                            "file_path": str(self.current_file_path) if hasattr(self, 'current_file_path') else "unknown"
                        })
                        
                        # Only report the best few matches
                        if len(results) >= 3:
                            break
                            
            except:
                continue
        
        return results
    
    def _detect_base64_content(self, data: bytes) -> List[Dict[str, Any]]:
        """Detect and analyze Base64 encoded content"""
        results = []
        
        try:
            text_data = data.decode('utf-8', errors='ignore')
        except:
            return results
        
        # Find Base64-like strings
        base64_pattern = re.compile(r'[A-Za-z0-9+/]{20,}={0,2}')
        matches = base64_pattern.findall(text_data)
        
        valid_base64 = []
        
        for match in matches:
            try:
                # Try to decode as Base64
                decoded = base64.b64decode(match, validate=True)
                
                # Analyze decoded content
                if len(decoded) > 10:
                    analysis = {
                        'original': match[:50] + '...' if len(match) > 50 else match,
                        'decoded_length': len(decoded),
                        'decoded_entropy': self._calculate_shannon_entropy(decoded)
                    }
                    
                    # Try to determine what the decoded content might be
                    try:
                        decoded_text = decoded.decode('utf-8', errors='ignore')
                        if self._is_likely_text(decoded_text):
                            analysis['content_type'] = 'text'
                            analysis['decoded_preview'] = decoded_text[:100]
                        else:
                            analysis['content_type'] = 'binary'
                            analysis['decoded_hex'] = decoded[:32].hex()
                    except:
                        analysis['content_type'] = 'binary'
                        analysis['decoded_hex'] = decoded[:32].hex()
                    
                    valid_base64.append(analysis)
                    
            except Exception:
                continue
        
        if valid_base64:
            results.append({
                "type": "base64_encoded_content",
                "method": "base64_detection",
                "tool_name": "crypto_analysis",
                "confidence": 0.8,
                "details": f"Found {len(valid_base64)} valid Base64 encoded strings",
                "base64_count": len(valid_base64),
                "base64_strings": valid_base64[:5],  # First 5 strings
                "file_path": str(self.current_file_path) if hasattr(self, 'current_file_path') else "unknown"
            })
        
        return results
    
    def _hash_analysis(self, file_path: Path) -> List[Dict[str, Any]]:
        """Analyze potential hash values in the file"""
        results = []
        
        # This would analyze hash patterns and potentially use tools
        # like hash-identifier or online hash databases
        
        return results
    
    def _xor_analysis(self, file_path: Path) -> List[Dict[str, Any]]:
        """Advanced XOR analysis including multi-byte keys"""
        results = []
        
        # This would implement more sophisticated XOR analysis
        # including multi-byte keys and key length detection
        
        return results
    
    def _substitution_analysis(self, file_path: Path) -> List[Dict[str, Any]]:
        """Analyze for substitution ciphers"""
        results = []
        
        # This would implement substitution cipher detection
        # using frequency analysis and pattern matching
        
        return results
    
    def _block_cipher_analysis(self, file_path: Path) -> List[Dict[str, Any]]:
        """Analyze for block cipher patterns"""
        results = []
        
        # This would implement block cipher analysis
        # looking for ECB patterns, CBC patterns, etc.
        
        return results
    
    def _stream_cipher_analysis(self, file_path: Path) -> List[Dict[str, Any]]:
        """Analyze for stream cipher patterns"""
        results = []
        
        # This would implement stream cipher analysis
        # looking for keystream patterns, period detection, etc.
        
        return results
    
    def _compression_detection(self, file_path: Path) -> List[Dict[str, Any]]:
        """Detect if data is compressed vs encrypted"""
        results = []
        
        # This would implement compression detection
        # using entropy analysis and known compression signatures
        
        return results
    
    def _randomness_tests(self, file_path: Path) -> List[Dict[str, Any]]:
        """Perform statistical randomness tests"""
        results = []
        
        # This would implement various randomness tests
        # like runs test, serial test, etc.
        
        return results
    
    def _correlation_analysis(self, file_path: Path) -> List[Dict[str, Any]]:
        """Perform correlation analysis on data"""
        results = []
        
        # This would implement correlation analysis
        # to detect patterns and relationships in data
        
        return results
    
    def _autocorrelation_analysis(self, file_path: Path) -> List[Dict[str, Any]]:
        """Perform autocorrelation analysis"""
        results = []
        
        # This would implement autocorrelation analysis
        # to detect periodic patterns in data
        
        return results
    
    def _bit_distribution_analysis(self, file_path: Path) -> List[Dict[str, Any]]:
        """Analyze bit distribution patterns"""
        results = []
        
        # This would implement bit-level analysis
        # looking at bit patterns and distributions
        
        return results
    
    def _ml_pattern_detection(self, file_path: Path) -> List[Dict[str, Any]]:
        """Use machine learning for pattern detection"""
        results = []
        
        if not SKLEARN_AVAILABLE:
            return results
        
        # This would implement ML-based pattern detection
        # using clustering, classification, etc.
        
        return results
    
    def cleanup(self):
        """Cleanup resources"""
        if self.temp_dir.exists():
            import shutil
            shutil.rmtree(self.temp_dir)
        
        self.executor.shutdown(wait=True)