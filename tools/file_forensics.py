#!/usr/bin/env python3
"""
File Forensics Tools - Complete Working Version
"""

import os
import sys
import subprocess
import hashlib
import logging
import magic
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import json
import time
import math

class FileForensicsTools:
    """File forensics and analysis tools"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize magic
        try:
            self.magic_mime = magic.Magic(mime=True)
            self.magic_description = magic.Magic()
        except Exception as e:
            self.logger.warning(f"Magic library initialization failed: {e}")
            self.magic_mime = None
            self.magic_description = None
    
    def magic_analysis(self, file_path: Path) -> List[Dict[str, Any]]:
        """Perform magic file type analysis"""
        results = []
        
        try:
            if not file_path.exists():
                return [{"type": "error", "error": "File not found", "confidence": 0.0}]
            
            # Get MIME type
            mime_type = "unknown"
            description = "unknown"
            
            if self.magic_mime and self.magic_description:
                try:
                    mime_type = self.magic_mime.from_file(str(file_path))
                    description = self.magic_description.from_file(str(file_path))
                except:
                    pass
            
            # Fallback to file command
            if mime_type == "unknown":
                try:
                    result = subprocess.run(['file', '-b', '--mime-type', str(file_path)], 
                                          capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        mime_type = result.stdout.strip()
                except:
                    pass
            
            # Get file size
            file_size = file_path.stat().st_size
            
            # Calculate file hash
            file_hash = self._calculate_file_hash(file_path)
            
            results.append({
                "type": "magic_analysis",
                "method": "magic_analysis",
                "mime_type": mime_type,
                "description": description,
                "file_size": file_size,
                "file_hash": file_hash,
                "confidence": 0.9,
                "details": f"File identified as {mime_type}: {description}"
            })
            
        except Exception as e:
            self.logger.error(f"Magic analysis failed: {e}")
            results.append({
                "type": "magic_analysis",
                "method": "magic_analysis", 
                "error": str(e),
                "confidence": 0.0
            })
        
        return results
    
    def basic_analysis(self, file_path: Path) -> List[Dict[str, Any]]:
        """Basic file analysis - alias for magic_analysis"""
        return self.magic_analysis(file_path)
    
    def entropy_analysis(self, file_path: Path) -> List[Dict[str, Any]]:
        """Calculate file entropy"""
        results = []
        
        try:
            if not file_path.exists():
                return [{"type": "error", "error": "File not found", "confidence": 0.0}]
            
            # Read file data (limit to first 1MB for large files)
            max_bytes = 1024 * 1024  # 1MB
            with open(file_path, 'rb') as f:
                data = f.read(max_bytes)
            
            if not data:
                return [{"type": "entropy_analysis", "entropy": 0.0, "confidence": 0.5}]
            
            # Calculate entropy
            entropy = self._calculate_entropy(data)
            
            # Determine significance
            confidence = 0.5
            analysis = "Normal entropy"
            
            if entropy > 7.5:
                confidence = 0.8
                analysis = "High entropy - possibly encrypted or compressed"
            elif entropy < 1.0:
                confidence = 0.7
                analysis = "Very low entropy - highly structured data"
            elif entropy > 6.0:
                confidence = 0.6
                analysis = "Moderately high entropy - some randomness"
            
            results.append({
                "type": "entropy_analysis",
                "method": "entropy_analysis",
                "entropy": entropy,
                "analysis": analysis,
                "confidence": confidence,
                "details": f"File entropy: {entropy:.2f} bits per byte"
            })
            
        except Exception as e:
            self.logger.error(f"Entropy analysis failed: {e}")
            results.append({
                "type": "entropy_analysis",
                "method": "entropy_analysis",
                "error": str(e),
                "confidence": 0.0
            })
        
        return results
    
    def hex_analysis(self, file_path: Path) -> List[Dict[str, Any]]:
        """Analyze file hex patterns"""
        results = []
        
        try:
            if not file_path.exists():
                return [{"type": "error", "error": "File not found", "confidence": 0.0}]
            
            # Read first 512 bytes
            with open(file_path, 'rb') as f:
                header_data = f.read(512)
            
            if not header_data:
                return [{"type": "hex_analysis", "patterns": [], "confidence": 0.3}]
            
            # Check for common signatures
            signatures = {
                b'\x89PNG\r\n\x1a\n': 'PNG Image',
                b'\xff\xd8\xff': 'JPEG Image', 
                b'GIF8': 'GIF Image',
                b'BM': 'Bitmap Image',
                b'RIFF': 'RIFF Container',
                b'%PDF': 'PDF Document',
                b'PK\x03\x04': 'ZIP Archive',
                b'\x1f\x8b': 'GZIP Archive'
            }
            
            detected_signatures = []
            for sig, file_type in signatures.items():
                if header_data.startswith(sig):
                    detected_signatures.append({
                        "signature": sig.hex(),
                        "file_type": file_type,
                        "offset": 0
                    })
            
            # Look for embedded signatures
            for sig, file_type in signatures.items():
                offset = header_data.find(sig)
                if offset > 0:  # Found at non-zero offset
                    detected_signatures.append({
                        "signature": sig.hex(),
                        "file_type": file_type,
                        "offset": offset,
                        "embedded": True
                    })
            
            confidence = 0.7 if detected_signatures else 0.3
            
            results.append({
                "type": "hex_analysis",
                "method": "hex_analysis",
                "signatures": detected_signatures,
                "confidence": confidence,
                "details": f"Found {len(detected_signatures)} signature patterns"
            })
            
        except Exception as e:
            self.logger.error(f"Hex analysis failed: {e}")
            results.append({
                "type": "hex_analysis",
                "method": "hex_analysis",
                "error": str(e),
                "confidence": 0.0
            })
        
        return results
    
    def _calculate_entropy(self, data: bytes) -> float:
        """Calculate Shannon entropy of data"""
        if not data:
            return 0.0
        
        # Count frequency of each byte value
        frequency = [0] * 256
        for byte in data:
            frequency[byte] += 1
        
        # Calculate entropy
        entropy = 0.0
        length = len(data)
        
        for count in frequency:
            if count > 0:
                probability = count / length
                entropy -= probability * math.log2(probability)
        
        return entropy
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file"""
        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except:
            return "unknown"
    
    def execute_method(self, method: str, file_path: Path) -> List[Dict[str, Any]]:
        """Execute a specific analysis method"""
        
        method_map = {
            "magic_analysis": self.magic_analysis,
            "basic_analysis": self.basic_analysis,
            "entropy_analysis": self.entropy_analysis,
            "hex_analysis": self.hex_analysis,
        }
        
        if method not in method_map:
            return [{"type": "error", "error": f"Unknown method: {method}", "confidence": 0.0}]
        
        try:
            return method_map[method](file_path)
        except Exception as e:
            self.logger.error(f"Method {method} execution failed: {e}")
            return [{"type": "error", "error": str(e), "confidence": 0.0}]
    
    def get_supported_methods(self) -> List[str]:
        """Get list of supported analysis methods"""
        return ["magic_analysis", "basic_analysis", "entropy_analysis", "hex_analysis"]
    
    def get_tool_info(self) -> Dict[str, Any]:
        """Get tool information"""
        return {
            "name": "file_forensics",
            "version": "1.0",
            "methods": self.get_supported_methods(),
            "magic_available": self.magic_mime is not None,
            "description": "File forensics and analysis tools"
        }
