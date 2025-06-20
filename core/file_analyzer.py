"""
Core File Analyzer - Basic file analysis and type detection
"""

import magic
import hashlib
import mimetypes
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import subprocess
import tempfile
import os
import struct
import time
from PIL import Image
import cv2
import numpy as np

class FileAnalyzer:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize libmagic
        try:
            self.magic_mime = magic.Magic(mime=True)
            self.magic_desc = magic.Magic()
        except Exception as e:
            self.logger.warning(f"Failed to initialize libmagic: {e}")
            self.magic_mime = None
            self.magic_desc = None
    
    async def analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """Comprehensive file analysis"""
        start_time = time.time()
        
        try:
            file_info = {
                "path": str(file_path),
                "name": file_path.name,
                "size": file_path.stat().st_size,
                "modified": file_path.stat().st_mtime,
                "extension": file_path.suffix.lower(),
                "analysis_time": 0.0
            }
            
            # Basic file type detection
            file_info.update(self._detect_file_type(file_path))
            
            # Hash calculations
            file_info.update(self._calculate_hashes(file_path))
            
            # Header/footer analysis
            file_info.update(self._analyze_file_structure(file_path))
            
            # Entropy calculation
            file_info.update(self._calculate_entropy(file_path))
            
            # Metadata extraction
            file_info.update(self._extract_basic_metadata(file_path))
            
            # File format validation
            file_info.update(self._validate_file_format(file_path))
            
            file_info["analysis_time"] = time.time() - start_time
            
            self.logger.debug(f"File analysis completed for {file_path} in {file_info['analysis_time']:.2f}s")
            return file_info
            
        except Exception as e:
            self.logger.error(f"File analysis failed for {file_path}: {e}")
            return {
                "path": str(file_path),
                "error": str(e),
                "analysis_time": time.time() - start_time
            }
    
    def _detect_file_type(self, file_path: Path) -> Dict[str, Any]:
        """Detect file type using multiple methods"""
        result = {
            "type": "unknown",
            "mime_type": "application/octet-stream",
            "description": "Unknown file type",
            "magic_detected": False
        }
        
        try:
            # Use libmagic if available
            if self.magic_mime:
                mime_type = self.magic_mime.from_file(str(file_path))
                description = self.magic_desc.from_file(str(file_path))
                result.update({
                    "mime_type": mime_type,
                    "description": description,
                    "magic_detected": True
                })
            
            # Fallback to mimetypes module
            if not result["magic_detected"]:
                mime_type, _ = mimetypes.guess_type(str(file_path))
                if mime_type:
                    result["mime_type"] = mime_type
            
            # Extract primary type
            result["type"] = result["mime_type"].split('/')[0]
            
            # Additional file type categorization
            result.update(self._categorize_file_type(file_path, result["mime_type"]))
            
        except Exception as e:
            self.logger.debug(f"File type detection failed: {e}")
        
        return result
    
    def _categorize_file_type(self, file_path: Path, mime_type: str) -> Dict[str, Any]:
        """Categorize file for analysis purposes"""
        category_info = {
            "category": "generic",
            "steganography_potential": "low",
            "analysis_priority": 3
        }
        
        # Define categories based on steganography potential
        if mime_type.startswith("image/"):
            category_info.update({
                "category": "image",
                "steganography_potential": "high",
                "analysis_priority": 1,
                "supports_lsb": True,
                "supports_dct": mime_type in ["image/jpeg", "image/jpg"]
            })
        
        elif mime_type.startswith("audio/"):
            category_info.update({
                "category": "audio", 
                "steganography_potential": "high",
                "analysis_priority": 1,
                "supports_spectral": True,
                "supports_echo": True
            })
        
        elif mime_type == "application/pdf":
            category_info.update({
                "category": "document",
                "steganography_potential": "medium",
                "analysis_priority": 2,
                "supports_streams": True,
                "supports_metadata": True
            })
        
        elif mime_type.startswith("video/"):
            category_info.update({
                "category": "video",
                "steganography_potential": "high", 
                "analysis_priority": 2,
                "supports_frame_analysis": True
            })
        
        elif mime_type in ["application/zip", "application/x-rar", "application/x-7z-compressed"]:
            category_info.update({
                "category": "archive",
                "steganography_potential": "medium",
                "analysis_priority": 2,
                "supports_hidden_files": True
            })
        
        elif mime_type.startswith("text/"):
            category_info.update({
                "category": "text",
                "steganography_potential": "medium",
                "analysis_priority": 2,
                "supports_whitespace": True,
                "supports_unicode": True
            })
        
        return category_info
    
    def _calculate_hashes(self, file_path: Path) -> Dict[str, str]:
        """Calculate file hashes"""
        hashes = {}
        
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
            
            # Common hash algorithms
            algorithms = ['md5', 'sha1', 'sha256']
            
            for algo in algorithms:
                hasher = hashlib.new(algo)
                hasher.update(data)
                hashes[f"{algo}_hash"] = hasher.hexdigest()
            
            # Additional hashes for file identification
            hashes["ssdeep"] = self._calculate_ssdeep(data)
            
        except Exception as e:
            self.logger.debug(f"Hash calculation failed: {e}")
        
        return hashes
    
    def _calculate_ssdeep(self, data: bytes) -> Optional[str]:
        """Calculate fuzzy hash using ssdeep if available"""
        try:
            import ssdeep
            return ssdeep.hash(data)
        except ImportError:
            return None
    
    def _analyze_file_structure(self, file_path: Path) -> Dict[str, Any]:
        """Analyze file headers, footers, and structure"""
        structure = {
            "header_valid": False,
            "footer_valid": False,
            "structure_anomalies": []
        }
        
        try:
            with open(file_path, 'rb') as f:
                # Read header (first 1024 bytes)
                header = f.read(1024)
                
                # Read footer (last 1024 bytes)
                f.seek(-min(1024, file_path.stat().st_size), 2)
                footer = f.read()
            
            # Validate header/footer based on file type
            structure.update(self._validate_signatures(header, footer, file_path))
            
            # Look for embedded file signatures
            structure.update(self._detect_embedded_signatures(header + footer))
            
        except Exception as e:
            self.logger.debug(f"Structure analysis failed: {e}")
        
        return structure
    
    def _validate_signatures(self, header: bytes, footer: bytes, file_path: Path) -> Dict[str, Any]:
        """Validate file signatures"""
        validation = {
            "header_valid": False,
            "footer_valid": False,
            "signature_matches": []
        }
        
        # Common file signatures
        signatures = {
            "JPEG": (b'\xFF\xD8\xFF', b'\xFF\xD9'),
            "PNG": (b'\x89PNG\r\n\x1a\n', None),
            "GIF": (b'GIF8', b'\x00;'),
            "PDF": (b'%PDF', b'%%EOF'),
            "ZIP": (b'PK\x03\x04', b'PK\x05\x06'),
            "RAR": (b'Rar!\x1a\x07', None),
            "MP3": (b'ID3', None),
            "WAV": (b'RIFF', b'fmt '),
            "AVI": (b'RIFF', b'AVI '),
            "EXE": (b'MZ', None)
        }
        
        ext = file_path.suffix.upper().lstrip('.')
        
        for format_name, (header_sig, footer_sig) in signatures.items():
            header_match = header.startswith(header_sig) if header_sig else True
            footer_match = footer.endswith(footer_sig) if footer_sig else True
            
            if header_match and footer_match:
                validation["signature_matches"].append(format_name)
                
                # Check if signature matches extension
                if format_name.lower() == ext.lower() or (format_name == "JPEG" and ext in ["JPG", "JPEG"]):
                    validation["header_valid"] = True
                    validation["footer_valid"] = True
        
        return validation
    
    def _detect_embedded_signatures(self, data: bytes) -> Dict[str, Any]:
        """Detect embedded file signatures"""
        embedded = {
            "embedded_signatures": [],
            "signature_count": 0
        }
        
        # Look for common file signatures within the data
        signatures = [
            (b'\xFF\xD8\xFF', "JPEG"),
            (b'\x89PNG\r\n\x1a\n', "PNG"), 
            (b'GIF8', "GIF"),
            (b'%PDF', "PDF"),
            (b'PK\x03\x04', "ZIP"),
            (b'Rar!\x1a\x07', "RAR"),
            (b'RIFF', "RIFF")
        ]
        
        for sig, name in signatures:
            count = data.count(sig)
            if count > 1:  # More than one occurrence might indicate embedding
                embedded["embedded_signatures"].append({
                    "type": name,
                    "occurrences": count
                })
                embedded["signature_count"] += count
        
        return embedded
    
    def _calculate_entropy(self, file_path: Path, chunk_size: int = 8192) -> Dict[str, Any]:
        """Calculate file entropy"""
        entropy_info = {
            "entropy": 0.0,
            "entropy_analysis": "unknown",
            "chunk_entropies": []
        }
        
        try:
            with open(file_path, 'rb') as f:
                chunk_entropies = []
                
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    
                    chunk_entropy = self._calculate_chunk_entropy(chunk)
                    chunk_entropies.append(chunk_entropy)
                
                if chunk_entropies:
                    entropy_info["entropy"] = sum(chunk_entropies) / len(chunk_entropies)
                    entropy_info["chunk_entropies"] = chunk_entropies[:10]  # Store first 10 chunks
                    
                    # Analyze entropy level
                    if entropy_info["entropy"] > 7.5:
                        entropy_info["entropy_analysis"] = "high_entropy_compressed_or_encrypted"
                    elif entropy_info["entropy"] > 6.0:
                        entropy_info["entropy_analysis"] = "medium_entropy_normal"
                    else:
                        entropy_info["entropy_analysis"] = "low_entropy_structured"
        
        except Exception as e:
            self.logger.debug(f"Entropy calculation failed: {e}")
        
        return entropy_info
    
    def _calculate_chunk_entropy(self, data: bytes) -> float:
        """Calculate Shannon entropy for a data chunk"""
        if not data:
            return 0.0
        
        # Count byte frequencies
        byte_counts = [0] * 256
        for byte in data:
            byte_counts[byte] += 1
        
        # Calculate entropy
        entropy = 0.0
        data_len = len(data)
        
        for count in byte_counts:
            if count > 0:
                probability = count / data_len
                entropy -= probability * np.log2(probability)
        
        return entropy
    
    def _extract_basic_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract basic metadata using multiple tools"""
        metadata = {
            "has_metadata": False,
            "metadata_tools": [],
            "metadata_size": 0
        }
        
        # Try exiftool if available
        metadata.update(self._extract_exiftool_metadata(file_path))
        
        # Try PIL for images
        if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
            metadata.update(self._extract_pil_metadata(file_path))
        
        return metadata
    
    def _extract_exiftool_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata using exiftool"""
        result = {"exiftool_available": False}
        
        try:
            cmd = ['exiftool', '-json', '-a', str(file_path)]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if proc.returncode == 0:
                import json
                metadata = json.loads(proc.stdout)
                if metadata:
                    result.update({
                        "exiftool_available": True,
                        "has_metadata": True,
                        "metadata_fields": len(metadata[0]) if metadata else 0,
                        "metadata_tools": ["exiftool"]
                    })
        
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
            pass
        
        return result
    
    def _extract_pil_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata using PIL"""
        result = {"pil_metadata": False}
        
        try:
            with Image.open(file_path) as img:
                if hasattr(img, '_getexif') and img._getexif():
                    result.update({
                        "pil_metadata": True,
                        "has_metadata": True,
                        "metadata_tools": result.get("metadata_tools", []) + ["PIL"]
                    })
        
        except Exception:
            pass
        
        return result
    
    def _validate_file_format(self, file_path: Path) -> Dict[str, Any]:
        """Validate file format consistency"""
        validation = {
            "format_consistent": True,
            "format_issues": []
        }
        
        try:
            # Basic validation based on file type
            if file_path.suffix.lower() in ['.jpg', '.jpeg']:
                validation.update(self._validate_jpeg(file_path))
            elif file_path.suffix.lower() == '.png':
                validation.update(self._validate_png(file_path))
            elif file_path.suffix.lower() == '.pdf':
                validation.update(self._validate_pdf(file_path))
        
        except Exception as e:
            validation["format_issues"].append(str(e))
            validation["format_consistent"] = False
        
        return validation
    
    def _validate_jpeg(self, file_path: Path) -> Dict[str, Any]:
        """Validate JPEG file format"""
        validation = {"jpeg_valid": True}
        
        try:
            with open(file_path, 'rb') as f:
                # Check JPEG header
                header = f.read(4)
                if not header.startswith(b'\xFF\xD8\xFF'):
                    validation["jpeg_valid"] = False
                    validation["format_issues"] = ["Invalid JPEG header"]
                
                # Check for proper JPEG ending
                f.seek(-2, 2)
                footer = f.read(2)
                if footer != b'\xFF\xD9':
                    validation["jpeg_valid"] = False
                    validation.setdefault("format_issues", []).append("Invalid JPEG footer")
        
        except Exception as e:
            validation["jpeg_valid"] = False
            validation["format_issues"] = [str(e)]
        
        return validation
    
    def _validate_png(self, file_path: Path) -> Dict[str, Any]:
        """Validate PNG file format"""
        validation = {"png_valid": True}
        
        try:
            with open(file_path, 'rb') as f:
                # Check PNG header
                header = f.read(8)
                if header != b'\x89PNG\r\n\x1a\n':
                    validation["png_valid"] = False
                    validation["format_issues"] = ["Invalid PNG header"]
        
        except Exception as e:
            validation["png_valid"] = False
            validation["format_issues"] = [str(e)]
        
        return validation
    
    def _validate_pdf(self, file_path: Path) -> Dict[str, Any]:
        """Validate PDF file format"""
        validation = {"pdf_valid": True}
        
        try:
            with open(file_path, 'rb') as f:
                # Check PDF header
                header = f.read(8)
                if not header.startswith(b'%PDF-'):
                    validation["pdf_valid"] = False
                    validation["format_issues"] = ["Invalid PDF header"]
                
                # Check for EOF marker
                f.seek(-1024, 2)
                footer = f.read()
                if b'%%EOF' not in footer:
                    validation["pdf_valid"] = False
                    validation.setdefault("format_issues", []).append("Missing PDF EOF marker")
        
        except Exception as e:
            validation["pdf_valid"] = False
            validation["format_issues"] = [str(e)]
        
        return validation
    
    def get_file_category(self, file_path: Path) -> str:
        """Get simplified file category for analysis routing"""
        mime_type = self._detect_file_type(file_path).get("mime_type", "")
        
        if mime_type.startswith("image/"):
            return "image"
        elif mime_type.startswith("audio/"):
            return "audio"
        elif mime_type.startswith("video/"):
            return "video"
        elif mime_type == "application/pdf":
            return "pdf"
        elif mime_type in ["application/zip", "application/x-rar"]:
            return "archive"
        elif mime_type.startswith("text/"):
            return "text"
        else:
            return "binary"
