"""
File Forensics Tools - Advanced File Format Analysis and Data Carving
Supports PDF analysis, Office documents, polyglot detection, file carving, and signature analysis
"""

import asyncio
import logging
import magic
import struct
import hashlib
import zlib
import tempfile
import subprocess
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import json
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
import mimetypes

# PDF analysis
try:
    import PyPDF2
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# Office document analysis
try:
    import zipfile
    import oletools.olevba
    import oletools.rtfobj
    OLETOOLS_AVAILABLE = True
except ImportError:
    OLETOOLS_AVAILABLE = False

# Advanced file analysis
try:
    import yara
    YARA_AVAILABLE = True
except ImportError:
    YARA_AVAILABLE = False

try:
    import ssdeep
    SSDEEP_AVAILABLE = True
except ImportError:
    SSDEEP_AVAILABLE = False

class FileForensicsTools:
    def __init__(self, config):
        self.config = config.file_forensics
        self.logger = logging.getLogger(__name__)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="file_forensics_"))
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # File signature database
        self.file_signatures = self._load_file_signatures()
        
        # YARA rules for advanced detection
        self.yara_rules = None
        if YARA_AVAILABLE:
            self._load_yara_rules()
        
        # Tool availability
        self.tool_availability = self._check_tool_availability()
    
    def _load_file_signatures(self) -> Dict[str, Dict]:
        """Load comprehensive file signature database"""
        return {
            # Image formats
            'JPEG': {'signature': b'\xFF\xD8\xFF', 'footer': b'\xFF\xD9', 'offset': 0},
            'PNG': {'signature': b'\x89PNG\r\n\x1A\n', 'footer': b'IEND\xAE\x42\x60\x82', 'offset': 0},
            'GIF87a': {'signature': b'GIF87a', 'offset': 0},
            'GIF89a': {'signature': b'GIF89a', 'offset': 0},
            'BMP': {'signature': b'BM', 'offset': 0},
            'TIFF_LE': {'signature': b'II\x2A\x00', 'offset': 0},
            'TIFF_BE': {'signature': b'MM\x00\x2A', 'offset': 0},
            'WEBP': {'signature': b'RIFF', 'secondary': b'WEBP', 'offset': 0, 'secondary_offset': 8},
            
            # Archive formats
            'ZIP': {'signature': b'PK\x03\x04', 'offset': 0},
            'ZIP_EMPTY': {'signature': b'PK\x05\x06', 'offset': 0},
            'ZIP_SPAN': {'signature': b'PK\x07\x08', 'offset': 0},
            'RAR': {'signature': b'Rar!\x1A\x07\x00', 'offset': 0},
            'RAR5': {'signature': b'Rar!\x1A\x07\x01\x00', 'offset': 0},
            '7Z': {'signature': b'7z\xBC\xAF\x27\x1C', 'offset': 0},
            'GZIP': {'signature': b'\x1F\x8B', 'offset': 0},
            'BZIP2': {'signature': b'BZ', 'offset': 0},
            'TAR': {'signature': b'ustar', 'offset': 257},
            
            # Document formats
            'PDF': {'signature': b'%PDF', 'offset': 0},
            'DOC': {'signature': b'\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1', 'offset': 0},
            'DOCX': {'signature': b'PK\x03\x04', 'secondary': b'word/', 'offset': 0},
            'RTF': {'signature': b'{\\rtf', 'offset': 0},
            'PS': {'signature': b'%!PS', 'offset': 0},
            'EPS': {'signature': b'\xC5\xD0\xD3\xC6', 'offset': 0},
            
            # Audio/Video formats
            'MP3': {'signature': b'ID3', 'offset': 0},
            'MP3_ALT': {'signature': b'\xFF\xFB', 'offset': 0},
            'WAV': {'signature': b'RIFF', 'secondary': b'WAVE', 'offset': 0, 'secondary_offset': 8},
            'AVI': {'signature': b'RIFF', 'secondary': b'AVI ', 'offset': 0, 'secondary_offset': 8},
            'MP4': {'signature': b'ftyp', 'offset': 4},
            'MOV': {'signature': b'moov', 'offset': 4},
            'FLAC': {'signature': b'fLaC', 'offset': 0},
            'OGG': {'signature': b'OggS', 'offset': 0},
            
            # Executable formats
            'PE': {'signature': b'MZ', 'secondary': b'PE\x00\x00', 'offset': 0},
            'ELF': {'signature': b'\x7FELF', 'offset': 0},
            'MACH_O_32': {'signature': b'\xFE\xED\xFA\xCE', 'offset': 0},
            'MACH_O_64': {'signature': b'\xFE\xED\xFA\xCF', 'offset': 0},
            
            # Other formats
            'SQLITE': {'signature': b'SQLite format 3\x00', 'offset': 0},
            'CLASS': {'signature': b'\xCA\xFE\xBA\xBE', 'offset': 0},
            'PCAP': {'signature': b'\xD4\xC3\xB2\xA1', 'offset': 0},
            'PCAP_NG': {'signature': b'\x0A\x0D\x0D\x0A', 'offset': 0},
        }
    
    def _load_yara_rules(self):
        """Load YARA rules for advanced pattern detection"""
        try:
            # Basic YARA rules for steganography detection
            rules_source = '''
            rule steganography_tools {
                meta:
                    description = "Detects steganography tool signatures"
                strings:
                    $steghide = "steghide"
                    $outguess = "OutGuess"
                    $jphide = "jphide"
                    $invisible_secrets = "Invisible Secrets"
                    $s_tools = "S-Tools"
                condition:
                    any of them
            }
            
            rule embedded_files {
                meta:
                    description = "Detects embedded file signatures"
                strings:
                    $pdf = "%PDF"
                    $zip = {50 4B 03 04}
                    $rar = "Rar!"
                    $jpeg = {FF D8 FF}
                    $png = {89 50 4E 47}
                condition:
                    any of them at 100..filesize
            }
            
            rule suspicious_strings {
                meta:
                    description = "Detects suspicious strings"
                strings:
                    $flag1 = /flag\{[^}]+\}/
                    $flag2 = /FLAG\{[^}]+\}/
                    $base64 = /[A-Za-z0-9+\/]{50,}={0,2}/
                    $hex = /[0-9A-Fa-f]{40,}/
                condition:
                    any of them
            }
            '''
            
            self.yara_rules = yara.compile(source=rules_source)
            self.logger.info("YARA rules loaded successfully")
            
        except Exception as e:
            self.logger.warning(f"Failed to load YARA rules: {e}")
            self.yara_rules = None
    
    def _check_tool_availability(self) -> Dict[str, bool]:
        """Check availability of external tools"""
        tools = {}
        
        tool_checks = {
            'file': 'file --version',
            'binwalk': 'binwalk --help',
            'foremost': 'foremost -V',
            'scalpel': 'scalpel -V',
            'bulk_extractor': 'bulk_extractor -h',
            'strings': 'strings --version',
            'hexdump': 'hexdump -V',
            'xxd': 'xxd -v',
            'trid': 'trid',
            'magika': 'magika --version'
        }
        
        for tool, check_cmd in tool_checks.items():
            try:
                result = subprocess.run(
                    check_cmd.split(), 
                    capture_output=True, 
                    timeout=10
                )
                tools[tool] = result.returncode in [0, 1]
            except:
                tools[tool] = False
        
        return tools
    
    def execute_method(self, method: str, file_path: Path) -> List[Dict[str, Any]]:
        """Execute file forensics method"""
        method_map = {
            'magic_analysis': self._magic_analysis,
            'signature_verification': self._signature_verification,
            'header_analysis': self._header_analysis,
            'footer_analysis': self._footer_analysis,
            'polyglot_detection': self._polyglot_detection,
            'file_carving': self._file_carving,
            'bulk_extractor': self._bulk_extractor_analysis,
            'pdf_analysis': self._pdf_analysis,
            'office_analysis': self._office_analysis,
            'archive_analysis': self._archive_analysis,
            'executable_analysis': self._executable_analysis,
            'entropy_analysis': self._entropy_analysis,
            'string_analysis': self._string_analysis,
            'yara_scan': self._yara_scan,
            'overlay_analysis': self._overlay_analysis,
            'file_structure_analysis': self._file_structure_analysis
        }
        
        if method not in method_map:
            raise ValueError(f"Unknown file forensics method: {method}")
        
        try:
            return method_map[method](file_path)
        except Exception as e:
            self.logger.error(f"File forensics method {method} failed for {file_path}: {e}")
            return [{
                "type": "error",
                "method": method,
                "tool_name": "file_forensics",
                "confidence": 0.0,
                "details": f"File analysis failed: {str(e)}",
                "file_path": str(file_path)
            }]
    
    def _magic_analysis(self, file_path: Path) -> List[Dict[str, Any]]:
        """Analyze file using libmagic"""
        results = []
        
        try:
            # Get MIME type
            mime_type = magic.from_file(str(file_path), mime=True)
            
            # Get detailed description
            file_description = magic.from_file(str(file_path))
            
            # Get file extension
            file_extension = file_path.suffix.lower()
            
            # Check for mismatch between extension and detected type
            expected_mime = mimetypes.guess_type(str(file_path))[0]
            
            if expected_mime and expected_mime != mime_type:
                confidence = 0.7
                if 'text' in expected_mime and 'text' in mime_type:
                    confidence = 0.3  # Text type variations are common
                elif 'image' in expected_mime and 'image' in mime_type:
                    confidence = 0.4  # Image format variations
                
                results.append({
                    "type": "file_type_mismatch",
                    "method": "magic_analysis",
                    "tool_name": "file_forensics",
                    "confidence": confidence,
                    "details": f"Extension suggests {expected_mime}, magic detected {mime_type}",
                    "expected_mime": expected_mime,
                    "detected_mime": mime_type,
                    "file_extension": file_extension,
                    "file_description": file_description,
                    "file_path": str(file_path)
                })
            
            # Check for suspicious file descriptions
            suspicious_keywords = [
                'data', 'Non-ISO extended-ASCII', 'very long line',
                'with very long lines', 'with no line terminators'
            ]
            
            for keyword in suspicious_keywords:
                if keyword in file_description.lower():
                    results.append({
                        "type": "suspicious_file_description",
                        "method": "magic_description_analysis",
                        "tool_name": "file_forensics",
                        "confidence": 0.5,
                        "details": f"Suspicious file description: {file_description}",
                        "description": file_description,
                        "suspicious_keyword": keyword,
                        "file_path": str(file_path)
                    })
                    break
                    
        except Exception as e:
            self.logger.error(f"Magic analysis failed: {e}")
        
        return results
    
    def _signature_verification(self, file_path: Path) -> List[Dict[str, Any]]:
        """Verify file signatures against known patterns"""
        results = []
        
        try:
            with open(file_path, 'rb') as f:
                header = f.read(1024)  # Read first 1KB
                
                # Get file size for footer analysis
                f.seek(0, 2)
                file_size = f.tell()
                
                # Read footer
                footer_size = min(1024, file_size)
                f.seek(-footer_size, 2)
                footer = f.read(footer_size)
            
            detected_signatures = []
            
            # Check each signature
            for format_name, sig_info in self.file_signatures.items():
                signature = sig_info['signature']
                offset = sig_info.get('offset', 0)
                
                # Check main signature
                if len(header) > offset + len(signature):
                    if header[offset:offset + len(signature)] == signature:
                        match_info = {
                            'format': format_name,
                            'offset': offset,
                            'signature': signature.hex(),
                            'confidence': 0.9
                        }
                        
                        # Check secondary signature if present
                        if 'secondary' in sig_info:
                            sec_sig = sig_info['secondary']
                            sec_offset = sig_info.get('secondary_offset', 0)
                            
                            if len(header) > sec_offset + len(sec_sig):
                                if header[sec_offset:sec_offset + len(sec_sig)] == sec_sig:
                                    match_info['confidence'] = 0.95
                                    match_info['secondary_match'] = True
                                else:
                                    match_info['confidence'] = 0.6
                                    match_info['secondary_match'] = False
                        
                        # Check footer if specified
                        if 'footer' in sig_info:
                            footer_sig = sig_info['footer']
                            if footer_sig in footer:
                                match_info['confidence'] = min(match_info['confidence'] + 0.1, 1.0)
                                match_info['footer_match'] = True
                        
                        detected_signatures.append(match_info)
            
            # Analyze signature results
            if len(detected_signatures) > 1:
                # Multiple signatures detected - possible polyglot
                results.append({
                    "type": "multiple_signatures",
                    "method": "signature_verification",
                    "tool_name": "file_forensics",
                    "confidence": 0.8,
                    "details": f"Multiple file signatures detected: {len(detected_signatures)}",
                    "detected_signatures": detected_signatures,
                    "signature_count": len(detected_signatures),
                    "file_path": str(file_path)
                })
            
            elif len(detected_signatures) == 1:
                sig = detected_signatures[0]
                
                # Check if detected signature matches file extension
                file_ext = file_path.suffix.lower()
                format_extensions = {
                    'JPEG': ['.jpg', '.jpeg'],
                    'PNG': ['.png'],
                    'GIF87a': ['.gif'],
                    'GIF89a': ['.gif'],
                    'BMP': ['.bmp'],
                    'PDF': ['.pdf'],
                    'ZIP': ['.zip'],
                    'RAR': ['.rar'],
                    'PE': ['.exe', '.dll'],
                    'ELF': [''],
                }
                
                expected_exts = format_extensions.get(sig['format'], [])
                if expected_exts and file_ext not in expected_exts:
                    results.append({
                        "type": "signature_extension_mismatch",
                        "method": "signature_verification",
                        "tool_name": "file_forensics",
                        "confidence": 0.7,
                        "details": f"Signature indicates {sig['format']}, but extension is {file_ext}",
                        "detected_format": sig['format'],
                        "file_extension": file_ext,
                        "expected_extensions": expected_exts,
                        "signature_info": sig,
                        "file_path": str(file_path)
                    })
            
            elif len(detected_signatures) == 0:
                # No known signatures - might be hidden/obfuscated
                results.append({
                    "type": "no_known_signature",
                    "method": "signature_verification",
                    "tool_name": "file_forensics",
                    "confidence": 0.4,
                    "details": "No known file signatures detected",
                    "header_hex": header[:32].hex(),
                    "file_path": str(file_path)
                })
                
        except Exception as e:
            self.logger.error(f"Signature verification failed: {e}")
        
        return results
    
    def _header_analysis(self, file_path: Path) -> List[Dict[str, Any]]:
        """Detailed analysis of file headers"""
        results = []
        
        try:
            with open(file_path, 'rb') as f:
                header = f.read(512)  # Read first 512 bytes
            
            # Analyze header characteristics
            header_analysis = self._analyze_header_characteristics(header)
            if header_analysis:
                results.extend(header_analysis)
            
            # Look for embedded signatures within header
            embedded_sigs = self._find_embedded_signatures(header, 'header')
            if embedded_sigs:
                results.extend(embedded_sigs)
                
        except Exception as e:
            self.logger.error(f"Header analysis failed: {e}")
        
        return results
    
    def _analyze_header_characteristics(self, header: bytes) -> List[Dict[str, Any]]:
        """Analyze characteristics of file header"""
        results = []
        
        if len(header) == 0:
            return results
        
        # Calculate entropy of header
        header_entropy = self._calculate_entropy(header)
        
        if header_entropy > 7.5:  # High entropy
            results.append({
                "type": "high_header_entropy",
                "method": "header_entropy_analysis",
                "tool_name": "file_forensics",
                "confidence": min((header_entropy - 7.0) * 2, 0.9),
                "details": f"High entropy in file header: {header_entropy:.3f}",
                "header_entropy": float(header_entropy),
                "file_path": str(self.current_file_path) if hasattr(self, 'current_file_path') else "unknown"
            })
        
        # Check for null bytes in header
        null_count = header.count(b'\x00')
        null_percentage = null_count / len(header) * 100
        
        if null_percentage > 50:  # More than 50% null bytes
            results.append({
                "type": "excessive_nulls_in_header",
                "method": "header_null_analysis",
                "tool_name": "file_forensics",
                "confidence": 0.6,
                "details": f"Excessive null bytes in header: {null_percentage:.1f}%",
                "null_percentage": float(null_percentage),
                "file_path": str(self.current_file_path) if hasattr(self, 'current_file_path') else "unknown"
            })
        
        # Check for ASCII text in binary header regions
        ascii_chars = sum(1 for b in header if 32 <= b <= 126)
        ascii_percentage = ascii_chars / len(header) * 100
        
        if ascii_percentage > 80:  # Mostly ASCII in what should be binary header
            results.append({
                "type": "ascii_in_binary_header",
                "method": "header_ascii_analysis",
                "tool_name": "file_forensics",
                "confidence": 0.5,
                "details": f"High ASCII content in header: {ascii_percentage:.1f}%",
                "ascii_percentage": float(ascii_percentage),
                "file_path": str(self.current_file_path) if hasattr(self, 'current_file_path') else "unknown"
            })
        
        return results
    
    def _find_embedded_signatures(self, data: bytes, region: str) -> List[Dict[str, Any]]:
        """Find embedded file signatures within data"""
        results = []
        
        # Look for file signatures at non-standard positions
        for format_name, sig_info in self.file_signatures.items():
            signature = sig_info['signature']
            expected_offset = sig_info.get('offset', 0)
            
            # Find all occurrences of this signature
            start = 0
            while True:
                pos = data.find(signature, start)
                if pos == -1:
                    break
                
                # If found at unexpected position
                if pos != expected_offset and pos > 10:  # Not at expected position and not near start
                    results.append({
                        "type": "embedded_signature",
                        "method": f"{region}_signature_analysis",
                        "tool_name": "file_forensics",
                        "confidence": 0.7,
                        "details": f"{format_name} signature found at unexpected position {pos} in {region}",
                        "format": format_name,
                        "position": pos,
                        "expected_position": expected_offset,
                        "signature": signature.hex(),
                        "region": region,
                        "file_path": str(self.current_file_path) if hasattr(self, 'current_file_path') else "unknown"
                    })
                
                start = pos + 1
        
        return results
    
    def _footer_analysis(self, file_path: Path) -> List[Dict[str, Any]]:
        """Analysis of file footers/trailers"""
        results = []
        
        try:
            with open(file_path, 'rb') as f:
                # Get file size
                f.seek(0, 2)
                file_size = f.tell()
                
                if file_size < 1024:
                    footer_size = file_size
                else:
                    footer_size = 1024
                
                # Read footer
                f.seek(-footer_size, 2)
                footer = f.read(footer_size)
            
            # Analyze footer characteristics
            footer_analysis = self._analyze_footer_characteristics(footer)
            if footer_analysis:
                results.extend(footer_analysis)
            
            # Look for embedded signatures in footer
            embedded_sigs = self._find_embedded_signatures(footer, 'footer')
            if embedded_sigs:
                results.extend(embedded_sigs)
            
            # Check for data after known file endings
            trailing_data = self._check_trailing_data(footer, file_path)
            if trailing_data:
                results.extend(trailing_data)
                
        except Exception as e:
            self.logger.error(f"Footer analysis failed: {e}")
        
        return results
    
    def _analyze_footer_characteristics(self, footer: bytes) -> List[Dict[str, Any]]:
        """Analyze characteristics of file footer"""
        results = []
        
        if len(footer) == 0:
            return results
        
        # Calculate entropy of footer
        footer_entropy = self._calculate_entropy(footer)
        
        if footer_entropy > 7.5:  # High entropy in footer
            results.append({
                "type": "high_footer_entropy",
                "method": "footer_entropy_analysis",
                "tool_name": "file_forensics",
                "confidence": min((footer_entropy - 7.0) * 2, 0.8),
                "details": f"High entropy in file footer: {footer_entropy:.3f}",
                "footer_entropy": float(footer_entropy),
                "file_path": str(self.current_file_path) if hasattr(self, 'current_file_path') else "unknown"
            })
        
        # Look for structured data in footer
        if self._has_structured_data(footer):
            results.append({
                "type": "structured_data_in_footer",
                "method": "footer_structure_analysis",
                "tool_name": "file_forensics",
                "confidence": 0.6,
                "details": "Structured data detected in file footer",
                "file_path": str(self.current_file_path) if hasattr(self, 'current_file_path') else "unknown"
            })
        
        return results
    
    def _has_structured_data(self, data: bytes) -> bool:
        """Check if data contains structured patterns"""
        # Look for repeating patterns
        if len(data) < 16:
            return False
        
        # Check for XML-like structures
        if b'<' in data and b'>' in data:
            return True
        
        # Check for JSON-like structures
        if b'{' in data and b'}' in data:
            return True
        
        # Check for binary structures (repeated 4-byte patterns)
        for i in range(0, len(data) - 8, 4):
            pattern = data[i:i+4]
            if data.count(pattern) > 3:  # Pattern repeats more than 3 times
                return True
        
        return False
    
    def _check_trailing_data(self, footer: bytes, file_path: Path) -> List[Dict[str, Any]]:
        """Check for data after known file endings"""
        results = []
        
        # Known file endings
        file_endings = {
            'JPEG': b'\xFF\xD9',
            'PNG': b'IEND\xAE\x42\x60\x82',
            'GIF': b'\x00\x3B',
            'PDF': b'%%EOF',
            'ZIP': b'PK\x05\x06',
        }
        
        for format_name, ending in file_endings.items():
            pos = footer.find(ending)
            if pos != -1:
                # Check if there's significant data after the ending
                after_ending = footer[pos + len(ending):]
                
                if len(after_ending) > 10:  # More than 10 bytes after ending
                    # Filter out common padding (null bytes, whitespace)
                    significant_data = after_ending.strip(b'\x00\x20\x0A\x0D')
                    
                    if len(significant_data) > 5:
                        results.append({
                            "type": "data_after_file_ending",
                            "method": "trailing_data_analysis",
                            "tool_name": "file_forensics",
                            "confidence": 0.8,
                            "details": f"Data found after {format_name} ending: {len(significant_data)} bytes",
                            "format": format_name,
                            "trailing_bytes": len(significant_data),
                            "trailing_data_preview": significant_data[:50].hex(),
                            "file_path": str(file_path)
                        })
        
        return results
    
    def _polyglot_detection(self, file_path: Path) -> List[Dict[str, Any]]:
        """Detect polyglot files (files valid as multiple formats)"""
        results = []
        
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
            
            # Test file against multiple format validators
            valid_formats = []
            
            # Test common polyglot combinations
            format_tests = {
                'PDF': self._test_pdf_validity,
                'ZIP': self._test_zip_validity,
                'JPEG': self._test_jpeg_validity,
                'GIF': self._test_gif_validity,
                'HTML': self._test_html_validity,
                'SCRIPT': self._test_script_validity
            }
            
            for format_name, test_func in format_tests.items():
                try:
                    if test_func(data):
                        valid_formats.append(format_name)
                except:
                    continue
            
            if len(valid_formats) > 1:
                results.append({
                    "type": "polyglot_file",
                    "method": "polyglot_detection",
                    "tool_name": "file_forensics",
                    "confidence": 0.9,
                    "details": f"File valid as multiple formats: {', '.join(valid_formats)}",
                    "valid_formats": valid_formats,
                    "format_count": len(valid_formats),
                    "file_path": str(file_path)
                })
                
        except Exception as e:
            self.logger.error(f"Polyglot detection failed: {e}")
        
        return results
    
    def _test_pdf_validity(self, data: bytes) -> bool:
        """Test if data is valid PDF"""
        return data.startswith(b'%PDF') and b'%%EOF' in data
    
    def _test_zip_validity(self, data: bytes) -> bool:
        """Test if data is valid ZIP"""
        try:
            import zipfile
            import io
            with zipfile.ZipFile(io.BytesIO(data), 'r') as zf:
                zf.testzip()
            return True
        except:
            return False
    
    def _test_jpeg_validity(self, data: bytes) -> bool:
        """Test if data is valid JPEG"""
        return data.startswith(b'\xFF\xD8\xFF') and data.endswith(b'\xFF\xD9')
    
    def _test_gif_validity(self, data: bytes) -> bool:
        """Test if data is valid GIF"""
        return (data.startswith(b'GIF87a') or data.startswith(b'GIF89a')) and data.endswith(b'\x00\x3B')
    
    def _test_html_validity(self, data: bytes) -> bool:
        """Test if data contains valid HTML"""
        try:
            text = data.decode('utf-8', errors='ignore').lower()
            return '<html' in text or '<!doctype html' in text
        except:
            return False
    
    def _test_script_validity(self, data: bytes) -> bool:
        """Test if data contains script code"""
        try:
            text = data.decode('utf-8', errors='ignore').lower()
            script_indicators = ['<script', 'javascript:', 'function(', 'var ', 'let ', 'const ']
            return any(indicator in text for indicator in script_indicators)
        except:
            return False
    
    def _file_carving(self, file_path: Path) -> List[Dict[str, Any]]:
        """Perform file carving to find embedded files"""
        results = []
        
        if not self.tool_availability.get('foremost', False):
            return results
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Run foremost
                cmd = ['foremost', '-i', str(file_path), '-o', temp_dir, '-q']
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                
                # Check results
                output_dir = Path(temp_dir)
                carved_files = []
                
                for subdir in output_dir.iterdir():
                    if subdir.is_dir():
                        for carved_file in subdir.iterdir():
                            if carved_file.is_file() and carved_file.stat().st_size > 0:
                                file_info = {
                                    'name': carved_file.name,
                                    'type': subdir.name,
                                    'size': carved_file.stat().st_size,
                                    'hash': self._calculate_file_hash(carved_file)
                                }
                                carved_files.append(file_info)
                
                if carved_files:
                    results.append({
                        "type": "carved_files",
                        "method": "file_carving",
                        "tool_name": "foremost",
                        "confidence": 0.8,
                        "details": f"File carving found {len(carved_files)} embedded files",
                        "carved_files": carved_files,
                        "carving_tool": "foremost",
                        "file_path": str(file_path)
                    })
                    
        except subprocess.TimeoutExpired:
            self.logger.warning(f"File carving timeout for {file_path}")
        except Exception as e:
            self.logger.error(f"File carving failed: {e}")
        
        return results
    
    def _bulk_extractor_analysis(self, file_path: Path) -> List[Dict[str, Any]]:
        """Run bulk_extractor for comprehensive data extraction"""
        results = []
        
        if not self.tool_availability.get('bulk_extractor', False):
            return results
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Run bulk_extractor
                cmd = ['bulk_extractor', '-o', temp_dir, str(file_path)]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                
                # Analyze extracted features
                feature_files = Path(temp_dir).glob('*.txt')
                
                extracted_features = {}
                for feature_file in feature_files:
                    feature_name = feature_file.stem
                    
                    try:
                        with open(feature_file, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read().strip()
                        
                        if content and len(content) > 10:  # Non-empty meaningful content
                            lines = content.split('\n')
                            # Filter out comments and headers
                            data_lines = [line for line in lines if not line.startswith('#')]
                            
                            if data_lines:
                                extracted_features[feature_name] = {
                                    'count': len(data_lines),
                                    'samples': data_lines[:5]  # First 5 samples
                                }
                    except Exception as e:
                        self.logger.debug(f"Failed to read feature file {feature_file}: {e}")
                
                if extracted_features:
                    results.append({
                        "type": "bulk_extractor_features",
                        "method": "bulk_extractor",
                        "tool_name": "bulk_extractor",
                        "confidence": 0.7,
                        "details": f"Bulk extractor found {len(extracted_features)} feature types",
                        "extracted_features": extracted_features,
                        "feature_count": len(extracted_features),
                        "file_path": str(file_path)
                    })
                    
        except subprocess.TimeoutExpired:
            self.logger.warning(f"Bulk extractor timeout for {file_path}")
        except Exception as e:
            self.logger.error(f"Bulk extractor failed: {e}")
        
        return results
    
    def _pdf_analysis(self, file_path: Path) -> List[Dict[str, Any]]:
        """Comprehensive PDF analysis"""
        results = []
        
        if not PDF_AVAILABLE or not str(file_path).lower().endswith('.pdf'):
            return results
        
        try:
            # Basic PDF structure analysis
            pdf_structure = self._analyze_pdf_structure(file_path)
            if pdf_structure:
                results.extend(pdf_structure)
            
            # PDF content analysis
            pdf_content = self._analyze_pdf_content(file_path)
            if pdf_content:
                results.extend(pdf_content)
            
            # PDF metadata analysis
            pdf_metadata = self._analyze_pdf_metadata(file_path)
            if pdf_metadata:
                results.extend(pdf_metadata)
                
        except Exception as e:
            self.logger.error(f"PDF analysis failed: {e}")
        
        return results
    
    def _analyze_pdf_structure(self, file_path: Path) -> List[Dict[str, Any]]:
        """Analyze PDF internal structure"""
        results = []
        
        try:
            with open(file_path, 'rb') as f:
                pdf_data = f.read()
            
            # Count PDF objects
            obj_pattern = re.compile(rb'\d+\s+\d+\s+obj')
            objects = obj_pattern.findall(pdf_data)
            
            # Count streams
            stream_count = pdf_data.count(b'stream')
            
            # Look for JavaScript
            js_indicators = [b'/JS', b'/JavaScript', b'function', b'var ']
            js_found = any(indicator in pdf_data for indicator in js_indicators)
            
            # Look for forms
            form_indicators = [b'/AcroForm', b'/XFA']
            forms_found = any(indicator in pdf_data for indicator in form_indicators)
            
            # Look for embedded files
            embed_indicators = [b'/EmbeddedFile', b'/Filespec']
            embedded_files = any(indicator in pdf_data for indicator in embed_indicators)
            
            if len(objects) > 100:  # Many objects
                results.append({
                    "type": "complex_pdf_structure",
                    "method": "pdf_structure_analysis",
                    "tool_name": "file_forensics",
                    "confidence": 0.5,
                    "details": f"PDF has many objects: {len(objects)}",
                    "object_count": len(objects),
                    "stream_count": stream_count,
                    "file_path": str(file_path)
                })
            
            if js_found:
                results.append({
                    "type": "pdf_javascript",
                    "method": "pdf_js_detection",
                    "tool_name": "file_forensics",
                    "confidence": 0.7,
                    "details": "JavaScript detected in PDF",
                    "file_path": str(file_path)
                })
            
            if embedded_files:
                results.append({
                    "type": "pdf_embedded_files",
                    "method": "pdf_embed_detection",
                    "tool_name": "file_forensics",
                    "confidence": 0.8,
                    "details": "Embedded files detected in PDF",
                    "file_path": str(file_path)
                })
                
        except Exception as e:
            self.logger.debug(f"PDF structure analysis failed: {e}")
        
        return results
    
    def _analyze_pdf_content(self, file_path: Path) -> List[Dict[str, Any]]:
        """Analyze PDF content using pdfplumber"""
        results = []
        
        try:
            import pdfplumber
            
            with pdfplumber.open(file_path) as pdf:
                # Extract text from all pages
                full_text = ""
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        full_text += text + "\n"
                
                if full_text:
                    # Look for suspicious patterns in text
                    text_analysis = self._analyze_extracted_text(full_text, "PDF")
                    if text_analysis:
                        results.extend(text_analysis)
                
                # Analyze images in PDF
                for page_num, page in enumerate(pdf.pages):
                    if hasattr(page, 'images'):
                        images = page.images
                        if len(images) > 10:  # Many images
                            results.append({
                                "type": "pdf_many_images",
                                "method": "pdf_image_analysis",
                                "tool_name": "file_forensics",
                                "confidence": 0.4,
                                "details": f"Page {page_num + 1} has many images: {len(images)}",
                                "page_number": page_num + 1,
                                "image_count": len(images),
                                "file_path": str(file_path)
                            })
                            
        except Exception as e:
            self.logger.debug(f"PDF content analysis failed: {e}")
        
        return results
    
    def _analyze_pdf_metadata(self, file_path: Path) -> List[Dict[str, Any]]:
        """Analyze PDF metadata"""
        results = []
        
        try:
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                
                if reader.metadata:
                    metadata = dict(reader.metadata)
                    
                    # Check for suspicious metadata
                    for key, value in metadata.items():
                        if isinstance(value, str) and len(value) > 200:
                            results.append({
                                "type": "suspicious_pdf_metadata",
                                "method": "pdf_metadata_analysis",
                                "tool_name": "file_forensics",
                                "confidence": 0.6,
                                "details": f"Unusually long metadata field: {key}",
                                "metadata_field": key,
                                "field_length": len(value),
                                "field_preview": value[:100],
                                "file_path": str(file_path)
                            })
                            
        except Exception as e:
            self.logger.debug(f"PDF metadata analysis failed: {e}")
        
        return results
    
    def _office_analysis(self, file_path: Path) -> List[Dict[str, Any]]:
        """Analyze Microsoft Office documents"""
        results = []
        
        file_ext = file_path.suffix.lower()
        office_extensions = {'.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx'}
        
        if file_ext not in office_extensions:
            return results
        
        try:
            if file_ext in {'.docx', '.xlsx', '.pptx'}:
                # Modern Office format (ZIP-based)
                office_zip_analysis = self._analyze_office_zip(file_path)
                if office_zip_analysis:
                    results.extend(office_zip_analysis)
            else:
                # Legacy Office format (OLE-based)
                office_ole_analysis = self._analyze_office_ole(file_path)
                if office_ole_analysis:
                    results.extend(office_ole_analysis)
                    
        except Exception as e:
            self.logger.error(f"Office analysis failed: {e}")
        
        return results
    
    def _analyze_office_zip(self, file_path: Path) -> List[Dict[str, Any]]:
        """Analyze modern Office documents (ZIP-based)"""
        results = []
        
        try:
            with zipfile.ZipFile(file_path, 'r') as zf:
                file_list = zf.namelist()
                
                # Look for suspicious files
                suspicious_files = []
                for filename in file_list:
                    if filename.startswith('..') or '/' in filename and not filename.startswith(('word/', 'xl/', 'ppt/')):
                        suspicious_files.append(filename)
                
                if suspicious_files:
                    results.append({
                        "type": "suspicious_office_files",
                        "method": "office_zip_analysis",
                        "tool_name": "file_forensics",
                        "confidence": 0.7,
                        "details": f"Suspicious files in Office document: {len(suspicious_files)}",
                        "suspicious_files": suspicious_files[:10],
                        "file_path": str(file_path)
                    })
                
                # Check for macros (VBA files)
                vba_files = [f for f in file_list if 'vba' in f.lower() or f.endswith('.bin')]
                if vba_files:
                    results.append({
                        "type": "office_macros_detected",
                        "method": "office_macro_detection",
                        "tool_name": "file_forensics",
                        "confidence": 0.8,
                        "details": f"Macros detected in Office document: {len(vba_files)} VBA files",
                        "vba_files": vba_files,
                        "file_path": str(file_path)
                    })
                
                # Check file sizes for anomalies
                large_files = []
                for filename in file_list:
                    try:
                        info = zf.getinfo(filename)
                        if info.file_size > 1024 * 1024:  # Files larger than 1MB
                            large_files.append({
                                'name': filename,
                                'size': info.file_size,
                                'compressed_size': info.compress_size
                            })
                    except:
                        continue
                
                if large_files:
                    results.append({
                        "type": "large_embedded_files",
                        "method": "office_size_analysis",
                        "tool_name": "file_forensics",
                        "confidence": 0.5,
                        "details": f"Large embedded files detected: {len(large_files)}",
                        "large_files": large_files,
                        "file_path": str(file_path)
                    })
                    
        except Exception as e:
            self.logger.debug(f"Office ZIP analysis failed: {e}")
        
        return results
    
    def _analyze_office_ole(self, file_path: Path) -> List[Dict[str, Any]]:
        """Analyze legacy Office documents (OLE-based)"""
        results = []
        
        if not OLETOOLS_AVAILABLE:
            return results
        
        try:
            # Use oletools for VBA analysis
            from oletools.olevba import VBA_Parser
            
            vba_parser = VBA_Parser(str(file_path))
            
            if vba_parser.detect_vba_macros():
                results.append({
                    "type": "ole_macros_detected",
                    "method": "ole_vba_analysis",
                    "tool_name": "oletools",
                    "confidence": 0.8,
                    "details": "VBA macros detected in OLE document",
                    "file_path": str(file_path)
                })
                
                # Analyze macro code
                for (filename, stream_path, vba_filename, vba_code) in vba_parser.extract_macros():
                    if vba_code:
                        macro_analysis = self._analyze_vba_code(vba_code)
                        if macro_analysis:
                            results.extend(macro_analysis)
            
            vba_parser.close()
            
        except Exception as e:
            self.logger.debug(f"OLE analysis failed: {e}")
        
        return results
    
    def _analyze_vba_code(self, vba_code: str) -> List[Dict[str, Any]]:
        """Analyze VBA macro code for suspicious patterns"""
        results = []
        
        # Suspicious VBA patterns
        suspicious_patterns = {
            'auto_execution': ['Auto_Open', 'Workbook_Open', 'Document_Open'],
            'file_operations': ['Open', 'SaveAs', 'CreateObject'],
            'network': ['URLDownloadToFile', 'WinHttp', 'XMLHTTP'],
            'execution': ['Shell', 'CreateProcess', 'Run'],
            'encryption': ['Xor', 'Chr', 'Asc'],
            'obfuscation': ['Replace', 'Mid', 'Left', 'Right']
        }
        
        found_patterns = {}
        for category, patterns in suspicious_patterns.items():
            matches = []
            for pattern in patterns:
                if pattern.lower() in vba_code.lower():
                    matches.append(pattern)
            
            if matches:
                found_patterns[category] = matches
        
        if found_patterns:
            confidence = min(len(found_patterns) * 0.2, 0.9)
            
            results.append({
                "type": "suspicious_vba_patterns",
                "method": "vba_pattern_analysis",
                "tool_name": "file_forensics",
                "confidence": confidence,
                "details": f"Suspicious VBA patterns found: {', '.join(found_patterns.keys())}",
                "pattern_categories": list(found_patterns.keys()),
                "patterns": found_patterns,
                "vba_length": len(vba_code),
                "file_path": str(self.current_file_path) if hasattr(self, 'current_file_path') else "unknown"
            })
        
        return results
    
    def _archive_analysis(self, file_path: Path) -> List[Dict[str, Any]]:
        """Analyze archive files"""
        results = []
        
        # Detect archive type
        archive_type = self._detect_archive_type(file_path)
        if not archive_type:
            return results
        
        try:
            if archive_type == 'zip':
                results.extend(self._analyze_zip_archive(file_path))
            elif archive_type == 'rar':
                results.extend(self._analyze_rar_archive(file_path))
            # Add other archive types as needed
            
        except Exception as e:
            self.logger.error(f"Archive analysis failed: {e}")
        
        return results
    
    def _detect_archive_type(self, file_path: Path) -> Optional[str]:
        """Detect archive file type"""
        try:
            with open(file_path, 'rb') as f:
                header = f.read(10)
            
            if header.startswith(b'PK'):
                return 'zip'
            elif header.startswith(b'Rar!'):
                return 'rar'
            elif header.startswith(b'7z\xBC\xAF\x27\x1C'):
                return '7z'
            
        except Exception:
            pass
        
        return None
    
    def _analyze_zip_archive(self, file_path: Path) -> List[Dict[str, Any]]:
        """Analyze ZIP archive"""
        results = []
        
        try:
            with zipfile.ZipFile(file_path, 'r') as zf:
                file_list = zf.namelist()
                
                # Check for password protection
                for filename in file_list:
                    try:
                        zf.getinfo(filename)
                        # Try to read a small amount
                        with zf.open(filename) as f:
                            f.read(1)
                    except RuntimeError as e:
                        if 'password required' in str(e).lower():
                            results.append({
                                "type": "password_protected_archive",
                                "method": "archive_password_detection",
                                "tool_name": "file_forensics",
                                "confidence": 0.9,
                                "details": "Password-protected files detected in ZIP archive",
                                "protected_files": [filename],
                                "file_path": str(file_path)
                            })
                            break
                
                # Check for suspicious file paths
                suspicious_paths = []
                for filename in file_list:
                    if '..' in filename or filename.startswith('/'):
                        suspicious_paths.append(filename)
                
                if suspicious_paths:
                    results.append({
                        "type": "suspicious_archive_paths",
                        "method": "archive_path_analysis",
                        "tool_name": "file_forensics",
                        "confidence": 0.8,
                        "details": f"Suspicious file paths in archive: {len(suspicious_paths)}",
                        "suspicious_paths": suspicious_paths[:10],
                        "file_path": str(file_path)
                    })
                
                # Check compression ratios
                high_compression_files = []
                for filename in file_list:
                    try:
                        info = zf.getinfo(filename)
                        if info.file_size > 0:
                            ratio = info.compress_size / info.file_size
                            if ratio < 0.1:  # Very high compression
                                high_compression_files.append({
                                    'name': filename,
                                    'original_size': info.file_size,
                                    'compressed_size': info.compress_size,
                                    'ratio': ratio
                                })
                    except:
                        continue
                
                if high_compression_files:
                    results.append({
                        "type": "high_compression_files",
                        "method": "archive_compression_analysis",
                        "tool_name": "file_forensics",
                        "confidence": 0.6,
                        "details": f"Files with very high compression ratios: {len(high_compression_files)}",
                        "high_compression_files": high_compression_files[:5],
                        "file_path": str(file_path)
                    })
                    
        except Exception as e:
            self.logger.debug(f"ZIP analysis failed: {e}")
        
        return results
    
    def _analyze_rar_archive(self, file_path: Path) -> List[Dict[str, Any]]:
        """Analyze RAR archive"""
        results = []
        
        # RAR analysis would require external tools or libraries
        # This is a placeholder for RAR-specific analysis
        
        return results
    
    def _executable_analysis(self, file_path: Path) -> List[Dict[str, Any]]:
        """Analyze executable files"""
        results = []
        
        try:
            with open(file_path, 'rb') as f:
                header = f.read(1024)
            
            # Check for PE format
            if header.startswith(b'MZ'):
                pe_analysis = self._analyze_pe_file(file_path, header)
                if pe_analysis:
                    results.extend(pe_analysis)
            
            # Check for ELF format
            elif header.startswith(b'\x7FELF'):
                elf_analysis = self._analyze_elf_file(file_path, header)
                if elf_analysis:
                    results.extend(elf_analysis)
                    
        except Exception as e:
            self.logger.error(f"Executable analysis failed: {e}")
        
        return results
    
    def _analyze_pe_file(self, file_path: Path, header: bytes) -> List[Dict[str, Any]]:
        """Analyze PE (Windows executable) files"""
        results = []
        
        try:
            # Check for overlay (data after the end of the PE file)
            with open(file_path, 'rb') as f:
                f.seek(0, 2)
                file_size = f.tell()
                
                # Simple PE header parsing to find end of file
                f.seek(0)
                dos_header = f.read(64)
                
                if len(dos_header) >= 60:
                    pe_offset = struct.unpack('<I', dos_header[60:64])[0]
                    
                    # Read PE header
                    f.seek(pe_offset)
                    pe_signature = f.read(4)
                    
                    if pe_signature == b'PE\x00\x00':
                        # This is a simplified analysis
                        # A complete PE parser would be needed for thorough analysis
                        
                        # Check file size vs expected size
                        if file_size > 10 * 1024 * 1024:  # Large executable
                            results.append({
                                "type": "large_executable",
                                "method": "pe_size_analysis",
                                "tool_name": "file_forensics",
                                "confidence": 0.5,
                                "details": f"Large PE file: {file_size // (1024*1024)} MB",
                                "file_size_mb": file_size // (1024*1024),
                                "file_path": str(file_path)
                            })
                            
        except Exception as e:
            self.logger.debug(f"PE analysis failed: {e}")
        
        return results
    
    def _analyze_elf_file(self, file_path: Path, header: bytes) -> List[Dict[str, Any]]:
        """Analyze ELF (Linux executable) files"""
        results = []
        
        # ELF analysis placeholder
        # Would require parsing ELF headers and sections
        
        return results
    
    def _entropy_analysis(self, file_path: Path) -> List[Dict[str, Any]]:
        """Analyze file entropy for randomness detection"""
        results = []
        
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
            
            # Calculate overall entropy
            overall_entropy = self._calculate_entropy(data)
            
            # Calculate entropy in chunks
            chunk_size = 8192
            chunk_entropies = []
            
            for i in range(0, len(data), chunk_size):
                chunk = data[i:i+chunk_size]
                if len(chunk) > 100:  # Only analyze meaningful chunks
                    chunk_entropy = self._calculate_entropy(chunk)
                    chunk_entropies.append(chunk_entropy)
            
            if chunk_entropies:
                avg_chunk_entropy = sum(chunk_entropies) / len(chunk_entropies)
                entropy_variance = sum((e - avg_chunk_entropy) ** 2 for e in chunk_entropies) / len(chunk_entropies)
                
                # High overall entropy
                if overall_entropy > 7.5:
                    results.append({
                        "type": "high_file_entropy",
                        "method": "entropy_analysis",
                        "tool_name": "file_forensics",
                        "confidence": min((overall_entropy - 7.0) * 2, 0.9),
                        "details": f"High file entropy: {overall_entropy:.3f}",
                        "overall_entropy": float(overall_entropy),
                        "avg_chunk_entropy": float(avg_chunk_entropy),
                        "entropy_variance": float(entropy_variance),
                        "file_path": str(file_path)
                    })
                
                # Low entropy variance (uniform randomness)
                if entropy_variance < 0.1 and avg_chunk_entropy > 7.0:
                    results.append({
                        "type": "uniform_high_entropy",
                        "method": "entropy_variance_analysis",
                        "tool_name": "file_forensics",
                        "confidence": 0.7,
                        "details": f"Uniform high entropy across file chunks",
                        "avg_chunk_entropy": float(avg_chunk_entropy),
                        "entropy_variance": float(entropy_variance),
                        "file_path": str(file_path)
                    })
                    
        except Exception as e:
            self.logger.error(f"Entropy analysis failed: {e}")
        
        return results
    
    def _calculate_entropy(self, data: bytes) -> float:
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
                entropy -= probability * (probability.bit_length() - 1)
        
        return entropy
    
    def _string_analysis(self, file_path: Path) -> List[Dict[str, Any]]:
        """Extract and analyze strings from file"""
        results = []
        
        try:
            # Extract strings using the strings command if available
            if self.tool_availability.get('strings', False):
                cmd = ['strings', '-n', '4', str(file_path)]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                
                if result.stdout:
                    strings_list = result.stdout.strip().split('\n')
                    string_analysis = self._analyze_extracted_strings(strings_list)
                    if string_analysis:
                        results.extend(string_analysis)
            else:
                # Fallback: manual string extraction
                with open(file_path, 'rb') as f:
                    data = f.read()
                
                strings_list = self._extract_strings_manual(data)
                string_analysis = self._analyze_extracted_strings(strings_list)
                if string_analysis:
                    results.extend(string_analysis)
                    
        except Exception as e:
            self.logger.error(f"String analysis failed: {e}")
        
        return results
    
    def _extract_strings_manual(self, data: bytes, min_length: int = 4) -> List[str]:
        """Manually extract printable strings from binary data"""
        strings = []
        current_string = ""
        
        for byte in data:
            if 32 <= byte <= 126:  # Printable ASCII
                current_string += chr(byte)
            else:
                if len(current_string) >= min_length:
                    strings.append(current_string)
                current_string = ""
        
        # Don't forget the last string
        if len(current_string) >= min_length:
            strings.append(current_string)
        
        return strings
    
    def _analyze_extracted_strings(self, strings_list: List[str]) -> List[Dict[str, Any]]:
        """Analyze extracted strings for suspicious patterns"""
        results = []
        
        suspicious_patterns = {
            'base64': re.compile(r'^[A-Za-z0-9+/]{20,}={0,2}$'),
            'hex': re.compile(r'^[0-9A-Fa-f]{20,}$'),
            'flag': re.compile(r'flag\{.*?\}', re.IGNORECASE),
            'password': re.compile(r'password\s*[:=]\s*\S+', re.IGNORECASE),
            'url': re.compile(r'https?://[^\s]+'),
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'ip_address': re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'),
            'file_path': re.compile(r'[A-Za-z]:\\[^<>:"|?*\n\r]+|/[^<>:"|?*\n\r]+'),
        }
        
        pattern_matches = {}
        for string in strings_list:
            for pattern_name, pattern in suspicious_patterns.items():
                if pattern.search(string):
                    if pattern_name not in pattern_matches:
                        pattern_matches[pattern_name] = []
                    pattern_matches[pattern_name].append(string)
        
        for pattern_name, matches in pattern_matches.items():
            if matches:
                confidence = min(len(matches) * 0.1, 0.8)
                
                results.append({
                    "type": f"suspicious_strings_{pattern_name}",
                    "method": "string_pattern_analysis",
                    "tool_name": "file_forensics",
                    "confidence": confidence,
                    "details": f"Found {len(matches)} {pattern_name} patterns in strings",
                    "pattern_type": pattern_name,
                    "match_count": len(matches),
                    "samples": matches[:5],  # First 5 matches
                    "file_path": str(self.current_file_path) if hasattr(self, 'current_file_path') else "unknown"
                })
        
        return results
    
    def _analyze_extracted_text(self, text: str, source: str) -> List[Dict[str, Any]]:
        """Analyze extracted text content"""
        results = []
        
        # Reuse string analysis patterns
        suspicious_patterns = {
            'base64': re.compile(r'[A-Za-z0-9+/]{20,}={0,2}'),
            'hex': re.compile(r'[0-9A-Fa-f]{20,}'),
            'flag': re.compile(r'flag\{.*?\}', re.IGNORECASE),
            'password': re.compile(r'password\s*[:=]\s*\S+', re.IGNORECASE),
            'url': re.compile(r'https?://[^\s]+'),
        }
        
        for pattern_name, pattern in suspicious_patterns.items():
            matches = pattern.findall(text)
            if matches:
                results.append({
                    "type": f"text_pattern_{pattern_name}",
                    "method": f"{source.lower()}_text_analysis",
                    "tool_name": "file_forensics",
                    "confidence": 0.7,
                    "details": f"Found {len(matches)} {pattern_name} patterns in {source} text",
                    "pattern_type": pattern_name,
                    "match_count": len(matches),
                    "samples": matches[:5],
                    "source": source,
                    "file_path": str(self.current_file_path) if hasattr(self, 'current_file_path') else "unknown"
                })
        
        return results
    
    def _yara_scan(self, file_path: Path) -> List[Dict[str, Any]]:
        """Scan file with YARA rules"""
        results = []
        
        if not self.yara_rules:
            return results
        
        try:
            matches = self.yara_rules.match(str(file_path))
            
            for match in matches:
                confidence = 0.7
                if 'steganography' in match.rule.lower():
                    confidence = 0.8
                elif 'embedded' in match.rule.lower():
                    confidence = 0.9
                
                match_details = {
                    "type": "yara_match",
                    "method": "yara_scan",
                    "tool_name": "yara",
                    "confidence": confidence,
                    "details": f"YARA rule matched: {match.rule}",
                    "rule_name": match.rule,
                    "rule_meta": dict(match.meta) if hasattr(match, 'meta') else {},
                    "file_path": str(file_path)
                }
                
                # Add string matches if available
                if hasattr(match, 'strings'):
                    string_matches = []
                    for string_match in match.strings:
                        string_matches.append({
                            'offset': string_match.offset,
                            'identifier': string_match.identifier,
                            'value': string_match.value.decode('utf-8', errors='ignore')[:100]
                        })
                    match_details['string_matches'] = string_matches[:10]  # Limit to 10
                
                results.append(match_details)
                
        except Exception as e:
            self.logger.error(f"YARA scan failed: {e}")
        
        return results
    
    def _overlay_analysis(self, file_path: Path) -> List[Dict[str, Any]]:
        """Detect and analyze file overlays (data appended to end of file)"""
        results = []
        
        # This is a simplified overlay detection
        # Real implementation would need format-specific knowledge
        
        try:
            with open(file_path, 'rb') as f:
                f.seek(0, 2)
                file_size = f.tell()
                
                # Read the last portion of the file
                overlay_size = min(1024, file_size)
                f.seek(-overlay_size, 2)
                overlay_data = f.read()
            
            # Check if overlay looks like it contains structured data
            if self._has_structured_data(overlay_data):
                results.append({
                    "type": "potential_overlay",
                    "method": "overlay_analysis",
                    "tool_name": "file_forensics",
                    "confidence": 0.6,
                    "details": "Potential overlay data detected at end of file",
                    "overlay_size": len(overlay_data),
                    "file_path": str(file_path)
                })
                
        except Exception as e:
            self.logger.debug(f"Overlay analysis failed: {e}")
        
        return results
    
    def _file_structure_analysis(self, file_path: Path) -> List[Dict[str, Any]]:
        """Analyze overall file structure"""
        results = []
        
        try:
            with open(file_path, 'rb') as f:
                file_size = f.seek(0, 2)
                
                # Analyze file in chunks to detect structure changes
                chunk_size = 4096
                chunk_entropies = []
                
                for offset in range(0, file_size, chunk_size):
                    f.seek(offset)
                    chunk = f.read(chunk_size)
                    
                    if len(chunk) > 100:
                        entropy = self._calculate_entropy(chunk)
                        chunk_entropies.append((offset, entropy))
            
            # Look for significant entropy changes
            if len(chunk_entropies) > 2:
                entropy_changes = []
                
                for i in range(1, len(chunk_entropies)):
                    prev_entropy = chunk_entropies[i-1][1]
                    curr_entropy = chunk_entropies[i][1]
                    entropy_diff = abs(curr_entropy - prev_entropy)
                    
                    if entropy_diff > 2.0:  # Significant change
                        entropy_changes.append({
                            'offset': chunk_entropies[i][0],
                            'prev_entropy': prev_entropy,
                            'curr_entropy': curr_entropy,
                            'difference': entropy_diff
                        })
                
                if entropy_changes:
                    results.append({
                        "type": "entropy_transitions",
                        "method": "file_structure_analysis",
                        "tool_name": "file_forensics",
                        "confidence": 0.6,
                        "details": f"Significant entropy changes detected: {len(entropy_changes)}",
                        "entropy_changes": entropy_changes[:5],  # First 5 changes
                        "change_count": len(entropy_changes),
                        "file_path": str(file_path)
                    })
                    
        except Exception as e:
            self.logger.debug(f"File structure analysis failed: {e}")
        
        return results
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file"""
        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except Exception:
            return ""
    
    def cleanup(self):
        """Cleanup resources"""
        if self.temp_dir.exists():
            import shutil
            shutil.rmtree(self.temp_dir)
        
        self.executor.shutdown(wait=True)