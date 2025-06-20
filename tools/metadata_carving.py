"""
Metadata Carving Tools - Extract and analyze metadata from various file formats
"""

import subprocess
import tempfile
import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor
import hashlib
import struct
import xml.etree.ElementTree as ET
from datetime import datetime
import mimetypes

class MetadataCarving:
    def __init__(self, config):
        self.config = config.file_forensics
        self.logger = logging.getLogger(__name__)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="metadata_"))
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Tool availability cache
        self.tools_available = self._check_tool_availability()
        
        # Metadata extraction patterns
        self.metadata_patterns = self._load_metadata_patterns()
        
        # File format handlers
        self.format_handlers = {
            'image/jpeg': self._extract_jpeg_metadata,
            'image/png': self._extract_png_metadata,
            'image/tiff': self._extract_tiff_metadata,
            'image/gif': self._extract_gif_metadata,
            'application/pdf': self._extract_pdf_metadata,
            'application/msword': self._extract_office_metadata,
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': self._extract_office_metadata,
            'audio/mpeg': self._extract_audio_metadata,
            'audio/wav': self._extract_audio_metadata,
            'video/mp4': self._extract_video_metadata,
            'video/avi': self._extract_video_metadata
        }
    
    def _check_tool_availability(self) -> Dict[str, bool]:
        """Check availability of external tools"""
        tools = {}
        tool_commands = {
            'exiftool': ['exiftool', '-ver'],
            'strings': ['strings', '--version'],
            'hexdump': ['hexdump', '-V'],
            'file': ['file', '--version']
        }
        
        for tool, cmd in tool_commands.items():
            try:
                result = subprocess.run(cmd, capture_output=True, timeout=5)
                tools[tool] = result.returncode == 0
            except:
                tools[tool] = False
        
        return tools
    
    def _load_metadata_patterns(self) -> Dict[str, re.Pattern]:
        """Load regex patterns for metadata detection"""
        return {
            'xml_metadata': re.compile(rb'<\?xml[^>]*\?>.*?</[^>]+>', re.DOTALL),
            'json_metadata': re.compile(rb'\{[^}]*"[^"]*":[^}]*\}'),
            'exif_marker': re.compile(rb'\xff\xe1[^\x00]*Exif\x00\x00'),
            'iptc_marker': re.compile(rb'Photoshop 3.0'),
            'xmp_marker': re.compile(rb'<x:xmpmeta'),
            'gps_coordinates': re.compile(rb'GPS[^}]*(?:latitude|longitude)[^}]*'),
            'timestamp_pattern': re.compile(rb'\d{4}[:-]\d{2}[:-]\d{2}[T ]\d{2}:\d{2}:\d{2}'),
            'email_pattern': re.compile(rb'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'),
            'file_path_pattern': re.compile(rb'[A-Za-z]:\\[^<>:"|?*\x00-\x1f]*|/[^<>:"|?*\x00-\x1f]*')
        }
    
    def execute_method(self, method: str, file_path: Path) -> List[Dict[str, Any]]:
        """Execute a specific metadata carving method"""
        method_map = {
            'metadata_extraction': self._comprehensive_metadata_extraction,
            'bulk_extractor': self._run_bulk_extractor,
            'exif_analysis': self._detailed_exif_analysis,
            'hidden_metadata': self._search_hidden_metadata,
            'timestamp_analysis': self._analyze_timestamps,
            'embedded_files': self._extract_embedded_files,
            'metadata_anomalies': self._detect_metadata_anomalies,
            'geolocation_data': self._extract_geolocation_data,
            'software_signatures': self._identify_software_signatures,
            'user_traces': self._find_user_traces
        }
        
        if method not in method_map:
            self.logger.warning(f"Unknown metadata method: {method}")
            return []
        
        try:
            start_time = time.time()
            results = method_map[method](file_path)
            
            # Add timing and metadata
            for result in results:
                result.update({
                    "analysis_time": time.time() - start_time,
                    "file_path": str(file_path),
                    "method": method,
                    "tool_name": "metadata_carving"
                })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Metadata method {method} failed: {e}")
            return [{
                "type": "metadata_analysis_error",
                "method": method,
                "tool_name": "metadata_carving",
                "error": str(e),
                "confidence": 0.0,
                "file_path": str(file_path)
            }]
    
    def _comprehensive_metadata_extraction(self, file_path: Path) -> List[Dict[str, Any]]:
        """Comprehensive metadata extraction using multiple tools and methods"""
        results = []
        
        try:
            # Detect file type
            mime_type, _ = mimetypes.guess_type(str(file_path))
            if not mime_type:
                with open(file_path, 'rb') as f:
                    header = f.read(512)
                mime_type = self._detect_mime_from_header(header)
            
            # Use format-specific handler if available
            if mime_type in self.format_handlers:
                format_results = self.format_handlers[mime_type](file_path)
                results.extend(format_results)
            
            # Use exiftool if available
            if self.tools_available.get('exiftool'):
                exiftool_results = self._extract_exiftool_metadata(file_path)
                results.extend(exiftool_results)
            
            # Use strings extraction
            strings_results = self._extract_metadata_strings(file_path)
            results.extend(strings_results)
            
            # Search for embedded files
            embedded_results = self._find_embedded_files(file_path)
            results.extend(embedded_results)
            
            # Analyze file structure
            structure_results = self._analyze_file_structure(file_path)
            results.extend(structure_results)
            
        except Exception as e:
            self.logger.error(f"Comprehensive metadata extraction failed: {e}")
        
        return results
    
    def _detect_mime_from_header(self, header: bytes) -> str:
        """Detect MIME type from file header"""
        signatures = {
            b'\xFF\xD8\xFF': 'image/jpeg',
            b'\x89PNG\r\n\x1a\n': 'image/png',
            b'GIF8': 'image/gif',
            b'II*\x00': 'image/tiff',
            b'MM\x00*': 'image/tiff',
            b'%PDF': 'application/pdf',
            b'PK\x03\x04': 'application/zip',
            b'RIFF': 'audio/wav',
            b'ID3': 'audio/mpeg'
        }
        
        for sig, mime_type in signatures.items():
            if header.startswith(sig):
                return mime_type
        
        return 'application/octet-stream'
    
    def _extract_jpeg_metadata(self, file_path: Path) -> List[Dict[str, Any]]:
        """Extract JPEG-specific metadata"""
        results = []
        
        try:
            # Try PIL first
            try:
                from PIL import Image
                from PIL.ExifTags import TAGS
                
                with Image.open(file_path) as img:
                    if hasattr(img, '_getexif'):
                        exifdata = img._getexif()
                        if exifdata:
                            for tag_id, value in exifdata.items():
                                tag = TAGS.get(tag_id, tag_id)
                                results.append({
                                    "type": "jpeg_exif",
                                    "tag": str(tag),
                                    "value": str(value)[:200],
                                    "confidence": 0.8,
                                    "details": f"JPEG EXIF tag: {tag}"
                                })
            except ImportError:
                pass
            
            # Manual JPEG parsing for segments
            with open(file_path, 'rb') as f:
                data = f.read()
            
            # Find JPEG segments
            pos = 0
            while pos < len(data) - 1:
                if data[pos] == 0xFF and data[pos + 1] != 0xFF:
                    marker = data[pos + 1]
                    if marker == 0xE1:  # APP1 (EXIF)
                        results.append({
                            "type": "jpeg_app1_segment",
                            "marker": f"0xFF{marker:02X}",
                            "confidence": 0.7,
                            "details": "JPEG APP1 segment found (likely EXIF)"
                        })
                    elif marker in [0xE0, 0xE2, 0xE3, 0xE4]:  # Other APP segments
                        results.append({
                            "type": "jpeg_app_segment",
                            "marker": f"0xFF{marker:02X}",
                            "confidence": 0.5,
                            "details": f"JPEG APP{marker-0xE0} segment found"
                        })
                pos += 1
        
        except Exception as e:
            self.logger.debug(f"JPEG metadata extraction failed: {e}")
        
        return results
    
    def _extract_png_metadata(self, file_path: Path) -> List[Dict[str, Any]]:
        """Extract PNG-specific metadata"""
        results = []
        
        try:
            # Try PIL first
            try:
                from PIL import Image
                
                with Image.open(file_path) as img:
                    # PNG text chunks
                    if hasattr(img, 'text'):
                        for key, value in img.text.items():
                            results.append({
                                "type": "png_text_chunk",
                                "key": key,
                                "value": str(value)[:200],
                                "confidence": 0.7,
                                "details": f"PNG text chunk: {key}"
                            })
                    
                    # PNG info
                    if hasattr(img, 'info'):
                        for key, value in img.info.items():
                            if key not in img.text:  # Avoid duplicates
                                results.append({
                                    "type": "png_info",
                                    "key": key,
                                    "value": str(value)[:200],
                                    "confidence": 0.6,
                                    "details": f"PNG info: {key}"
                                })
            except ImportError:
                pass
            
            # Manual PNG chunk parsing
            with open(file_path, 'rb') as f:
                data = f.read()
            
            # Skip PNG signature
            pos = 8
            while pos < len(data) - 8:
                try:
                    chunk_length = struct.unpack('>I', data[pos:pos+4])[0]
                    chunk_type = data[pos+4:pos+8].decode('ascii', errors='ignore')
                    
                    if chunk_type in ['tEXt', 'iTXt', 'zTXt']:
                        results.append({
                            "type": "png_text_chunk_raw",
                            "chunk_type": chunk_type,
                            "length": chunk_length,
                            "confidence": 0.8,
                            "details": f"PNG {chunk_type} chunk found"
                        })
                    
                    pos += 8 + chunk_length + 4  # Skip chunk data and CRC
                except:
                    break
        
        except Exception as e:
            self.logger.debug(f"PNG metadata extraction failed: {e}")
        
        return results
    
    def _extract_tiff_metadata(self, file_path: Path) -> List[Dict[str, Any]]:
        """Extract TIFF-specific metadata"""
        results = []
        
        try:
            # Try PIL first
            try:
                from PIL import Image
                from PIL.ExifTags import TAGS
                
                with Image.open(file_path) as img:
                    # TIFF tags
                    if hasattr(img, 'tag_v2'):
                        for tag_id, value in img.tag_v2.items():
                            tag_name = TAGS.get(tag_id, f"Tag{tag_id}")
                            results.append({
                                "type": "tiff_tag",
                                "tag_id": tag_id,
                                "tag_name": tag_name,
                                "value": str(value)[:200],
                                "confidence": 0.8,
                                "details": f"TIFF tag: {tag_name}"
                            })
            except ImportError:
                pass
            
            # Manual TIFF parsing
            with open(file_path, 'rb') as f:
                header = f.read(8)
            
            if header[:2] in [b'II', b'MM']:
                endian = '<' if header[:2] == b'II' else '>'
                magic = struct.unpack(endian + 'H', header[2:4])[0]
                
                if magic == 42:
                    results.append({
                        "type": "tiff_header",
                        "endian": "little" if header[:2] == b'II' else "big",
                        "magic": magic,
                        "confidence": 0.9,
                        "details": "Valid TIFF header found"
                    })
        
        except Exception as e:
            self.logger.debug(f"TIFF metadata extraction failed: {e}")
        
        return results
    
    def _extract_gif_metadata(self, file_path: Path) -> List[Dict[str, Any]]:
        """Extract GIF-specific metadata"""
        results = []
        
        try:
            with open(file_path, 'rb') as f:
                header = f.read(13)
            
            if header.startswith(b'GIF'):
                version = header[3:6].decode('ascii')
                width = struct.unpack('<H', header[6:8])[0]
                height = struct.unpack('<H', header[8:10])[0]
                
                results.append({
                    "type": "gif_header",
                    "version": version,
                    "width": width,
                    "height": height,
                    "confidence": 0.9,
                    "details": f"GIF {version} header"
                })
        
        except Exception as e:
            self.logger.debug(f"GIF metadata extraction failed: {e}")
        
        return results
    
    def _extract_pdf_metadata(self, file_path: Path) -> List[Dict[str, Any]]:
        """Extract PDF-specific metadata"""
        results = []
        
        try:
            # Manual PDF metadata parsing
            with open(file_path, 'rb') as f:
                content = f.read(8192)  # Read first 8KB
            
            # Look for PDF info dictionary
            info_match = re.search(rb'/Info\s*(\d+)\s+\d+\s+R', content)
            if info_match:
                results.append({
                    "type": "pdf_info_dict",
                    "object_ref": info_match.group(1).decode(),
                    "confidence": 0.7,
                    "details": "PDF contains Info dictionary"
                })
            
            # Look for metadata streams
            metadata_match = re.search(rb'/Metadata\s*(\d+)\s+\d+\s+R', content)
            if metadata_match:
                results.append({
                    "type": "pdf_metadata_stream",
                    "object_ref": metadata_match.group(1).decode(),
                    "confidence": 0.8,
                    "details": "PDF contains XMP metadata stream"
                })
            
            # Look for creator/producer
            for field in [b'/Creator', b'/Producer', b'/Author', b'/Title']:
                if field in content:
                    results.append({
                        "type": "pdf_metadata_field",
                        "field": field.decode(),
                        "confidence": 0.6,
                        "details": f"PDF metadata field: {field.decode()}"
                    })
        
        except Exception as e:
            self.logger.debug(f"PDF metadata extraction failed: {e}")
        
        return results
    
    def _extract_office_metadata(self, file_path: Path) -> List[Dict[str, Any]]:
        """Extract Microsoft Office document metadata"""
        results = []
        
        try:
            # Office documents are ZIP files
            import zipfile
            
            with zipfile.ZipFile(file_path, 'r') as zf:
                # Look for document properties
                if 'docProps/core.xml' in zf.namelist():
                    core_xml = zf.read('docProps/core.xml')
                    results.append({
                        "type": "office_core_properties",
                        "size": len(core_xml),
                        "confidence": 0.8,
                        "details": "Office core properties found"
                    })
                
                if 'docProps/app.xml' in zf.namelist():
                    app_xml = zf.read('docProps/app.xml')
                    results.append({
                        "type": "office_app_properties",
                        "size": len(app_xml),
                        "confidence": 0.8,
                        "details": "Office application properties found"
                    })
        
        except Exception as e:
            self.logger.debug(f"Office metadata extraction failed: {e}")
        
        return results
    
    def _extract_audio_metadata(self, file_path: Path) -> List[Dict[str, Any]]:
        """Extract audio metadata"""
        results = []
        
        try:
            # Look for ID3 tags in MP3 files
            with open(file_path, 'rb') as f:
                header = f.read(10)
            
            if header.startswith(b'ID3'):
                version = f"2.{header[3]}.{header[4]}"
                size = struct.unpack('>I', b'\x00' + header[6:9])[0]
                
                results.append({
                    "type": "id3_tag",
                    "version": version,
                    "size": size,
                    "confidence": 0.9,
                    "details": f"ID3 v{version} tag found"
                })
        
        except Exception as e:
            self.logger.debug(f"Audio metadata extraction failed: {e}")
        
        return results
    
    def _extract_video_metadata(self, file_path: Path) -> List[Dict[str, Any]]:
        """Extract video metadata"""
        results = []
        
        try:
            # Basic video file analysis
            with open(file_path, 'rb') as f:
                header = f.read(32)
            
            if header[4:8] == b'ftyp':  # MP4
                results.append({
                    "type": "mp4_header",
                    "brand": header[8:12].decode('ascii', errors='ignore'),
                    "confidence": 0.9,
                    "details": "MP4 file type header found"
                })
            elif header.startswith(b'RIFF') and b'AVI ' in header:
                results.append({
                    "type": "avi_header",
                    "confidence": 0.9,
                    "details": "AVI RIFF header found"
                })
        
        except Exception as e:
            self.logger.debug(f"Video metadata extraction failed: {e}")
        
        return results
    
    def _extract_exiftool_metadata(self, file_path: Path) -> List[Dict[str, Any]]:
        """Extract metadata using exiftool"""
        results = []
        
        if not self.tools_available.get('exiftool'):
            return results
        
        try:
            cmd = ['exiftool', '-json', '-a', '-G', '-s', str(file_path)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                metadata = json.loads(result.stdout)[0]
                
                # Process metadata fields
                for key, value in metadata.items():
                    if key != 'SourceFile' and value:
                        results.append({
                            "type": "exiftool_metadata",
                            "field": key,
                            "value": str(value)[:500],  # Limit value length
                            "confidence": 0.8,
                            "details": f"ExifTool metadata field: {key}"
                        })
                
                # Check for unusual or suspicious fields
                suspicious_fields = self._identify_suspicious_exif_fields(metadata)
                results.extend(suspicious_fields)
            
        except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception) as e:
            self.logger.debug(f"ExifTool extraction failed: {e}")
        
        return results
    
    def _identify_suspicious_exif_fields(self, metadata: Dict) -> List[Dict[str, Any]]:
        """Identify potentially suspicious EXIF fields"""
        results = []
        
        suspicious_patterns = [
            'UserComment', 'ImageDescription', 'Artist', 'Copyright',
            'Software', 'ProcessingSoftware', 'OriginalRawFileName'
        ]
        
        for field, value in metadata.items():
            if any(pattern in field for pattern in suspicious_patterns):
                if len(str(value)) > 100:  # Unusually long values
                    results.append({
                        "type": "suspicious_exif_field",
                        "field": field,
                        "value_length": len(str(value)),
                        "confidence": 0.6,
                        "details": f"Unusually long EXIF field: {field}"
                    })
        
        return results
    
    def _extract_metadata_strings(self, file_path: Path) -> List[Dict[str, Any]]:
        """Extract metadata using strings command"""
        results = []
        
        if not self.tools_available.get('strings'):
            return results
        
        try:
            cmd = ['strings', '-n', '10', str(file_path)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                strings_output = result.stdout
                
                # Look for metadata patterns
                for pattern_name, pattern in self.metadata_patterns.items():
                    matches = pattern.findall(strings_output.encode())
                    for match in matches[:5]:  # Limit results
                        results.append({
                            "type": "metadata_string",
                            "pattern": pattern_name,
                            "match": match.decode('utf-8', errors='ignore')[:200],
                            "confidence": 0.5,
                            "details": f"Metadata pattern found: {pattern_name}"
                        })
        
        except Exception as e:
            self.logger.debug(f"Strings extraction failed: {e}")
        
        return results
    
    def _find_embedded_files(self, file_path: Path) -> List[Dict[str, Any]]:
        """Find embedded files within the target file"""
        results = []
        
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
            
            # Look for file signatures
            signatures = {
                b'\xFF\xD8\xFF': 'JPEG',
                b'\x89PNG': 'PNG',
                b'%PDF': 'PDF',
                b'PK\x03\x04': 'ZIP',
                b'RIFF': 'RIFF'
            }
            
            for sig, file_type in signatures.items():
                pos = data.find(sig)
                if pos > 0:  # Found signature not at beginning
                    results.append({
                        "type": "embedded_file_signature",
                        "file_type": file_type,
                        "offset": pos,
                        "confidence": 0.7,
                        "details": f"Embedded {file_type} signature at offset {pos}"
                    })
        
        except Exception as e:
            self.logger.debug(f"Embedded file search failed: {e}")
        
        return results
    
    def _analyze_file_structure(self, file_path: Path) -> List[Dict[str, Any]]:
        """Analyze file structure for anomalies"""
        results = []
        
        try:
            file_size = file_path.stat().st_size
            
            # Check for unusual file size patterns
            if file_size % 1024 == 0:
                results.append({
                    "type": "file_size_anomaly",
                    "size": file_size,
                    "anomaly": "exact_kb_size",
                    "confidence": 0.3,
                    "details": f"File size is exactly {file_size // 1024}KB"
                })
        
        except Exception as e:
            self.logger.debug(f"File structure analysis failed: {e}")
        
        return results
    
    # Stub methods for other analysis types
    def _run_bulk_extractor(self, file_path: Path) -> List[Dict[str, Any]]:
        """Run bulk_extractor for metadata extraction"""
        # TODO: Implement bulk_extractor integration
        return []
    
    def _detailed_exif_analysis(self, file_path: Path) -> List[Dict[str, Any]]:
        """Perform detailed EXIF analysis"""
        return self._extract_exiftool_metadata(file_path)
    
    def _search_hidden_metadata(self, file_path: Path) -> List[Dict[str, Any]]:
        """Search for hidden metadata"""
        return self._extract_metadata_strings(file_path)
    
    def _analyze_timestamps(self, file_path: Path) -> List[Dict[str, Any]]:
        """Analyze timestamp metadata"""
        results = []
        
        # File system timestamps
        stat = file_path.stat()
        results.append({
            "type": "filesystem_timestamp",
            "created": stat.st_ctime,
            "modified": stat.st_mtime,
            "accessed": stat.st_atime,
            "confidence": 0.9,
            "details": "Filesystem timestamps"
        })
        
        return results
    
    def _extract_embedded_files(self, file_path: Path) -> List[Dict[str, Any]]:
        """Extract embedded files"""
        return self._find_embedded_files(file_path)
    
    def _detect_metadata_anomalies(self, file_path: Path) -> List[Dict[str, Any]]:
        """Detect metadata anomalies"""
        return self._analyze_file_structure(file_path)
    
    def _extract_geolocation_data(self, file_path: Path) -> List[Dict[str, Any]]:
        """Extract geolocation data from metadata"""
        results = []
        
        if self.tools_available.get('exiftool'):
            try:
                cmd = ['exiftool', '-json', '-gps:all', str(file_path)]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    metadata = json.loads(result.stdout)[0]
                    
                    gps_fields = [k for k in metadata.keys() if 'GPS' in k.upper()]
                    if gps_fields:
                        results.append({
                            "type": "gps_metadata",
                            "fields": gps_fields,
                            "confidence": 0.9,
                            "details": f"GPS metadata found: {', '.join(gps_fields)}"
                        })
            except:
                pass
        
        return results
    
    def _identify_software_signatures(self, file_path: Path) -> List[Dict[str, Any]]:
        """Identify software signatures in metadata"""
        results = []
        
        # Look for software signatures in strings
        if self.tools_available.get('strings'):
            try:
                cmd = ['strings', str(file_path)]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    software_patterns = [
                        'Photoshop', 'GIMP', 'Paint.NET', 'Adobe', 'Microsoft',
                        'Canon', 'Nikon', 'Sony', 'Apple', 'Google'
                    ]
                    
                    for pattern in software_patterns:
                        if pattern in result.stdout:
                            results.append({
                                "type": "software_signature",
                                "software": pattern,
                                "confidence": 0.6,
                                "details": f"Software signature found: {pattern}"
                            })
            except:
                pass
        
        return results
    
    def _find_user_traces(self, file_path: Path) -> List[Dict[str, Any]]:
        """Find user traces in metadata"""
        results = []
        
        # Look for username patterns in strings
        if self.tools_available.get('strings'):
            try:
                cmd = ['strings', str(file_path)]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    # Look for common username patterns
                    username_patterns = [
                        r'C:\\Users\\([^\\]+)',
                        r'/home/([^/]+)',
                        r'/Users/([^/]+)',
                        r'Created by ([^\\n]+)'
                    ]
                    
                    for pattern in username_patterns:
                        matches = re.findall(pattern, result.stdout)
                        for match in matches[:3]:  # Limit results
                            results.append({
                                "type": "user_trace",
                                "username": match,
                                "confidence": 0.7,
                                "details": f"User trace found: {match}"
                            })
            except:
                pass
        
        return results
    
    def __del__(self):
        """Cleanup temporary directory"""
        try:
            import shutil
            if hasattr(self, 'temp_dir') and self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
        except:
            pass