"""
Classic Steganography Tools Integration
All major CLI steganography detection and extraction tools
"""

import subprocess
import asyncio
import tempfile
import shutil
import re
import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import hashlib
import magic
from concurrent.futures import ThreadPoolExecutor
import time

class ClassicStegoTools:
    def __init__(self, config):
        self.config = config.classic_stego
        self.logger = logging.getLogger(__name__)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="steg_"))
        self.executor = ThreadPoolExecutor(max_workers=8)
        
        # Tool availability cache
        self._tool_availability = {}
        self._check_tool_availability()
    
    def _check_tool_availability(self):
        """Check which tools are available on the system"""
        tools = {
            'steghide': 'steghide --version',
            'outguess': 'outguess',
            'zsteg': 'zsteg --help',
            'stegseek': 'stegseek --version',
            'stegoveritas': 'stegoveritas --version',
            'stegcracker': 'stegcracker --help',
            'exiftool': 'exiftool -ver',
            'binwalk': 'binwalk --help',
            'foremost': 'foremost -V',
            'strings': 'strings --version',
            'magika': 'magika --version',
            'tesseract': 'tesseract --version',
            'sox': 'sox --version',
            'ffmpeg': 'ffmpeg -version'
        }
        
        for tool, check_cmd in tools.items():
            try:
                result = subprocess.run(
                    check_cmd.split(), 
                    capture_output=True, 
                    text=True, 
                    timeout=10
                )
                self._tool_availability[tool] = result.returncode == 0
                if self._tool_availability[tool]:
                    self.logger.debug(f"Tool {tool} is available")
                else:
                    self.logger.warning(f"Tool {tool} is not available")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                self._tool_availability[tool] = False
                self.logger.warning(f"Tool {tool} is not available")
    
    def execute_method(self, method: str, file_path: Path) -> List[Dict[str, Any]]:
        """Execute a specific steganography method"""
        method_map = {
            'steghide': self._run_steghide,
            'outguess': self._run_outguess,
            'zsteg': self._run_zsteg,
            'stegseek': self._run_stegseek,
            'stegoveritas': self._run_stegoveritas,
            'stegcracker': self._run_stegcracker,
            'exif_analysis': self._run_exiftool,
            'binwalk': self._run_binwalk,
            'foremost': self._run_foremost,
            'strings': self._run_strings,
            'hexdump_analysis': self._run_hexdump_analysis,
            'magika': self._run_magika,
            'tesseract_ocr': self._run_tesseract,
            'sox_analysis': self._run_sox,
            'ffmpeg_analysis': self._run_ffmpeg
        }
        
        if method not in method_map:
            raise ValueError(f"Unknown method: {method}")
        
        try:
            return method_map[method](file_path)
        except Exception as e:
            self.logger.error(f"Method {method} failed for {file_path}: {e}")
            return [{
                "type": "error",
                "method": method,
                "tool_name": "classic_stego",
                "confidence": 0.0,
                "details": f"Tool execution failed: {str(e)}",
                "file_path": str(file_path)
            }]
    
    def _run_steghide(self, file_path: Path) -> List[Dict[str, Any]]:
        """Run steghide analysis"""
        if not self._tool_availability.get('steghide', False):
            return []
        
        results = []
        
        # Try to extract without password
        try:
            cmd = ['steghide', 'extract', '-sf', str(file_path), '-p', '']
            with tempfile.TemporaryDirectory() as temp_dir:
                result = subprocess.run(
                    cmd, 
                    cwd=temp_dir,
                    capture_output=True, 
                    text=True, 
                    timeout=60
                )
                
                if result.returncode == 0:
                    # Check for extracted files
                    extracted_files = list(Path(temp_dir).iterdir())
                    if extracted_files:
                        for extracted_file in extracted_files:
                            if extracted_file.name != file_path.name:
                                with open(extracted_file, 'rb') as f:
                                    content = f.read()
                                
                                results.append({
                                    "type": "steganography",
                                    "method": "steghide",
                                    "tool_name": "steghide",
                                    "confidence": 0.9,
                                    "details": f"Extracted hidden file: {extracted_file.name}",
                                    "extracted_file": extracted_file.name,
                                    "extracted_size": len(content),
                                    "extracted_content": content[:1024].decode('utf-8', errors='ignore'),
                                    "file_path": str(file_path),
                                    "raw_output": result.stdout
                                })
                
                # Try with common passwords
                if not results and self.config.steghide_wordlist and Path(self.config.steghide_wordlist).exists():
                    results.extend(self._steghide_bruteforce(file_path))
                
        except subprocess.TimeoutExpired:
            self.logger.warning(f"Steghide timeout for {file_path}")
        except Exception as e:
            self.logger.error(f"Steghide error for {file_path}: {e}")
        
        return results
    
    def _steghide_bruteforce(self, file_path: Path, max_passwords: int = 100) -> List[Dict[str, Any]]:
        """Bruteforce steghide with common passwords"""
        results = []
        
        try:
            with open(self.config.steghide_wordlist, 'r', encoding='utf-8', errors='ignore') as f:
                passwords = [line.strip() for line in f.readlines()[:max_passwords]]
        except Exception:
            return results
        
        for password in passwords:
            try:
                cmd = ['steghide', 'extract', '-sf', str(file_path), '-p', password]
                with tempfile.TemporaryDirectory() as temp_dir:
                    result = subprocess.run(
                        cmd,
                        cwd=temp_dir,
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    
                    if result.returncode == 0:
                        extracted_files = list(Path(temp_dir).iterdir())
                        if extracted_files:
                            for extracted_file in extracted_files:
                                if extracted_file.name != file_path.name:
                                    with open(extracted_file, 'rb') as f:
                                        content = f.read()
                                    
                                    results.append({
                                        "type": "steganography",
                                        "method": "steghide_bruteforce",
                                        "tool_name": "steghide",
                                        "confidence": 0.95,
                                        "details": f"Extracted with password: {password}",
                                        "password": password,
                                        "extracted_file": extracted_file.name,
                                        "extracted_size": len(content),
                                        "extracted_content": content[:1024].decode('utf-8', errors='ignore'),
                                        "file_path": str(file_path)
                                    })
                            break  # Found something, stop trying
                            
            except subprocess.TimeoutExpired:
                continue
            except Exception:
                continue
        
        return results
    
    def _run_outguess(self, file_path: Path) -> List[Dict[str, Any]]:
        """Run outguess analysis"""
        if not self._tool_availability.get('outguess', False):
            return []
        
        results = []
        
        try:
            # Try to extract data
            with tempfile.NamedTemporaryFile(delete=False) as output_file:
                cmd = ['outguess', '-r', str(file_path), output_file.name]
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if result.returncode == 0:
                    # Check if anything was extracted
                    if os.path.getsize(output_file.name) > 0:
                        with open(output_file.name, 'rb') as f:
                            content = f.read()
                        
                        results.append({
                            "type": "steganography",
                            "method": "outguess",
                            "tool_name": "outguess",
                            "confidence": 0.8,
                            "details": "OutGuess hidden data detected and extracted",
                            "extracted_size": len(content),
                            "extracted_content": content[:1024].decode('utf-8', errors='ignore'),
                            "file_path": str(file_path),
                            "raw_output": result.stdout
                        })
                
                os.unlink(output_file.name)
                
        except subprocess.TimeoutExpired:
            self.logger.warning(f"OutGuess timeout for {file_path}")
        except Exception as e:
            self.logger.error(f"OutGuess error for {file_path}: {e}")
        
        return results
    
    def _run_zsteg(self, file_path: Path) -> List[Dict[str, Any]]:
        """Run zsteg analysis"""
        if not self._tool_availability.get('zsteg', False):
            return []
        
        results = []
        
        try:
            cmd = ['zsteg', '--all', str(file_path)]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.stdout:
                # Parse zsteg output
                lines = result.stdout.split('\n')
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('imagedata'):
                        # Extract method and content
                        parts = line.split(' .. ')
                        if len(parts) >= 2:
                            method = parts[0].strip()
                            content = ' .. '.join(parts[1:]).strip()
                            
                            # Determine confidence based on content type
                            confidence = 0.3
                            if any(keyword in content.lower() for keyword in ['flag', 'password', 'secret', 'hidden']):
                                confidence = 0.8
                            elif len(content) > 20 and content.isprintable():
                                confidence = 0.6
                            
                            results.append({
                                "type": "steganography",
                                "method": "zsteg",
                                "tool_name": "zsteg",
                                "confidence": confidence,
                                "details": f"zsteg method: {method}",
                                "extraction_method": method,
                                "extracted_content": content[:1024],
                                "file_path": str(file_path)
                            })
                
        except subprocess.TimeoutExpired:
            self.logger.warning(f"zsteg timeout for {file_path}")
        except Exception as e:
            self.logger.error(f"zsteg error for {file_path}: {e}")
        
        return results
    
    def _run_stegseek(self, file_path: Path) -> List[Dict[str, Any]]:
        """Run stegseek analysis"""
        if not self._tool_availability.get('stegseek', False):
            return []
        
        results = []
        
        if not self.config.stegseek_wordlist or not Path(self.config.stegseek_wordlist).exists():
            return results
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                output_file = Path(temp_dir) / "extracted.txt"
                cmd = [
                    'stegseek', 
                    str(file_path), 
                    self.config.stegseek_wordlist,
                    '-xf', str(output_file)
                ]
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minutes
                )
                
                if result.returncode == 0 and output_file.exists():
                    with open(output_file, 'rb') as f:
                        content = f.read()
                    
                    # Extract password from stderr
                    password = None
                    if result.stderr:
                        password_match = re.search(r'Found passphrase: "([^"]*)"', result.stderr)
                        if password_match:
                            password = password_match.group(1)
                    
                    results.append({
                        "type": "steganography",
                        "method": "stegseek",
                        "tool_name": "stegseek",
                        "confidence": 0.95,
                        "details": f"Stegseek cracked steghide password: {password}",
                        "password": password,
                        "extracted_size": len(content),
                        "extracted_content": content[:1024].decode('utf-8', errors='ignore'),
                        "file_path": str(file_path),
                        "raw_output": result.stdout
                    })
                
        except subprocess.TimeoutExpired:
            self.logger.warning(f"Stegseek timeout for {file_path}")
        except Exception as e:
            self.logger.error(f"Stegseek error for {file_path}: {e}")
        
        return results
    
    def _run_stegoveritas(self, file_path: Path) -> List[Dict[str, Any]]:
        """Run stegoveritas analysis"""
        if not self._tool_availability.get('stegoveritas', False):
            return []
        
        results = []
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                cmd = ['stegoveritas', str(file_path), '-out', temp_dir]
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=180
                )
                
                # Check for generated files
                output_dir = Path(temp_dir)
                generated_files = list(output_dir.rglob('*'))
                
                for generated_file in generated_files:
                    if generated_file.is_file() and generated_file.stat().st_size > 0:
                        file_type = magic.from_file(str(generated_file), mime=True)
                        
                        results.append({
                            "type": "steganography",
                            "method": "stegoveritas",
                            "tool_name": "stegoveritas",
                            "confidence": 0.7,
                            "details": f"StegOveritas generated: {generated_file.name}",
                            "generated_file": generated_file.name,
                            "file_type": file_type,
                            "file_size": generated_file.stat().st_size,
                            "file_path": str(file_path)
                        })
                
                # Parse text output for findings
                if result.stdout:
                    if "Found something" in result.stdout or "LSB" in result.stdout:
                        results.append({
                            "type": "steganography",
                            "method": "stegoveritas_analysis",
                            "tool_name": "stegoveritas",
                            "confidence": 0.6,
                            "details": "StegOveritas detected potential steganography",
                            "file_path": str(file_path),
                            "raw_output": result.stdout[:2048]
                        })
                
        except subprocess.TimeoutExpired:
            self.logger.warning(f"StegOveritas timeout for {file_path}")
        except Exception as e:
            self.logger.error(f"StegOveritas error for {file_path}: {e}")
        
        return results
    
    def _run_binwalk(self, file_path: Path) -> List[Dict[str, Any]]:
        """Run binwalk analysis"""
        if not self._tool_availability.get('binwalk', False):
            return []
        
        results = []
        
        try:
            # Run binwalk analysis
            cmd = ['binwalk', '--entropy', '-B', str(file_path)]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.stdout:
                # Parse binwalk output
                lines = result.stdout.split('\n')
                for line in lines:
                    if line.strip() and not line.startswith('DECIMAL'):
                        parts = line.split(None, 2)
                        if len(parts) >= 3:
                            offset = parts[0]
                            type_info = parts[2] if len(parts) > 2 else ''
                            
                            confidence = 0.4
                            if any(keyword in type_info.lower() for keyword in 
                                   ['encrypted', 'compressed', 'archive', 'filesystem']):
                                confidence = 0.7
                            
                            results.append({
                                "type": "file_structure",
                                "method": "binwalk",
                                "tool_name": "binwalk",
                                "confidence": confidence,
                                "details": f"Found at offset {offset}: {type_info}",
                                "offset": offset,
                                "structure_type": type_info,
                                "file_path": str(file_path)
                            })
            
            # Run extraction if enabled
            if self.config.binwalk_extract:
                extract_results = self._binwalk_extract(file_path)
                results.extend(extract_results)
                
        except subprocess.TimeoutExpired:
            self.logger.warning(f"Binwalk timeout for {file_path}")
        except Exception as e:
            self.logger.error(f"Binwalk error for {file_path}: {e}")
        
        return results
    
    def _binwalk_extract(self, file_path: Path) -> List[Dict[str, Any]]:
        """Extract files using binwalk"""
        results = []
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                cmd = ['binwalk', '--extract', '--directory', temp_dir, str(file_path)]
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                
                # Check for extracted files
                extract_dir = Path(temp_dir)
                extracted_files = list(extract_dir.rglob('*'))
                
                for extracted_file in extracted_files:
                    if extracted_file.is_file() and extracted_file.stat().st_size > 0:
                        file_type = magic.from_file(str(extracted_file), mime=True)
                        
                        results.append({
                            "type": "extracted_file",
                            "method": "binwalk_extract",
                            "tool_name": "binwalk",
                            "confidence": 0.8,
                            "details": f"Extracted file: {extracted_file.name}",
                            "extracted_file": str(extracted_file.relative_to(extract_dir)),
                            "file_type": file_type,
                            "file_size": extracted_file.stat().st_size,
                            "file_path": str(file_path)
                        })
                
        except subprocess.TimeoutExpired:
            self.logger.warning(f"Binwalk extract timeout for {file_path}")
        except Exception as e:
            self.logger.error(f"Binwalk extract error for {file_path}: {e}")
        
        return results
    
    def _run_foremost(self, file_path: Path) -> List[Dict[str, Any]]:
        """Run foremost file carving"""
        if not self._tool_availability.get('foremost', False):
            return []
        
        results = []
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                cmd = ['foremost', '-i', str(file_path), '-o', temp_dir]
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                
                # Check audit file for results
                audit_file = Path(temp_dir) / 'audit.txt'
                if audit_file.exists():
                    with open(audit_file, 'r') as f:
                        audit_content = f.read()
                    
                    # Parse audit for file counts
                    file_counts = {}
                    for line in audit_content.split('\n'):
                        if ':' in line and 'FILES EXTRACTED' in line:
                            parts = line.split(':')
                            if len(parts) >= 2:
                                file_type = parts[0].strip()
                                count = parts[1].split()[0].strip()
                                try:
                                    file_counts[file_type] = int(count)
                                except ValueError:
                                    pass
                    
                    # Check for actual extracted files
                    extracted_files = []
                    for subdir in Path(temp_dir).iterdir():
                        if subdir.is_dir():
                            extracted_files.extend(list(subdir.glob('*')))
                    
                    if extracted_files:
                        results.append({
                            "type": "file_carving",
                            "method": "foremost",
                            "tool_name": "foremost",
                            "confidence": 0.8,
                            "details": f"Foremost extracted {len(extracted_files)} files",
                            "extracted_count": len(extracted_files),
                            "file_types": file_counts,
                            "file_path": str(file_path),
                            "audit_content": audit_content[:1024]
                        })
                
        except subprocess.TimeoutExpired:
            self.logger.warning(f"Foremost timeout for {file_path}")
        except Exception as e:
            self.logger.error(f"Foremost error for {file_path}: {e}")
        
        return results
    
    def _run_strings(self, file_path: Path) -> List[Dict[str, Any]]:
        """Run strings analysis"""
        if not self._tool_availability.get('strings', False):
            return []
        
        results = []
        
        try:
            cmd = ['strings', '-n', str(self.config.strings_min_length), str(file_path)]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.stdout:
                strings_output = result.stdout.split('\n')
                interesting_strings = []
                
                # Look for interesting patterns
                patterns = [
                    r'flag\{.*\}',  # CTF flags
                    r'password.*[:=].*',  # Passwords
                    r'[A-Za-z0-9+/]{20,}={0,2}',  # Base64
                    r'[0-9a-fA-F]{32,}',  # Hex strings
                    r'http[s]?://.*',  # URLs
                    r'.*\.(?:jpg|png|gif|mp3|zip|pdf)$',  # File references
                ]
                
                for string in strings_output:
                    string = string.strip()
                    if len(string) >= self.config.strings_min_length:
                        for pattern in patterns:
                            if re.search(pattern, string, re.IGNORECASE):
                                interesting_strings.append(string)
                                break
                
                if interesting_strings:
                    results.append({
                        "type": "strings_analysis",
                        "method": "strings",
                        "tool_name": "strings",
                        "confidence": 0.5,
                        "details": f"Found {len(interesting_strings)} interesting strings",
                        "interesting_strings": interesting_strings[:50],  # Limit output
                        "total_strings": len(strings_output),
                        "file_path": str(file_path)
                    })
                
        except subprocess.TimeoutExpired:
            self.logger.warning(f"Strings timeout for {file_path}")
        except Exception as e:
            self.logger.error(f"Strings error for {file_path}: {e}")
        
        return results
    
    def _run_exiftool(self, file_path: Path) -> List[Dict[str, Any]]:
        """Run exiftool metadata analysis"""
        if not self._tool_availability.get('exiftool', False):
            return []
        
        results = []
        
        try:
            cmd = ['exiftool', '-json', str(file_path)]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.stdout:
                try:
                    metadata = json.loads(result.stdout)[0]
                    
                    # Look for suspicious metadata
                    suspicious_fields = []
                    for key, value in metadata.items():
                        if isinstance(value, str):
                            # Check for hidden data indicators
                            if len(value) > 100 or any(char in value for char in ['\x00', '\xff']):
                                suspicious_fields.append({
                                    "field": key,
                                    "value": value[:200],
                                    "length": len(value)
                                })
                    
                    if suspicious_fields or len(metadata) > 50:
                        confidence = 0.3
                        if suspicious_fields:
                            confidence = 0.7
                        
                        results.append({
                            "type": "metadata_analysis",
                            "method": "exiftool",
                            "tool_name": "exiftool",
                            "confidence": confidence,
                            "details": f"Found {len(metadata)} metadata fields",
                            "metadata_count": len(metadata),
                            "suspicious_fields": suspicious_fields,
                            "file_path": str(file_path),
                            "all_metadata": metadata
                        })
                        
                except json.JSONDecodeError:
                    pass
                
        except subprocess.TimeoutExpired:
            self.logger.warning(f"ExifTool timeout for {file_path}")
        except Exception as e:
            self.logger.error(f"ExifTool error for {file_path}: {e}")
        
        return results
    
    def _run_hexdump_analysis(self, file_path: Path) -> List[Dict[str, Any]]:
        """Run hexdump analysis for patterns"""
        results = []
        
        try:
            # Read first and last chunks of file
            with open(file_path, 'rb') as f:
                file_size = f.seek(0, 2)
                f.seek(0)
                header = f.read(1024)
                
                if file_size > 2048:
                    f.seek(-1024, 2)
                    footer = f.read(1024)
                else:
                    footer = b''
            
            # Look for patterns in header and footer
            patterns_found = []
            
            # Common file signatures at wrong positions
            signatures = {
                b'\x89PNG': 'PNG',
                b'\xff\xd8\xff': 'JPEG',
                b'GIF8': 'GIF',
                b'PK\x03\x04': 'ZIP',
                b'%PDF': 'PDF',
                b'RIFF': 'RIFF/WAV'
            }
            
            for sig, file_type in signatures.items():
                # Check if signature appears in unexpected places
                pos = header.find(sig, 10)  # Skip first 10 bytes
                if pos > 10:
                    patterns_found.append({
                        "type": "embedded_signature",
                        "signature": file_type,
                        "position": pos,
                        "data": "header"
                    })
                
                if footer and sig in footer:
                    patterns_found.append({
                        "type": "embedded_signature",
                        "signature": file_type,
                        "position": file_size - len(footer) + footer.find(sig),
                        "data": "footer"
                    })
            
            # Check for high entropy regions
            if self._has_high_entropy(header) or self._has_high_entropy(footer):
                patterns_found.append({
                    "type": "high_entropy",
                    "details": "High entropy regions detected (possible encryption/compression)"
                })
            
            if patterns_found:
                results.append({
                    "type": "hexdump_analysis",
                    "method": "hexdump_analysis",
                    "tool_name": "classic_stego",
                    "confidence": 0.6,
                    "details": f"Found {len(patterns_found)} suspicious patterns",
                    "patterns": patterns_found,
                    "file_path": str(file_path)
                })
                
        except Exception as e:
            self.logger.error(f"Hexdump analysis error for {file_path}: {e}")
        
        return results
    
    def _has_high_entropy(self, data: bytes) -> bool:
        """Calculate entropy of byte data"""
        if len(data) == 0:
            return False
        
        # Count byte frequencies
        frequencies = [0] * 256
        for byte in data:
            frequencies[byte] += 1
        
        # Calculate entropy
        entropy = 0
        length = len(data)
        for freq in frequencies:
            if freq > 0:
                p = freq / length
                entropy -= p * (p.bit_length() - 1)
        
        # High entropy threshold (closer to 8 means more random)
        return entropy > 7.5
    
    def _run_magika(self, file_path: Path) -> List[Dict[str, Any]]:
        """Run Google Magika file type detection"""
        if not self._tool_availability.get('magika', False):
            return []
        
        results = []
        
        try:
            cmd = ['magika', '--json', str(file_path)]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.stdout:
                try:
                    magika_data = json.loads(result.stdout)
                    
                    # Compare with file extension
                    detected_type = magika_data.get('output', {}).get('ct_label', '')
                    confidence = magika_data.get('output', {}).get('score', 0)
                    
                    file_ext = file_path.suffix.lower()
                    expected_types = {
                        '.jpg': 'jpeg', '.jpeg': 'jpeg', '.png': 'png',
                        '.gif': 'gif', '.pdf': 'pdf', '.zip': 'zip'
                    }
                    
                    expected_type = expected_types.get(file_ext, '')
                    
                    if expected_type and detected_type.lower() != expected_type:
                        results.append({
                            "type": "file_type_mismatch",
                            "method": "magika",
                            "tool_name": "magika",
                            "confidence": 0.8,
                            "details": f"Extension says {expected_type}, Magika detected {detected_type}",
                            "expected_type": expected_type,
                            "detected_type": detected_type,
                            "detection_confidence": confidence,
                            "file_path": str(file_path)
                        })
                        
                except json.JSONDecodeError:
                    pass
                
        except subprocess.TimeoutExpired:
            self.logger.warning(f"Magika timeout for {file_path}")
        except Exception as e:
            self.logger.error(f"Magika error for {file_path}: {e}")
        
        return results
    
    def _run_tesseract(self, file_path: Path) -> List[Dict[str, Any]]:
        """Run Tesseract OCR on image files"""
        if not self._tool_availability.get('tesseract', False):
            return []
        
        # Only run on image files
        file_type = magic.from_file(str(file_path), mime=True)
        if not file_type.startswith('image/'):
            return []
        
        results = []
        
        try:
            with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp_file:
                cmd = ['tesseract', str(file_path), temp_file.name[:-4]]  # Remove .txt extension
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if os.path.exists(temp_file.name):
                    with open(temp_file.name, 'r', encoding='utf-8', errors='ignore') as f:
                        ocr_text = f.read().strip()
                    
                    if ocr_text and len(ocr_text) > 10:
                        # Look for interesting patterns in OCR text
                        confidence = 0.4
                        if any(keyword in ocr_text.lower() for keyword in 
                               ['flag', 'password', 'secret', 'hidden', 'key']):
                            confidence = 0.8
                        
                        results.append({
                            "type": "ocr_text",
                            "method": "tesseract",
                            "tool_name": "tesseract",
                            "confidence": confidence,
                            "details": f"OCR extracted {len(ocr_text)} characters",
                            "ocr_text": ocr_text[:1024],
                            "text_length": len(ocr_text),
                            "file_path": str(file_path)
                        })
                    
                    os.unlink(temp_file.name)
                
        except subprocess.TimeoutExpired:
            self.logger.warning(f"Tesseract timeout for {file_path}")
        except Exception as e:
            self.logger.error(f"Tesseract error for {file_path}: {e}")
        
        return results
    
    def _run_sox(self, file_path: Path) -> List[Dict[str, Any]]:
        """Run SoX audio analysis"""
        if not self._tool_availability.get('sox', False):
            return []
        
        # Only run on audio files
        file_type = magic.from_file(str(file_path), mime=True)
        if not file_type.startswith('audio/'):
            return []
        
        results = []
        
        try:
            # Get audio stats
            cmd = ['sox', str(file_path), '-n', 'stat']
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.stderr:  # Sox outputs stats to stderr
                stats = result.stderr
                
                # Parse stats for anomalies
                anomalies = []
                
                # Check for unusual characteristics
                if 'DC offset' in stats:
                    dc_offset_match = re.search(r'DC offset\s+([0-9.-]+)', stats)
                    if dc_offset_match:
                        dc_offset = float(dc_offset_match.group(1))
                        if abs(dc_offset) > 0.1:
                            anomalies.append(f"High DC offset: {dc_offset}")
                
                if 'Rough frequency' in stats:
                    freq_match = re.search(r'Rough frequency\s+(\d+)', stats)
                    if freq_match:
                        freq = int(freq_match.group(1))
                        if freq < 100 or freq > 20000:
                            anomalies.append(f"Unusual frequency: {freq} Hz")
                
                if anomalies:
                    results.append({
                        "type": "audio_anomaly",
                        "method": "sox",
                        "tool_name": "sox",
                        "confidence": 0.5,
                        "details": f"Audio anomalies detected: {', '.join(anomalies)}",
                        "anomalies": anomalies,
                        "file_path": str(file_path),
                        "full_stats": stats
                    })
                
        except subprocess.TimeoutExpired:
            self.logger.warning(f"SoX timeout for {file_path}")
        except Exception as e:
            self.logger.error(f"SoX error for {file_path}: {e}")
        
        return results
    
    def _run_ffmpeg(self, file_path: Path) -> List[Dict[str, Any]]:
        """Run FFmpeg media analysis"""
        if not self._tool_availability.get('ffmpeg', False):
            return []
        
        results = []
        
        try:
            cmd = ['ffmpeg', '-i', str(file_path), '-f', 'null', '-']
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            # FFmpeg outputs info to stderr
            if result.stderr:
                info = result.stderr
                
                # Look for multiple streams or unusual metadata
                stream_count = len(re.findall(r'Stream #\d+:\d+', info))
                metadata_lines = [line for line in info.split('\n') if 'metadata:' in line.lower()]
                
                if stream_count > 2 or len(metadata_lines) > 5:
                    results.append({
                        "type": "media_analysis",
                        "method": "ffmpeg",
                        "tool_name": "ffmpeg",
                        "confidence": 0.4,
                        "details": f"Media file has {stream_count} streams and {len(metadata_lines)} metadata entries",
                        "stream_count": stream_count,
                        "metadata_count": len(metadata_lines),
                        "file_path": str(file_path),
                        "ffmpeg_info": info[:2048]
                    })
                
        except subprocess.TimeoutExpired:
            self.logger.warning(f"FFmpeg timeout for {file_path}")
        except Exception as e:
            self.logger.error(f"FFmpeg error for {file_path}: {e}")
        
        return results
    
    def _run_stegcracker(self, file_path: Path) -> List[Dict[str, Any]]:
        """Run StegCracker with timeout"""
        if not self._tool_availability.get('stegcracker', False):
            return []
        
        if not self.config.stegcracker_enabled:
            return []
        
        results = []
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                output_file = Path(temp_dir) / "cracked.txt"
                cmd = ['stegcracker', str(file_path), '-o', str(output_file)]
                
                # Add wordlist if available
                if self.config.steghide_wordlist and Path(self.config.steghide_wordlist).exists():
                    cmd.extend(['-w', self.config.steghide_wordlist])
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.config.stegcracker_timeout
                )
                
                if result.returncode == 0 and output_file.exists():
                    with open(output_file, 'rb') as f:
                        content = f.read()
                    
                    results.append({
                        "type": "steganography",
                        "method": "stegcracker",
                        "tool_name": "stegcracker",
                        "confidence": 0.95,
                        "details": "StegCracker successfully cracked steghide",
                        "extracted_size": len(content),
                        "extracted_content": content[:1024].decode('utf-8', errors='ignore'),
                        "file_path": str(file_path),
                        "raw_output": result.stdout
                    })
                
        except subprocess.TimeoutExpired:
            self.logger.warning(f"StegCracker timeout for {file_path}")
        except Exception as e:
            self.logger.error(f"StegCracker error for {file_path}: {e}")
        
        return results
    
    def cleanup(self):
        """Cleanup temporary files"""
        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
        except Exception as e:
            self.logger.error(f"Failed to cleanup temp dir: {e}")
        
        self.executor.shutdown(wait=True)