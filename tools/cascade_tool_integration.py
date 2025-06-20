#!/usr/bin/env python3
"""
Cascade Analyzer Tool - Integrated into StegAnalyzer Framework
Recursive steganography analysis using zsteg + binwalk
"""

import os
import subprocess
import tempfile
import shutil
from pathlib import Path
import hashlib
import json
import time
import logging
from typing import Dict, List, Set, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict

@dataclass
class CascadeResult:
    """Result of cascade analysis"""
    file_path: str
    file_hash: str
    file_size: int
    depth: int
    parent_hash: Optional[str]
    extraction_method: str
    zsteg_results: List[Dict]
    binwalk_results: List[Dict]
    extracted_files: List[str]
    analysis_time: float
    file_type: str

class CascadeAnalyzer:
    """Cascade analyzer integrated with StegAnalyzer framework"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Cascade-specific config
        if hasattr(config, 'cascade'):
            self.max_depth = config.cascade.max_depth
            self.enable_zsteg = config.cascade.enable_zsteg
            self.enable_binwalk = config.cascade.enable_binwalk
            self.save_extracts = config.cascade.save_extracts
        else:
            # Default settings
            self.max_depth = 10
            self.enable_zsteg = True
            self.enable_binwalk = True
            self.save_extracts = True
        
        # Tracking
        self.analyzed_hashes: Set[str] = set()
        self.results: List[CascadeResult] = []
        self.file_tree: Dict[str, List[str]] = defaultdict(list)
        
        # Tool availability
        self.zsteg_available = self._check_tool_availability('zsteg')
        self.binwalk_available = self._check_tool_availability('binwalk')
        
        if not self.zsteg_available:
            self.logger.warning("zsteg not available - skipping zsteg analysis")
        if not self.binwalk_available:
            self.logger.warning("binwalk not available - skipping binwalk analysis")
        
        # Comprehensive zsteg parameters
        self.zsteg_params = self._generate_zsteg_parameters()
        
        # Supported file extensions
        self.image_extensions = {'.png', '.bmp', '.gif', '.tiff', '.tif', '.webp'}
        self.binwalk_extensions = {
            '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.tif', '.webp',
            '.pdf', '.zip', '.rar', '.7z', '.tar', '.gz', '.exe', '.bin',
            '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx'
        }
    
    def _check_tool_availability(self, tool_name: str) -> bool:
        """Check if external tool is available"""
        try:
            subprocess.run([tool_name, '--help'], 
                         capture_output=True, timeout=5)
            return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def _generate_zsteg_parameters(self) -> List[List[str]]:
        """Generate comprehensive zsteg parameter combinations"""
        params = []
        
        # Basic LSB parameters
        channels = ['r', 'g', 'b', 'rgb', 'bgr', 'a', 'rg', 'rb', 'gb', 'rgba', 'bgra']
        bit_orders = ['msb', 'lsb']
        bit_planes = ['0', '1', '2', '3', '4', '5', '6', '7']
        
        # Standard bitplane analysis
        for channel in channels:
            for order in bit_orders:
                for plane in bit_planes:
                    params.append([f'{channel}{order}', f'{plane}'])
        
        # Exotic parameters
        exotic_params = [
            # Different color spaces
            ['xyY', 'lsb', '1'], ['Lab', 'lsb', '1'], ['YUV', 'lsb', '1'],
            ['HSV', 'lsb', '1'], ['HSL', 'lsb', '1'],
            
            # Inverted bits
            ['inv', 'r', 'lsb', '1'], ['inv', 'g', 'lsb', '1'], ['inv', 'b', 'lsb', '1'],
            
            # Prime patterns
            ['prime', 'r', 'lsb'], ['prime', 'g', 'lsb'], ['prime', 'b', 'lsb'],
            
            # XOR patterns
            ['xor', 'r', 'lsb', '1'], ['xor', 'g', 'lsb', '1'], ['xor', 'b', 'lsb', '1'],
            
            # Different byte orders
            ['b1,r,lsb,xy'], ['b2,g,lsb,xy'], ['b3,b,lsb,xy'], ['b4,a,lsb,xy'],
            
            # Row/column patterns
            ['B1,rgb,lsb,xy'], ['B2,rgb,lsb,xy'], ['B3,rgb,lsb,xy'], ['B4,rgb,lsb,xy'],
        ]
        
        params.extend(exotic_params)
        
        # Add more exotic combinations
        for i in range(1, 8):
            params.extend([
                [f'b{i},r,lsb,xy'], [f'b{i},g,lsb,xy'], 
                [f'b{i},b,lsb,xy'], [f'b{i},rgb,lsb,xy']
            ])
        
        return params
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def _get_file_type(self, file_path: Path) -> str:
        """Get file type using file command"""
        try:
            result = subprocess.run(['file', '-b', str(file_path)], 
                                  capture_output=True, text=True, timeout=10)
            return result.stdout.strip() if result.returncode == 0 else "unknown"
        except:
            return "unknown"
    
    def _run_zsteg_analysis(self, file_path: Path) -> List[Dict]:
        """Run comprehensive zsteg analysis"""
        if not self.enable_zsteg or not self.zsteg_available:
            return []
        
        if file_path.suffix.lower() not in self.image_extensions:
            return []
        
        results = []
        self.logger.info(f"Running zsteg analysis on {file_path.name} with {len(self.zsteg_params)} parameter sets")
        
        for i, params in enumerate(self.zsteg_params):
            try:
                cmd = ['zsteg'] + params + [str(file_path)]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                
                if result.stdout.strip():
                    results.append({
                        'params': ' '.join(params),
                        'output': result.stdout.strip(),
                        'command': ' '.join(cmd),
                        'success': True,
                        'confidence': self._calculate_zsteg_confidence(result.stdout)
                    })
                    self.logger.debug(f"zsteg found data with params: {' '.join(params)}")
                
                # Progress logging
                if (i + 1) % 100 == 0:
                    self.logger.info(f"Tested {i + 1}/{len(self.zsteg_params)} zsteg parameter combinations")
                    
            except subprocess.TimeoutExpired:
                self.logger.debug(f"zsteg timeout with params: {' '.join(params)}")
                continue
            except Exception as e:
                self.logger.debug(f"zsteg error with params {' '.join(params)}: {e}")
                continue
        
        self.logger.info(f"zsteg analysis complete: {len(results)} results found")
        return results
    
    def _calculate_zsteg_confidence(self, output: str) -> float:
        """Calculate confidence score for zsteg output"""
        # Basic scoring based on output characteristics
        score = 0.3  # Base score
        
        # Higher confidence for readable text
        if any(keyword in output.lower() for keyword in ['flag', 'key', 'password', 'secret']):
            score += 0.4
        
        # Higher confidence for file signatures
        if any(sig in output for sig in ['PNG', 'JPEG', 'PDF', 'ZIP']):
            score += 0.3
        
        # Length factor
        if len(output) > 50:
            score += 0.2
        
        return min(score, 1.0)
    
    def _run_binwalk_analysis(self, file_path: Path, output_dir: Path) -> List[Dict]:
        """Run binwalk extraction and analysis"""
        if not self.enable_binwalk or not self.binwalk_available:
            return []
        
        results = []
        self.logger.info(f"Running binwalk analysis on {file_path.name}")
        
        try:
            # Create extraction directory
            extract_dir = output_dir / f"binwalk_{file_path.stem}_{int(time.time())}"
            extract_dir.mkdir(exist_ok=True)
            
            # Run binwalk extraction
            cmd = ['binwalk', '-e', '-C', str(extract_dir), str(file_path)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            # Find extracted files
            extracted_files = []
            if extract_dir.exists():
                for extracted_file in extract_dir.rglob('*'):
                    if extracted_file.is_file() and extracted_file.stat().st_size > 0:
                        extracted_files.append(str(extracted_file))
            
            results.append({
                'command': ' '.join(cmd),
                'output': result.stdout.strip(),
                'extracted_files': extracted_files,
                'extract_dir': str(extract_dir),
                'success': result.returncode == 0,
                'file_count': len(extracted_files)
            })
            
            self.logger.info(f"binwalk extracted {len(extracted_files)} files")
            
        except subprocess.TimeoutExpired:
            self.logger.warning(f"binwalk timeout on {file_path.name}")
        except Exception as e:
            self.logger.error(f"binwalk error: {e}")
        
        return results
    
    def _extract_zsteg_data(self, zsteg_results: List[Dict], file_path: Path, output_dir: Path) -> List[str]:
        """Extract embedded data found by zsteg"""
        if not self.save_extracts:
            return []
        
        extracted_files = []
        
        for i, result in enumerate(zsteg_results):
            if not result['success'] or result['confidence'] < 0.5:
                continue
                
            try:
                params = result['params'].split()
                extract_file = output_dir / f"zsteg_extract_{file_path.stem}_{i}.bin"
                
                cmd = ['zsteg', '-E'] + params + [str(file_path)]
                
                with open(extract_file, 'wb') as f:
                    subprocess.run(cmd, stdout=f, timeout=30)
                
                if extract_file.exists() and extract_file.stat().st_size > 0:
                    extracted_files.append(str(extract_file))
                    self.logger.debug(f"Extracted zsteg data to {extract_file.name}")
                
            except Exception as e:
                self.logger.debug(f"Failed to extract zsteg data with params {result['params']}: {e}")
                continue
        
        return extracted_files
    
    async def cascade_analyze(self, file_path: Path, session_id: str = None) -> List[Dict[str, Any]]:
        """Main cascade analysis method - StegAnalyzer compatible"""
        results = []
        
        try:
            # Create output directory
            output_dir = Path(tempfile.mkdtemp(prefix="cascade_"))
            
            # Start recursive analysis
            cascade_result = self._analyze_file_recursive(file_path, 0, None, output_dir)
            
            # Convert to StegAnalyzer result format
            for result in self.results:
                steg_result = {
                    "type": "cascade_analysis",
                    "method": "recursive_cascade",
                    "tool_name": "cascade_analyzer",
                    "confidence": self._calculate_overall_confidence(result),
                    "details": f"Cascade analysis at depth {result.depth}",
                    "file_path": result.file_path,
                    "file_hash": result.file_hash,
                    "depth": result.depth,
                    "parent_hash": result.parent_hash,
                    "zsteg_findings": len(result.zsteg_results),
                    "binwalk_extractions": len(result.extracted_files),
                    "analysis_time": result.analysis_time,
                    "extracted_files": result.extracted_files,
                    "cascade_tree": dict(self.file_tree)
                }
                results.append(steg_result)
        
        except Exception as e:
            self.logger.error(f"Cascade analysis failed: {e}")
            results.append({
                "type": "cascade_error",
                "method": "recursive_cascade",
                "tool_name": "cascade_analyzer",
                "confidence": 0.0,
                "details": f"Cascade analysis failed: {str(e)}",
                "file_path": str(file_path)
            })
        
        return results
    
    def _calculate_overall_confidence(self, result: CascadeResult) -> float:
        """Calculate overall confidence for cascade result"""
        confidence = 0.0
        
        # Base confidence for finding anything
        if result.zsteg_results or result.extracted_files:
            confidence = 0.3
        
        # Zsteg confidence
        if result.zsteg_results:
            zsteg_confidence = max(r.get('confidence', 0.0) for r in result.zsteg_results)
            confidence = max(confidence, zsteg_confidence)
        
        # File extraction bonus
        if result.extracted_files:
            confidence += min(len(result.extracted_files) * 0.1, 0.4)
        
        # Depth penalty (deeper = less certain)
        depth_penalty = result.depth * 0.05
        confidence = max(0.0, confidence - depth_penalty)
        
        return min(confidence, 1.0)
    
    def _analyze_file_recursive(self, file_path: Path, depth: int, parent_hash: Optional[str], output_dir: Path) -> Optional[CascadeResult]:
        """Recursive file analysis"""
        if depth > self.max_depth:
            self.logger.info(f"Max depth {self.max_depth} reached for {file_path.name}")
            return None
        
        if not file_path.exists():
            return None
        
        # Check if already analyzed
        file_hash = self._get_file_hash(file_path)
        if file_hash in self.analyzed_hashes:
            self.logger.debug(f"Already analyzed: {file_path.name}")
            return None
        
        self.analyzed_hashes.add(file_hash)
        self.logger.info(f"Analyzing {file_path.name} at depth {depth}")
        
        start_time = time.time()
        
        # Create analysis directory
        analysis_dir = output_dir / f"depth_{depth}" / file_hash[:8]
        analysis_dir.mkdir(parents=True, exist_ok=True)
        
        # Run analyses
        zsteg_results = self._run_zsteg_analysis(file_path)
        binwalk_results = self._run_binwalk_analysis(file_path, analysis_dir)
        
        # Extract embedded data
        zsteg_extracted = self._extract_zsteg_data(zsteg_results, file_path, analysis_dir)
        
        # Collect all extracted files
        all_extracted = zsteg_extracted.copy()
        for binwalk_result in binwalk_results:
            all_extracted.extend(binwalk_result.get('extracted_files', []))
        
        # Create result
        result = CascadeResult(
            file_path=str(file_path),
            file_hash=file_hash,
            file_size=file_path.stat().st_size,
            depth=depth,
            parent_hash=parent_hash,
            extraction_method="root" if depth == 0 else "cascade",
            zsteg_results=zsteg_results,
            binwalk_results=binwalk_results,
            extracted_files=all_extracted,
            analysis_time=time.time() - start_time,
            file_type=self._get_file_type(file_path)
        )
        
        self.results.append(result)
        
        # Update file tree
        if parent_hash:
            self.file_tree[parent_hash].append(file_hash)
        
        # Recursively analyze extracted files
        for extracted_file_path in all_extracted:
            extracted_path = Path(extracted_file_path)
            if extracted_path.exists() and extracted_path.is_file():
                self._analyze_file_recursive(extracted_path, depth + 1, file_hash, output_dir)
        
        return result
    
    def get_analysis_methods(self) -> List[str]:
        """Return available analysis methods"""
        methods = ["cascade_analyze"]
        if self.zsteg_available:
            methods.append("zsteg_comprehensive")
        if self.binwalk_available:
            methods.append("binwalk_extraction")
        return methods
    
    def get_tool_info(self) -> Dict[str, Any]:
        """Get tool information"""
        return {
            "name": "CascadeAnalyzer",
            "version": "1.0.0",
            "description": "Recursive steganography analysis with zsteg and binwalk",
            "capabilities": [
                "Recursive file extraction",
                "Comprehensive zsteg parameter testing",
                "Binwalk file carving",
                "Extraction tree mapping"
            ],
            "requirements": {
                "zsteg": self.zsteg_available,
                "binwalk": self.binwalk_available
            },
            "max_depth": self.max_depth,
            "zsteg_parameters": len(self.zsteg_params)
        }
