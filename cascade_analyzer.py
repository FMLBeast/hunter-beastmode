#!/usr/bin/env python3
"""
Recursive Cascade Steganography Analyzer
Recursively extracts and analyzes files using zsteg + binwalk + carving
"""

import os
import subprocess
import tempfile
import shutil
from pathlib import Path
import hashlib
import json
from typing import Dict, List, Set, Any, Optional
import re
import time
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

class RecursiveCascadeAnalyzer:
    """Recursive cascade analyzer using zsteg + binwalk"""
    
    def __init__(self, max_depth=10, output_dir="cascade_analysis"):
        self.max_depth = max_depth
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Tracking
        self.analyzed_hashes: Set[str] = set()
        self.results: List[CascadeResult] = []
        self.file_tree: Dict[str, List[str]] = defaultdict(list)
        
        # Comprehensive zsteg parameters
        self.zsteg_params = self._generate_zsteg_parameters()
        
        # Supported image formats for zsteg
        self.image_extensions = {'.png', '.bmp', '.gif', '.tiff', '.tif', '.webp'}
        
        # All file extensions to analyze with binwalk
        self.binwalk_extensions = {
            '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.tif', '.webp',
            '.pdf', '.zip', '.rar', '.7z', '.tar', '.gz', '.exe', '.bin',
            '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx'
        }
    
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
            ['xyY', 'lsb', '1'],
            ['Lab', 'lsb', '1'], 
            ['YUV', 'lsb', '1'],
            ['HSV', 'lsb', '1'],
            ['HSL', 'lsb', '1'],
            
            # Inverted bits
            ['inv', 'r', 'lsb', '1'],
            ['inv', 'g', 'lsb', '1'],
            ['inv', 'b', 'lsb', '1'],
            
            # Prime number patterns
            ['prime', 'r', 'lsb'],
            ['prime', 'g', 'lsb'],
            ['prime', 'b', 'lsb'],
            
            # XOR patterns
            ['xor', 'r', 'lsb', '1'],
            ['xor', 'g', 'lsb', '1'],
            ['xor', 'b', 'lsb', '1'],
            
            # Different byte orders
            ['b1,r,lsb,xy'],
            ['b2,g,lsb,xy'],
            ['b3,b,lsb,xy'],
            ['b4,a,lsb,xy'],
            
            # Row/column patterns
            ['B1,rgb,lsb,xy'],
            ['B2,rgb,lsb,xy'],
            ['B3,rgb,lsb,xy'],
            ['B4,rgb,lsb,xy'],
        ]
        
        params.extend(exotic_params)
        
        # Add more exotic combinations
        for i in range(1, 8):
            params.append(['b{},r,lsb,xy'.format(i)])
            params.append(['b{},g,lsb,xy'.format(i)])
            params.append(['b{},b,lsb,xy'.format(i)])
            params.append(['b{},rgb,lsb,xy'.format(i)])
        
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
    
    def _run_zsteg(self, file_path: Path) -> List[Dict]:
        """Run zsteg with all parameter combinations"""
        results = []
        
        if file_path.suffix.lower() not in self.image_extensions:
            return results
        
        print(f"🔍 Running comprehensive zsteg analysis on {file_path.name}")
        
        for i, params in enumerate(self.zsteg_params):
            try:
                # Build zsteg command
                cmd = ['zsteg'] + params + [str(file_path)]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                
                if result.stdout.strip():
                    results.append({
                        'params': ' '.join(params),
                        'output': result.stdout.strip(),
                        'command': ' '.join(cmd),
                        'success': True
                    })
                    print(f"  ✅ Found data with params: {' '.join(params)}")
                
                # Progress indicator
                if (i + 1) % 50 == 0:
                    print(f"    Tested {i + 1}/{len(self.zsteg_params)} parameter combinations")
                    
            except subprocess.TimeoutExpired:
                print(f"  ⏰ Timeout with params: {' '.join(params)}")
                continue
            except Exception as e:
                print(f"  ❌ Error with params {' '.join(params)}: {e}")
                continue
        
        print(f"  📊 zsteg found {len(results)} results with different parameters")
        return results
    
    def _run_binwalk(self, file_path: Path, output_dir: Path) -> List[Dict]:
        """Run binwalk extraction"""
        results = []
        
        print(f"🔍 Running binwalk on {file_path.name}")
        
        try:
            # Create extraction directory
            extract_dir = output_dir / f"binwalk_{file_path.stem}_{int(time.time())}"
            extract_dir.mkdir(exist_ok=True)
            
            # Run binwalk extraction
            cmd = ['binwalk', '-e', '-C', str(extract_dir), str(file_path)]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            # Find extracted files
            extracted_files = []
            if extract_dir.exists():
                for extracted_file in extract_dir.rglob('*'):
                    if extracted_file.is_file():
                        extracted_files.append(str(extracted_file))
            
            results.append({
                'command': ' '.join(cmd),
                'output': result.stdout.strip(),
                'extracted_files': extracted_files,
                'extract_dir': str(extract_dir),
                'success': result.returncode == 0
            })
            
            print(f"  📦 binwalk extracted {len(extracted_files)} files")
            
        except subprocess.TimeoutExpired:
            print(f"  ⏰ binwalk timeout on {file_path.name}")
        except Exception as e:
            print(f"  ❌ binwalk error: {e}")
        
        return results
    
    def _extract_embedded_data(self, zsteg_results: List[Dict], file_path: Path, output_dir: Path) -> List[str]:
        """Extract embedded data found by zsteg"""
        extracted_files = []
        
        for i, result in enumerate(zsteg_results):
            if not result['success']:
                continue
                
            try:
                # Try to extract the data
                params = result['params'].split()
                extract_file = output_dir / f"zsteg_extract_{file_path.stem}_{i}.bin"
                
                # Run zsteg with extraction
                cmd = ['zsteg', '-E'] + params + [str(file_path)]
                
                with open(extract_file, 'wb') as f:
                    result_extract = subprocess.run(cmd, stdout=f, timeout=30)
                
                if extract_file.exists() and extract_file.stat().st_size > 0:
                    extracted_files.append(str(extract_file))
                    print(f"  💾 Extracted data to {extract_file.name}")
                
            except Exception as e:
                print(f"  ❌ Failed to extract with params {result['params']}: {e}")
                continue
        
        return extracted_files
    
    def analyze_file(self, file_path: Path, depth: int = 0, parent_hash: Optional[str] = None) -> Optional[CascadeResult]:
        """Analyze a single file recursively"""
        
        if depth > self.max_depth:
            print(f"⚠️  Max depth {self.max_depth} reached for {file_path.name}")
            return None
        
        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            return None
        
        # Calculate hash to avoid reprocessing
        file_hash = self._get_file_hash(file_path)
        if file_hash in self.analyzed_hashes:
            print(f"⏭️  Already analyzed: {file_path.name} (hash: {file_hash[:8]})")
            return None
        
        self.analyzed_hashes.add(file_hash)
        
        print(f"\n{'  ' * depth}🎯 Analyzing: {file_path.name} (depth: {depth})")
        print(f"{'  ' * depth}📊 Hash: {file_hash[:16]}...")
        print(f"{'  ' * depth}📁 Size: {file_path.stat().st_size} bytes")
        
        start_time = time.time()
        
        # Create analysis directory for this file
        analysis_dir = self.output_dir / f"depth_{depth}" / file_hash[:8]
        analysis_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy original file
        original_copy = analysis_dir / file_path.name
        shutil.copy2(file_path, original_copy)
        
        # Run zsteg analysis
        zsteg_results = self._run_zsteg(file_path)
        
        # Run binwalk analysis
        binwalk_results = self._run_binwalk(file_path, analysis_dir)
        
        # Extract embedded data from zsteg
        zsteg_extracted = self._extract_embedded_data(zsteg_results, file_path, analysis_dir)
        
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
        
        print(f"{'  ' * depth}✅ Analysis complete: {len(zsteg_results)} zsteg results, {len(all_extracted)} files extracted")
        
        # Recursively analyze extracted files
        for extracted_file_path in all_extracted:
            extracted_path = Path(extracted_file_path)
            if extracted_path.exists() and extracted_path.is_file():
                print(f"{'  ' * depth}🔄 Recursively analyzing: {extracted_path.name}")
                self.analyze_file(extracted_path, depth + 1, file_hash)
        
        return result
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive analysis report"""
        
        total_files = len(self.results)
        total_zsteg_findings = sum(len(r.zsteg_results) for r in self.results)
        total_extracted = sum(len(r.extracted_files) for r in self.results)
        
        # Depth analysis
        depth_stats = defaultdict(int)
        for result in self.results:
            depth_stats[result.depth] += 1
        
        # File type analysis
        type_stats = defaultdict(int)
        for result in self.results:
            type_stats[result.file_type] += 1
        
        report = {
            'summary': {
                'total_files_analyzed': total_files,
                'total_zsteg_findings': total_zsteg_findings,
                'total_files_extracted': total_extracted,
                'max_depth_reached': max(depth_stats.keys()) if depth_stats else 0,
                'analysis_duration': sum(r.analysis_time for r in self.results)
            },
            'depth_distribution': dict(depth_stats),
            'file_types': dict(type_stats),
            'file_tree': dict(self.file_tree),
            'detailed_results': [asdict(r) for r in self.results]
        }
        
        # Save report
        report_file = self.output_dir / 'cascade_analysis_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📊 Analysis Report:")
        print(f"  Total files analyzed: {total_files}")
        print(f"  Total zsteg findings: {total_zsteg_findings}")
        print(f"  Total files extracted: {total_extracted}")
        print(f"  Max depth reached: {max(depth_stats.keys()) if depth_stats else 0}")
        print(f"  Report saved: {report_file}")
        
        return report

def main():
    """Main cascade analysis function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Recursive Cascade Steganography Analyzer")
    parser.add_argument("file_path", help="Path to initial file for analysis")
    parser.add_argument("--max-depth", type=int, default=10, help="Maximum recursion depth")
    parser.add_argument("--output-dir", default="cascade_analysis", help="Output directory")
    
    args = parser.parse_args()
    
    file_path = Path(args.file_path)
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return
    
    print("🚀 Recursive Cascade Steganography Analyzer")
    print("=" * 60)
    print(f"📁 Input file: {file_path}")
    print(f"📊 Max depth: {args.max_depth}")
    print(f"📂 Output directory: {args.output_dir}")
    print()
    
    # Create analyzer
    analyzer = RecursiveCascadeAnalyzer(max_depth=args.max_depth, output_dir=args.output_dir)
    
    # Start cascade analysis
    start_time = time.time()
    analyzer.analyze_file(file_path)
    
    # Generate report
    report = analyzer.generate_report()
    
    print(f"\n🎉 Cascade analysis complete!")
    print(f"⏱️  Total time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
