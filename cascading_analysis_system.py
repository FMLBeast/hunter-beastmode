#!/usr/bin/env python3
"""
Cascading Analysis System for Deep Steganography Extraction
- Analyzes original file
- Extracts/carves files using tools like binwalk, steghide, zsteg
- Automatically analyzes extracted files recursively
- Builds hierarchical file tree
- Continues until no more files can be extracted
"""

import asyncio
import logging
import shutil
import tempfile
import json
from pathlib import Path
from typing import Dict, List, Any, Set, Optional
from dataclasses import dataclass, asdict
import hashlib
import time
import subprocess

@dataclass
class ExtractionNode:
    """Represents a file in the extraction tree"""
    file_path: Path
    parent_path: Optional[Path]
    extraction_method: str
    tool_name: str
    depth: int
    file_hash: str
    file_size: int
    mime_type: str
    children: List['ExtractionNode']
    findings: List[Dict[str, Any]]
    extracted_at: float

class CascadingAnalyzer:
    def __init__(self, orchestrator, output_dir: Path = None):
        self.orchestrator = orchestrator
        self.logger = logging.getLogger(__name__)
        self.output_dir = output_dir or Path(tempfile.mkdtemp(prefix="cascading_analysis_"))
        self.extraction_tree = None
        self.processed_hashes: Set[str] = set()
        self.max_depth = 20  # Prevent infinite recursion
        self.max_files = 50000  # Safety limit
        self.total_files_processed = 0
        
        # Tools that can extract files
        self.extraction_tools = {
            'binwalk': self._run_binwalk_extraction,
            'steghide': self._run_steghide_extraction,
            'zsteg': self._run_zsteg_extraction,
            'foremost': self._run_foremost_extraction,
            'scalpel': self._run_scalpel_extraction,
            'strings': self._extract_from_strings,
            'exiftool': self._extract_from_exiftool
        }
        
        self.logger.info(f"Cascading analyzer initialized, output: {self.output_dir}")

    async def analyze_cascading(self, initial_file: Path, session_id: str) -> ExtractionNode:
        """Start cascading analysis from initial file"""
        self.logger.info(f"🚀 Starting cascading analysis of {initial_file}")
        
        # Create initial node
        file_hash = self._calculate_hash(initial_file)
        initial_node = ExtractionNode(
            file_path=initial_file,
            parent_path=None,
            extraction_method="initial",
            tool_name="user_input",
            depth=0,
            file_hash=file_hash,
            file_size=initial_file.stat().st_size,
            mime_type=self._get_mime_type(initial_file),
            children=[],
            findings=[],
            extracted_at=time.time()
        )
        
        self.extraction_tree = initial_node
        self.processed_hashes.add(file_hash)
        
        # Start recursive analysis
        await self._analyze_node_recursive(initial_node, session_id)
        
        # Print final tree
        self._print_extraction_tree()
        
        # Save tree to JSON
        await self._save_extraction_tree()
        
        return self.extraction_tree

    async def _analyze_node_recursive(self, node: ExtractionNode, session_id: str):
        """Recursively analyze a node and its extracted files"""
        
        if node.depth >= self.max_depth:
            self.logger.warning(f"Max depth {self.max_depth} reached for {node.file_path}")
            return
            
        if self.total_files_processed >= self.max_files:
            self.logger.warning(f"Max files {self.max_files} processed, stopping")
            return
        
        self.total_files_processed += 1
        
        self.logger.info(f"🔍 Analyzing [{node.depth}] {node.file_path.name} ({node.file_size} bytes)")
        
        try:
            # Run standard analysis on this file
            results = await self.orchestrator.analyze(node.file_path, session_id)
            node.findings.extend(results or [])
            
            # Extract files using all available tools
            extracted_files = await self._extract_files_from_node(node)
            
            if extracted_files:
                self.logger.info(f"   ✅ Extracted {len(extracted_files)} files from {node.file_path.name}")
                
                # Create child nodes and analyze them
                for extracted_file_info in extracted_files:
                    child_node = await self._create_child_node(node, extracted_file_info)
                    if child_node:
                        node.children.append(child_node)
                        # Recursive analysis
                        await self._analyze_node_recursive(child_node, session_id)
            else:
                self.logger.info(f"   ⚪ No files extracted from {node.file_path.name}")
                
        except Exception as e:
            self.logger.error(f"Error analyzing {node.file_path}: {e}")

    async def _extract_files_from_node(self, node: ExtractionNode) -> List[Dict[str, Any]]:
        """Extract files from a node using all available tools"""
        all_extracted = []
        
        # Create extraction directory for this node
        node_dir = self.output_dir / f"depth_{node.depth}" / node.file_path.stem
        node_dir.mkdir(parents=True, exist_ok=True)
        
        # Try each extraction tool
        for tool_name, extraction_func in self.extraction_tools.items():
            try:
                extracted = await extraction_func(node.file_path, node_dir / tool_name)
                if extracted:
                    self.logger.debug(f"   {tool_name}: extracted {len(extracted)} files")
                    all_extracted.extend(extracted)
            except Exception as e:
                self.logger.debug(f"   {tool_name}: failed - {e}")
        
        return all_extracted

    async def _create_child_node(self, parent: ExtractionNode, extracted_info: Dict[str, Any]) -> Optional[ExtractionNode]:
        """Create a child node from extracted file info"""
        extracted_path = Path(extracted_info['file_path'])
        
        if not extracted_path.exists() or extracted_path.stat().st_size == 0:
            return None
        
        # Calculate hash to avoid duplicates
        file_hash = self._calculate_hash(extracted_path)
        if file_hash in self.processed_hashes:
            self.logger.debug(f"   Skipping duplicate: {extracted_path.name}")
            return None
        
        self.processed_hashes.add(file_hash)
        
        child_node = ExtractionNode(
            file_path=extracted_path,
            parent_path=parent.file_path,
            extraction_method=extracted_info.get('method', 'unknown'),
            tool_name=extracted_info.get('tool', 'unknown'),
            depth=parent.depth + 1,
            file_hash=file_hash,
            file_size=extracted_path.stat().st_size,
            mime_type=self._get_mime_type(extracted_path),
            children=[],
            findings=[],
            extracted_at=time.time()
        )
        
        return child_node

    # Tool-specific extraction methods
    async def _run_binwalk_extraction(self, file_path: Path, output_dir: Path) -> List[Dict[str, Any]]:
        """Extract files using binwalk"""
        output_dir.mkdir(parents=True, exist_ok=True)
        extracted = []
        
        try:
            cmd = ['binwalk', '--extract', '--directory', str(output_dir), str(file_path)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                # Find extracted files
                for extracted_file in output_dir.rglob('*'):
                    if extracted_file.is_file() and extracted_file.stat().st_size > 0:
                        extracted.append({
                            'file_path': str(extracted_file),
                            'method': 'binwalk_extraction',
                            'tool': 'binwalk',
                            'details': f'Extracted by binwalk from offset'
                        })
        except Exception as e:
            self.logger.debug(f"Binwalk extraction failed: {e}")
        
        return extracted

    async def _run_steghide_extraction(self, file_path: Path, output_dir: Path) -> List[Dict[str, Any]]:
        """Extract files using steghide"""
        output_dir.mkdir(parents=True, exist_ok=True)
        extracted = []
        
        # Try common passwords and no password
        passwords = ['', 'password', '123456', 'admin', 'secret', 'hidden', 'flag', 'ctf']
        
        for i, password in enumerate(passwords):
            try:
                output_file = output_dir / f"steghide_extracted_{i}.dat"
                cmd = ['steghide', 'extract', '-sf', str(file_path), '-xf', str(output_file), '-p', password]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0 and output_file.exists() and output_file.stat().st_size > 0:
                    extracted.append({
                        'file_path': str(output_file),
                        'method': 'steghide_extraction',
                        'tool': 'steghide',
                        'details': f'Extracted with password: {password or "(empty)"}'
                    })
                    break  # Stop at first successful extraction
            except Exception as e:
                self.logger.debug(f"Steghide with password '{password}' failed: {e}")
        
        return extracted

    async def _run_zsteg_extraction(self, file_path: Path, output_dir: Path) -> List[Dict[str, Any]]:
        """Extract files using zsteg"""
        output_dir.mkdir(parents=True, exist_ok=True)
        extracted = []
        
        try:
            # Run zsteg to find all possible extractions
            cmd = ['zsteg', '--all', str(file_path)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.stdout:
                lines = result.stdout.strip().split('\n')
                for i, line in enumerate(lines):
                    if 'file:' in line.lower() or len(line) > 50:  # Likely contains data
                        # Extract this specific channel
                        parts = line.split(':')[0].strip() if ':' in line else f"b{i},lsb,xy"
                        output_file = output_dir / f"zsteg_extracted_{i}.dat"
                        
                        try:
                            extract_cmd = ['zsteg', '-E', parts, str(file_path)]
                            extract_result = subprocess.run(extract_cmd, capture_output=True, timeout=30)
                            
                            if extract_result.stdout and len(extract_result.stdout) > 10:
                                with open(output_file, 'wb') as f:
                                    f.write(extract_result.stdout)
                                
                                extracted.append({
                                    'file_path': str(output_file),
                                    'method': 'zsteg_extraction',
                                    'tool': 'zsteg',
                                    'details': f'Extracted from channel: {parts}'
                                })
                        except Exception as e:
                            self.logger.debug(f"Zsteg channel {parts} extraction failed: {e}")
        except Exception as e:
            self.logger.debug(f"Zsteg extraction failed: {e}")
        
        return extracted

    async def _run_foremost_extraction(self, file_path: Path, output_dir: Path) -> List[Dict[str, Any]]:
        """Extract files using foremost"""
        output_dir.mkdir(parents=True, exist_ok=True)
        extracted = []
        
        try:
            cmd = ['foremost', '-i', str(file_path), '-o', str(output_dir)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            # Find extracted files
            for extracted_file in output_dir.rglob('*'):
                if extracted_file.is_file() and extracted_file.stat().st_size > 0:
                    extracted.append({
                        'file_path': str(extracted_file),
                        'method': 'foremost_carving',
                        'tool': 'foremost',
                        'details': 'Carved by foremost'
                    })
        except Exception as e:
            self.logger.debug(f"Foremost extraction failed: {e}")
        
        return extracted

    async def _run_scalpel_extraction(self, file_path: Path, output_dir: Path) -> List[Dict[str, Any]]:
        """Extract files using scalpel"""
        # Similar to foremost but using scalpel
        return []  # Implement if scalpel is available

    async def _extract_from_strings(self, file_path: Path, output_dir: Path) -> List[Dict[str, Any]]:
        """Extract potential files from strings output"""
        output_dir.mkdir(parents=True, exist_ok=True)
        extracted = []
        
        try:
            cmd = ['strings', '-n', '10', str(file_path)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.stdout:
                # Look for base64, hex, or other encoded data
                lines = result.stdout.strip().split('\n')
                for i, line in enumerate(lines):
                    # Check for base64-like strings (long alphanumeric with +/=)
                    if len(line) > 50 and any(c in line for c in '+/=') and line.replace('+', '').replace('/', '').replace('=', '').isalnum():
                        output_file = output_dir / f"strings_b64_{i}.dat"
                        try:
                            import base64
                            decoded = base64.b64decode(line + '==')  # Add padding
                            if len(decoded) > 10:
                                with open(output_file, 'wb') as f:
                                    f.write(decoded)
                                extracted.append({
                                    'file_path': str(output_file),
                                    'method': 'strings_base64',
                                    'tool': 'strings',
                                    'details': f'Base64 decoded string'
                                })
                        except:
                            pass
        except Exception as e:
            self.logger.debug(f"Strings extraction failed: {e}")
        
        return extracted

    async def _extract_from_exiftool(self, file_path: Path, output_dir: Path) -> List[Dict[str, Any]]:
        """Extract files from EXIF metadata"""
        output_dir.mkdir(parents=True, exist_ok=True)
        extracted = []
        
        try:
            cmd = ['exiftool', '-b', '-ThumbnailImage', str(file_path)]
            result = subprocess.run(cmd, capture_output=True, timeout=30)
            
            if result.stdout and len(result.stdout) > 100:
                output_file = output_dir / "exif_thumbnail.jpg"
                with open(output_file, 'wb') as f:
                    f.write(result.stdout)
                
                extracted.append({
                    'file_path': str(output_file),
                    'method': 'exif_thumbnail',
                    'tool': 'exiftool',
                    'details': 'Thumbnail extracted from EXIF'
                })
        except Exception as e:
            self.logger.debug(f"EXIF extraction failed: {e}")
        
        return extracted

    def _calculate_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except:
            return "unknown"

    def _get_mime_type(self, file_path: Path) -> str:
        """Get MIME type of file"""
        try:
            import magic
            return magic.from_file(str(file_path), mime=True)
        except:
            return "unknown"

    def _print_extraction_tree(self):
        """Print the extraction tree"""
        print(f"\n🌳 EXTRACTION TREE ({self.total_files_processed} files processed)")
        print("=" * 80)
        self._print_node(self.extraction_tree, "")

    def _print_node(self, node: ExtractionNode, prefix: str):
        """Recursively print tree node"""
        # File info
        size_str = f"{node.file_size:,} bytes" if node.file_size < 1024*1024 else f"{node.file_size/(1024*1024):.1f} MB"
        findings_str = f"({len(node.findings)} findings)" if node.findings else ""
        
        print(f"{prefix}├── {node.file_path.name} [{node.tool_name}] {size_str} {findings_str}")
        
        # Print findings
        for finding in node.findings[:3]:  # Show first 3 findings
            confidence = finding.get('confidence', 0)
            finding_type = finding.get('type', 'unknown')
            print(f"{prefix}│   └─ {finding_type} (confidence: {confidence:.2f})")
        
        if len(node.findings) > 3:
            print(f"{prefix}│   └─ ... {len(node.findings) - 3} more findings")
        
        # Print children
        for i, child in enumerate(node.children):
            child_prefix = prefix + ("│   " if i < len(node.children) - 1 else "    ")
            self._print_node(child, child_prefix)

    async def _save_extraction_tree(self):
        """Save extraction tree to JSON"""
        try:
            def serialize_node(node):
                data = asdict(node)
                data['file_path'] = str(node.file_path)
                data['parent_path'] = str(node.parent_path) if node.parent_path else None
                data['children'] = [serialize_node(child) for child in node.children]
                return data
            
            tree_file = self.output_dir / "extraction_tree.json"
            with open(tree_file, 'w') as f:
                json.dump(serialize_node(self.extraction_tree), f, indent=2)
            
            print(f"💾 Extraction tree saved to: {tree_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save extraction tree: {e}")

# Integration function to add to your main analyzer
async def run_cascading_analysis(file_path: str, orchestrator, session_id: str):
    """Run cascading analysis on a file"""
    print(f"🚀 Starting cascading analysis on {file_path}")
    print("This will extract files recursively and build a complete file tree...")
    
    cascading = CascadingAnalyzer(orchestrator)
    tree = await cascading.analyze_cascading(Path(file_path), session_id)
    
    print(f"\n🎉 Cascading analysis complete!")
    print(f"📊 Total files processed: {cascading.total_files_processed}")
    print(f"📁 Output directory: {cascading.output_dir}")
    print(f"🌳 Maximum depth reached: {tree.depth if tree else 0}")
    
    return tree

if __name__ == "__main__":
    print("This is a module for cascading analysis. Use it from your main analyzer.")
