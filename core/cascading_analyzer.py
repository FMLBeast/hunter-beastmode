"""
Cascading Analysis System for Deep Steganography Extraction
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
    def __init__(self, orchestrator, output_dir: Path = None, max_depth: int = 15, max_files: int = 10000):
        self.orchestrator = orchestrator
        self.logger = logging.getLogger(__name__)
        self.output_dir = output_dir or Path(tempfile.mkdtemp(prefix="cascading_analysis_"))
        self.extraction_tree = None
        self.processed_hashes: Set[str] = set()
        self.max_depth = max_depth
        self.max_files = max_files
        self.total_files_processed = 0
        
        print(f"🌳 Cascading analyzer initialized")
        print(f"   📁 Output: {self.output_dir}")
        print(f"   📊 Max depth: {max_depth}, Max files: {max_files}")

    async def analyze_cascading(self, initial_file: Path, session_id: str) -> ExtractionNode:
        """Start cascading analysis from initial file"""
        print(f"\n🚀 Starting deep extraction analysis...")
        print(f"📁 Target: {initial_file}")
        print(f"🔍 This will recursively extract and analyze all hidden files")
        
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
        
        # Print results
        self._print_extraction_tree()
        await self._save_extraction_tree()
        
        return self.extraction_tree

    async def _analyze_node_recursive(self, node: ExtractionNode, session_id: str):
        """Recursively analyze a node and its extracted files"""
        
        if node.depth >= self.max_depth:
            print(f"   ⚠️  Max depth {self.max_depth} reached")
            return
            
        if self.total_files_processed >= self.max_files:
            print(f"   ⚠️  Max files {self.max_files} processed")
            return
        
        self.total_files_processed += 1
        depth_prefix = "  " * node.depth
        
        print(f"{depth_prefix}🔍 [{node.depth}] Analyzing {node.file_path.name} ({node.file_size:,} bytes)")
        
        try:
            # Run standard analysis
            results = await self.orchestrator.analyze(node.file_path, session_id)
            node.findings.extend(results or [])
            
            # Extract files using tools
            extracted_files = await self._extract_files_from_node(node)
            
            if extracted_files:
                print(f"{depth_prefix}   ✅ Found {len(extracted_files)} embedded files!")
                
                # Create child nodes and analyze them
                for extracted_file_info in extracted_files:
                    child_node = await self._create_child_node(node, extracted_file_info)
                    if child_node:
                        node.children.append(child_node)
                        await self._analyze_node_recursive(child_node, session_id)
            else:
                print(f"{depth_prefix}   ⚪ No embedded files found")
                
        except Exception as e:
            print(f"{depth_prefix}   ❌ Error: {e}")

    async def _extract_files_from_node(self, node: ExtractionNode) -> List[Dict[str, Any]]:
        """Extract files from a node using all available tools"""
        all_extracted = []
        
        # Create extraction directory
        node_dir = self.output_dir / f"depth_{node.depth}" / node.file_path.stem
        node_dir.mkdir(parents=True, exist_ok=True)
        
        # Try each extraction tool
        tools = {
            'binwalk': self._run_binwalk_extraction,
            'steghide': self._run_steghide_extraction,
            'zsteg': self._run_zsteg_extraction,
            'foremost': self._run_foremost_extraction
        }
        
        for tool_name, extraction_func in tools.items():
            try:
                extracted = await extraction_func(node.file_path, node_dir / tool_name)
                if extracted:
                    all_extracted.extend(extracted)
            except Exception as e:
                self.logger.debug(f"   {tool_name}: {e}")
        
        return all_extracted

    async def _create_child_node(self, parent: ExtractionNode, extracted_info: Dict[str, Any]) -> Optional[ExtractionNode]:
        """Create a child node from extracted file info"""
        extracted_path = Path(extracted_info['file_path'])
        
        if not extracted_path.exists() or extracted_path.stat().st_size == 0:
            return None
        
        # Avoid duplicates
        file_hash = self._calculate_hash(extracted_path)
        if file_hash in self.processed_hashes:
            return None
        
        self.processed_hashes.add(file_hash)
        
        return ExtractionNode(
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

    async def _run_binwalk_extraction(self, file_path: Path, output_dir: Path) -> List[Dict[str, Any]]:
        """Extract files using binwalk"""
        output_dir.mkdir(parents=True, exist_ok=True)
        extracted = []
        
        try:
            cmd = ['binwalk', '--extract', '--directory', str(output_dir), str(file_path)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                for extracted_file in output_dir.rglob('*'):
                    if extracted_file.is_file() and extracted_file.stat().st_size > 0:
                        extracted.append({
                            'file_path': str(extracted_file),
                            'method': 'binwalk_extraction',
                            'tool': 'binwalk'
                        })
        except Exception:
            pass
        
        return extracted

    async def _run_steghide_extraction(self, file_path: Path, output_dir: Path) -> List[Dict[str, Any]]:
        """Extract files using steghide"""
        output_dir.mkdir(parents=True, exist_ok=True)
        extracted = []
        
        passwords = ['', 'password', 'flag', 'secret', 'hidden', 'ctf', '123456']
        
        for i, password in enumerate(passwords):
            try:
                output_file = output_dir / f"steghide_{i}.dat"
                cmd = ['steghide', 'extract', '-sf', str(file_path), '-xf', str(output_file), '-p', password]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0 and output_file.exists() and output_file.stat().st_size > 0:
                    extracted.append({
                        'file_path': str(output_file),
                        'method': 'steghide_extraction',
                        'tool': 'steghide'
                    })
                    break
            except Exception:
                pass
        
        return extracted

    async def _run_zsteg_extraction(self, file_path: Path, output_dir: Path) -> List[Dict[str, Any]]:
        """Extract files using zsteg"""
        output_dir.mkdir(parents=True, exist_ok=True)
        extracted = []
        
        try:
            cmd = ['zsteg', '--all', str(file_path)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.stdout:
                lines = result.stdout.strip().split('\n')
                for i, line in enumerate(lines[:10]):  # Limit to first 10 extractions
                    if len(line) > 50:  # Likely contains data
                        output_file = output_dir / f"zsteg_{i}.dat"
                        
                        try:
                            parts = line.split(':')[0].strip() if ':' in line else f"b{i},lsb,xy"
                            extract_cmd = ['zsteg', '-E', parts, str(file_path)]
                            extract_result = subprocess.run(extract_cmd, capture_output=True, timeout=30)
                            
                            if extract_result.stdout and len(extract_result.stdout) > 10:
                                with open(output_file, 'wb') as f:
                                    f.write(extract_result.stdout)
                                
                                extracted.append({
                                    'file_path': str(output_file),
                                    'method': 'zsteg_extraction',
                                    'tool': 'zsteg'
                                })
                        except Exception:
                            pass
        except Exception:
            pass
        
        return extracted

    async def _run_foremost_extraction(self, file_path: Path, output_dir: Path) -> List[Dict[str, Any]]:
        """Extract files using foremost"""
        output_dir.mkdir(parents=True, exist_ok=True)
        extracted = []
        
        try:
            cmd = ['foremost', '-i', str(file_path), '-o', str(output_dir)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            for extracted_file in output_dir.rglob('*'):
                if extracted_file.is_file() and extracted_file.stat().st_size > 0:
                    extracted.append({
                        'file_path': str(extracted_file),
                        'method': 'foremost_carving',
                        'tool': 'foremost'
                    })
        except Exception:
            pass
        
        return extracted

    def _calculate_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()[:16]  # Short hash
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
        print(f"\n🌳 EXTRACTION TREE COMPLETE")
        print("=" * 80)
        print(f"📊 Total files processed: {self.total_files_processed}")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"🎯 Maximum depth reached: {self._get_max_depth(self.extraction_tree)}")
        print()
        self._print_node(self.extraction_tree, "")

    def _print_node(self, node: ExtractionNode, prefix: str):
        """Recursively print tree node"""
        size_str = f"{node.file_size:,} bytes" if node.file_size < 1024*1024 else f"{node.file_size/(1024*1024):.1f} MB"
        findings_str = f"({len(node.findings)} findings)" if node.findings else ""
        
        print(f"{prefix}├── {node.file_path.name} [{node.tool_name}] {size_str} {findings_str}")
        
        for finding in node.findings[:2]:  # Show first 2 findings
            confidence = finding.get('confidence', 0)
            finding_type = finding.get('type', 'unknown')
            print(f"{prefix}│   └─ {finding_type} (confidence: {confidence:.2f})")
        
        if len(node.findings) > 2:
            print(f"{prefix}│   └─ ... {len(node.findings) - 2} more findings")
        
        for i, child in enumerate(node.children):
            child_prefix = prefix + ("│   " if i < len(node.children) - 1 else "    ")
            self._print_node(child, child_prefix)

    def _get_max_depth(self, node: ExtractionNode) -> int:
        """Get maximum depth in tree"""
        if not node.children:
            return node.depth
        return max(self._get_max_depth(child) for child in node.children)

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
            
            print(f"💾 Complete tree saved to: {tree_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save tree: {e}")
