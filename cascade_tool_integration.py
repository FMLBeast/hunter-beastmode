#!/usr/bin/env python3
"""
Cascade Integration Script – Ensures extracted items go through full pipeline
"""
import re
from pathlib import Path
import shutil

def main():
    """Integrate cascading analysis into the main analyzer pipeline."""
    print("🔧 Integrating cascading analysis system...")
    project_root = Path(".")

    # Step 1: Fix database storage (if needed)
    fix_database_storage(project_root)

    # Step 2: Add cascading analysis module
    add_cascading_module(project_root)

    # Step 3: Integrate cascade option with main script
    integrate_with_main(project_root)

    print("✅ Integration complete!\n")
    print("🚀 NEW CAPABILITIES:")
    print("   • Deep recursive extraction with automatic analysis of each extracted file")
    print("   • File tree visualization of embedded content")
    print("   • Handles thousands of nested files (configurable depth and file limits)\n")
    print("🎯 USAGE:")
    print("   python3 steg_main.py <file> --cascade")
    print("   python3 steg_main.py <file> --cascade --max-depth 10 --max-files 5000\n")

def fix_database_storage(project_root: Path):
    """Fix the database storage method to accept cascade results properly."""
    print("🗄️  Fixing database storage...")
    db_file = project_root / "core" / "database.py"
    if not db_file.exists():
        print("   ⚠️  Database file not found, skipping DB fix.")
        return

    with open(db_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Identify the broken store_analysis_result (if it has a missing argument issue)
    if 'async def store_analysis_result' in content and 'await self.store_finding(' in content:
        # Replace the entire store_analysis_result method with corrected version
        method_pattern = r'    async def store_analysis_result\(.*?\):.*?(?=\n    async def|\n    def|\nclass|\Z)'
        new_method = '''    async def store_analysis_result(self, session_id: str, method: str, results: list):
        """Store analysis results from tools (fixed version)"""
        if not results:
            return
        try:
            # Determine file_id for this session (last inserted file)
            file_id = None
            if self.db_type == "sqlite":
                cursor = self.sqlite_conn.cursor()
                cursor.execute("SELECT id FROM files WHERE session_id = ? ORDER BY created_at DESC LIMIT 1", (session_id,))
                row = cursor.fetchone()
                file_id = row[0] if row else None
                if not file_id:
                    self.logger.warning(f"No file record found for session {session_id}")
                    return
            # Store each result as a finding linked to the file
            for result in results:
                if isinstance(result, dict):
                    await self.store_finding(session_id, file_id, result)
        except Exception as e:
            self.logger.error(f"Error storing analysis results for method {method}: {e}")'''
        patched_content = re.sub(method_pattern, new_method, content, flags=re.DOTALL)
        if patched_content != content:
            with open(db_file, 'w', encoding='utf-8') as f:
                f.write(patched_content)
            print("   ✅ Database storage method updated.")
    else:
        print("   ℹ️  Database storage method already up-to-date or not applicable.")

def add_cascading_module(project_root: Path):
    """Add the CascadingAnalyzer module for recursive cascade analysis."""
    print("🌳 Adding cascading analysis module...")
    core_dir = project_root / "core"
    core_dir.mkdir(exist_ok=True)
    cascading_file = core_dir / "cascading_analyzer.py"

    # Define the CascadingAnalyzer code
    cascading_code = r'''"""
Cascading Analysis Module – Deep recursive steganography analyzer
Automatically extracts hidden files and analyzes each with the full pipeline.
"""
import asyncio, logging, shutil, tempfile, json, hashlib, time, subprocess
from pathlib import Path
from typing import Dict, List, Any, Set, Optional
from dataclasses import dataclass, asdict

@dataclass
class ExtractionNode:
    """Represents a file (and its findings) in the extraction tree."""
    file_path: Path
    parent_path: Optional[Path]
    extraction_method: str  # how this file was extracted
    tool_name: str          # which tool extracted it (or 'user_input' for root)
    depth: int
    file_hash: str
    file_size: int
    mime_type: str
    children: List["ExtractionNode"]
    findings: List[Dict[str, Any]]
    extracted_at: float

class CascadingAnalyzer:
    def __init__(self, orchestrator, output_dir: Path = None, max_depth: int = 15, max_files: int = 10000):
        self.orchestrator = orchestrator
        self.logger = logging.getLogger(__name__)
        self.output_dir = output_dir or Path(tempfile.mkdtemp(prefix="cascading_analysis_"))
        self.extraction_tree: Optional[ExtractionNode] = None
        self.processed_hashes: Set[str] = set()
        self.max_depth = max_depth
        self.max_files = max_files
        self.total_files_processed = 0

        print(f"🌳 CascadingAnalyzer initialized (max_depth={max_depth}, max_files={max_files})")
        print(f"   📁 Output directory: {self.output_dir}")

    async def analyze_cascading(self, initial_file: Path, session_id: str) -> ExtractionNode:
        """Start the cascading analysis from an initial file."""
        print(f"\\n🚀 Starting cascade analysis for {initial_file} ...")
        # Create the root extraction node
        file_hash = self._calculate_hash(initial_file)
        root_node = ExtractionNode(
            file_path=initial_file,
            parent_path=None,
            extraction_method="initial",
            tool_name="user_input",
            depth=0,
            file_hash=file_hash,
            file_size=initial_file.stat().st_size if initial_file.exists() else 0,
            mime_type=self._get_mime_type(initial_file),
            children=[],
            findings=[],
            extracted_at=time.time()
        )
        self.extraction_tree = root_node
        self.processed_hashes.add(file_hash)

        # Analyze the root file with full pipeline, then recursively extract/analyze children
        await self._analyze_node_recursive(root_node, session_id)

        # After recursion, print and save the extraction tree
        self._print_extraction_tree()
        await self._save_extraction_tree()
        return self.extraction_tree

    async def _analyze_node_recursive(self, node: ExtractionNode, session_id: str):
        """Recursively analyze a node: run full analysis, extract children, and repeat."""
        if node.depth >= self.max_depth:
            print(f"   ⚠️  Max depth {self.max_depth} reached at {node.file_path}")
            return
        if self.total_files_processed >= self.max_files:
            print(f"   ⚠️  Reached max files limit ({self.max_files}), stopping recursion.")
            return

        self.total_files_processed += 1
        indent = "  " * node.depth
        print(f"{indent}🔍 Analyzing {node.file_path.name} (depth {node.depth}, size {node.file_size} bytes)")

        try:
            # 1. Run the standard analysis pipeline on this file
            results = await self.orchestrator.analyze(node.file_path, session_id)
            node.findings.extend(results or [])
            if results:
                print(f"{indent}   ↪️ Collected {len(results)} findings from main analysis")

            # 2. Extract embedded files using various tools
            extracted_files = await self._extract_files_from(node)
            if extracted_files:
                print(f"{indent}   ✅ Extracted {len(extracted_files)} file(s) from {node.file_path.name}")
            else:
                print(f"{indent}   ⚪ No embedded files found in {node.file_path.name}")

            # 3. Recurse into each extracted file
            for info in extracted_files:
                child = await self._create_child_node(node, info)
                if child:
                    node.children.append(child)
                    await self._analyze_node_recursive(child, session_id)
        except Exception as e:
            print(f"{indent}   ❌ Error analyzing {node.file_path.name}: {e}")

    async def _extract_files_from(self, node: ExtractionNode) -> List[Dict[str, Any]]:
        """Try extracting embedded files from the given node using multiple tools."""
        extracted = []
        work_dir = self.output_dir / f"depth_{node.depth}" / node.file_path.stem
        work_dir.mkdir(parents=True, exist_ok=True)
        # Define extraction methods to apply
        extractors = {
            'binwalk': self._run_binwalk,
            'steghide': self._run_steghide,
            'zsteg': self._run_zsteg,
            'foremost': self._run_foremost
        }
        for tool, func in extractors.items():
            try:
                new_files = await func(node.file_path, work_dir / tool)
                if new_files:
                    extracted.extend(new_files)
            except Exception as ex:
                self.logger.debug(f"Extractor {tool} failed: {ex}")
        return extracted

    async def _create_child_node(self, parent: ExtractionNode, info: Dict[str, Any]) -> Optional[ExtractionNode]:
        """Create an ExtractionNode for a newly extracted file."""
        path = Path(info['file_path'])
        if not path.exists() or path.stat().st_size == 0:
            return None
        # Avoid processing the same file twice (by content hash)
        file_hash = self._calculate_hash(path)
        if file_hash in self.processed_hashes:
            return None
        self.processed_hashes.add(file_hash)
        return ExtractionNode(
            file_path=path,
            parent_path=parent.file_path,
            extraction_method=info.get('method', 'unknown'),
            tool_name=info.get('tool', 'unknown'),
            depth=parent.depth + 1,
            file_hash=file_hash,
            file_size=path.stat().st_size,
            mime_type=self._get_mime_type(path),
            children=[],
            findings=[],
            extracted_at=time.time()
        )

    # Extraction tool implementations:
    async def _run_binwalk(self, file_path: Path, out_dir: Path) -> List[Dict[str, Any]]:
        """Extract using binwalk."""
        out_dir.mkdir(parents=True, exist_ok=True)
        extracted = []
        try:
            cmd = ['binwalk', '--extract', '--directory', str(out_dir), str(file_path)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                # Collect all files extracted by binwalk
                for item in out_dir.rglob('*'):
                    if item.is_file() and item.stat().st_size > 0:
                        extracted.append({
                            'file_path': str(item),
                            'method': 'binwalk_extraction',
                            'tool': 'binwalk'
                        })
        except Exception:
            pass
        return extracted

    async def _run_steghide(self, file_path: Path, out_dir: Path) -> List[Dict[str, Any]]:
        """Extract using steghide (with a small dictionary of passwords)."""
        out_dir.mkdir(parents=True, exist_ok=True)
        extracted = []
        passwords = ["", "password", "secret", "hidden", "flag", "ctf", "123456"]
        for i, pwd in enumerate(passwords):
            try:
                output_file = out_dir / f"steghide_out_{i}.dat"
                cmd = ['steghide', 'extract', '-sf', str(file_path), '-xf', str(output_file), '-p', pwd]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                if result.returncode == 0 and output_file.exists() and output_file.stat().st_size > 0:
                    extracted.append({
                        'file_path': str(output_file),
                        'method': 'steghide_extraction',
                        'tool': 'steghide'
                    })
                    break  # stop after first successful extraction
            except Exception:
                continue
        return extracted

    async def _run_zsteg(self, file_path: Path, out_dir: Path) -> List[Dict[str, Any]]:
        """Extract data using zsteg (for image files)."""
        out_dir.mkdir(parents=True, exist_ok=True)
        extracted = []
        try:
            # Run zsteg in verbose mode to list potential data (limit to first 10 findings)
            result = subprocess.run(['zsteg', '--all', str(file_path)], capture_output=True, text=True, timeout=60)
            if result.stdout:
                lines = result.stdout.splitlines()
                for i, line in enumerate(lines[:10]):  # consider first 10 possible extractions
                    if ':' in line:
                        # Each line like "b1,rgb,lsb,xy : <data>" – extract with that parameter
                        param = line.split(':', 1)[0].strip()
                        output_file = out_dir / f"zsteg_{i}.dat"
                        try:
                            extract_cmd = ['zsteg', '-E', param, str(file_path)]
                            extract_proc = subprocess.run(extract_cmd, capture_output=True, timeout=30)
                            data = extract_proc.stdout
                            if data and len(data) > 0:
                                with open(output_file, 'wb') as f:
                                    f.write(data)
                                if output_file.stat().st_size > 0:
                                    extracted.append({
                                        'file_path': str(output_file),
                                        'method': f'zsteg[{param}]',
                                        'tool': 'zsteg'
                                    })
                        except Exception:
                            continue
        except Exception:
            pass
        return extracted

    async def _run_foremost(self, file_path: Path, out_dir: Path) -> List[Dict[str, Any]]:
        """Carve out files using foremost."""
        out_dir.mkdir(parents=True, exist_ok=True)
        extracted = []
        try:
            cmd = ['foremost', '-i', str(file_path), '-o', str(out_dir)]
            subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            # Collect carved files (foremost sorts by file type folders)
            for item in out_dir.rglob('*'):
                if item.is_file() and item.stat().st_size > 0:
                    extracted.append({
                        'file_path': str(item),
                        'method': 'foremost_carving',
                        'tool': 'foremost'
                    })
        except Exception:
            pass
        return extracted

    def _calculate_hash(self, file_path: Path) -> str:
        """Calculate a short SHA-256 hash of a file (for deduplication)."""
        try:
            sha = hashlib.sha256()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha.update(chunk)
            return sha.hexdigest()[:16]
        except Exception:
            return "unknown"

    def _get_mime_type(self, file_path: Path) -> str:
        """Get MIME type of the file (if python-magic is available)."""
        try:
            import magic
            return magic.from_file(str(file_path), mime=True)
        except Exception:
            return "unknown"

    def _print_extraction_tree(self):
        """Print the hierarchical tree of extracted files and findings."""
        print(f"\\n🌳 **Extraction Tree** (Total files processed: {self.total_files_processed})")
        print("-" * 60)
        if not self.extraction_tree:
            return
        def print_node(node: ExtractionNode, prefix: str = ""):
            size = node.file_size
            size_str = f"{size:,} bytes" if size < 1024*1024 else f"{size/1024**2:.1f} MB"
            finding_count = len(node.findings)
            find_str = f"{finding_count} finding(s)" if finding_count else "no findings"
            print(f"{prefix}- {node.file_path.name} [{node.tool_name}] ({size_str}, {find_str})")
            # Print a couple of findings for context
            for f in node.findings[:2]:
                ftype = f.get('type', 'finding')
                conf = f.get('confidence', 0)
                print(f"{prefix}  * {ftype} (confidence {conf:.2f})")
            if len(node.findings) > 2:
                print(f"{prefix}  * ... ({len(node.findings) - 2} more findings)")
            # Recurse into children
            for child in node.children:
                print_node(child, prefix + "    ")
        print_node(self.extraction_tree)

    async def _save_extraction_tree(self):
        """Save the extraction tree structure and findings to a JSON file."""
        if not self.extraction_tree:
            return
        tree_path = self.output_dir / "extraction_tree.json"
        def node_to_dict(node: ExtractionNode) -> dict:
            data = asdict(node)
            data['file_path'] = str(node.file_path)
            data['parent_path'] = str(node.parent_path) if node.parent_path else None
            data['children'] = [node_to_dict(c) for c in node.children]
            return data
        with open(tree_path, 'w') as f:
            json.dump(node_to_dict(self.extraction_tree), f, indent=2)
        print(f"💾 Extraction tree saved to {tree_path}")
'''
    # Write the cascading analyzer module to file
    with open(cascading_file, 'w', encoding='utf-8') as f:
        f.write(cascading_code)
    print(f"   ✅ Created {cascading_file.name}")

def integrate_with_main(project_root: Path):
    """Integrate cascade option into the main CLI script (steg_main.py)."""
    print("🔗 Integrating cascading analysis into steg_main.py...")
    main_file = project_root / "steg_main.py"
    if not main_file.exists():
        print("   ⚠️  Main script steg_main.py not found, skipping CLI integration.")
        return

    with open(main_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Add CLI arguments for cascade if not present
    if '--cascade' not in content:
        args_insertion = (
            '    parser.add_argument("--cascade", action="store_true", help="Enable recursive cascade analysis")\n'
            '    parser.add_argument("--max-depth", type=int, default=10, help="Maximum extraction depth for cascade")\n'
            '    parser.add_argument("--max-files", type=int, default=5000, help="Maximum number of files to process in cascade")'
        )
        # Insert after the verbose argument definition
        content = re.sub(r'parser\.add_argument\("--verbose".*?\n', 
                         lambda m: m.group(0) + args_insertion + "\n", content, count=1)

    # 2. Ensure CascadingAnalyzer is imported
    if 'from core.cascading_analyzer import CascadingAnalyzer' not in content:
        content = content.replace('from core.orchestrator import StegOrchestrator\n', 
                                  'from core.orchestrator import StegOrchestrator\nfrom core.cascading_analyzer import CascadingAnalyzer\n')

    # 3. Add logic in main() to invoke cascade analysis after normal analysis
    if 'if args.cascade' not in content:
        pattern = r'if target_path\.is_file\(\):\s*results = await analyzer\.analyze_file\([^)]*\)'
        replacement = (
            'if target_path.is_file():\n'
            '            results = await analyzer.analyze_file(str(target_path))\n'
            '            \n'
            '            # If cascade mode, perform deep analysis on extracted files\n'
            '            if args.cascade:\n'
            '                print("\\n🌳 Starting cascading analysis...")\n'
            '                casc = CascadingAnalyzer(analyzer.orchestrator, max_depth=args.max_depth, max_files=args.max_files)\n'
            '                tree = await casc.analyze_cascading(target_path, results["session_id"])\n'
            '                print("🎉 Cascade analysis complete. See extraction_tree.json for details.")'
        )
        content = re.sub(pattern, replacement, content, flags=re.DOTALL)

    # Write the updated main file
    with open(main_file, 'w', encoding='utf-8') as f:
        f.write(content)
    print("   ✅ steg_main.py updated with cascade integration")

if __name__ == "__main__":
    main()
