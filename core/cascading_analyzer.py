import asyncio, logging, shutil, tempfile, json, hashlib, time, subprocess
from pathlib import Path
from typing import Dict, List, Any, Set, Optional
from dataclasses import dataclass, asdict
import magic
import math

@dataclass
class ExtractionNode:
    file_path: Path
    parent_path: Optional[Path]
    extraction_method: str
    tool_name: str
    depth: int
    file_hash: str
    file_size: int
    mime_type: str
    children: List["ExtractionNode"]
    findings: List[Dict[str, Any]]
    extracted_at: float
    entropy: Optional[float] = None
    trap_flag: bool = False

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
        self.max_children_per_tool = 5
        self.seen_filenames: Dict[str, int] = {}

    async def analyze_cascading(self, initial_file: Path, session_id: str) -> ExtractionNode:
        root_node = await self._create_node(None, initial_file, "initial", "user_input", 0)
        if not root_node:
            raise Exception("Failed to initialize root node")
        self.extraction_tree = root_node
        await self._analyze_node_recursive(root_node, session_id)
        await self._save_tree()
        return self.extraction_tree

    async def _analyze_node_recursive(self, node: ExtractionNode, session_id: str):
        if node.depth >= self.max_depth or self.total_files_processed >= self.max_files:
            return

        self.total_files_processed += 1
        findings = await self.orchestrator.analyze(node.file_path, session_id)
        node.findings.extend(findings or [])
        extracted_files = await self._extract_files(node)

        per_tool_counter = {}
        for info in extracted_files:
            tool = info.get("tool", "generic")
            per_tool_counter[tool] = per_tool_counter.get(tool, 0) + 1
            if per_tool_counter[tool] > self.max_children_per_tool:
                continue
            child = await self._create_node(node, Path(info['file_path']), info['method'], tool, node.depth + 1)
            if child:
                node.children.append(child)
                await self._analyze_node_recursive(child, session_id)

    async def _extract_files(self, node: ExtractionNode) -> List[Dict[str, Any]]:
        work_dir = self.output_dir / f"depth_{node.depth}" / node.file_path.stem
        work_dir.mkdir(parents=True, exist_ok=True)
        extractors = {
            'binwalk': self._run_binwalk,
            'steghide': self._run_steghide,
            'zsteg': self._run_zsteg,
            'foremost': self._run_foremost
        }
        extracted = []
        for tool, func in extractors.items():
            try:
                files = await func(node.file_path, work_dir / tool)
                extracted.extend(files)
            except Exception:
                continue
        return extracted

    async def _create_node(self, parent: Optional[ExtractionNode], path: Path, method: str, tool: str, depth: int) -> Optional[ExtractionNode]:
        if not path.exists() or path.stat().st_size == 0:
            return None
        hash_val = self._hash(path)
        if hash_val in self.processed_hashes:
            return None
        self.processed_hashes.add(hash_val)
        mime = self._mime(path)
        ent = self._entropy(path)
        trap = self._detect_trap(path.name, ent)
        return ExtractionNode(
            file_path=path,
            parent_path=parent.file_path if parent else None,
            extraction_method=method,
            tool_name=tool,
            depth=depth,
            file_hash=hash_val,
            file_size=path.stat().st_size,
            mime_type=mime,
            children=[],
            findings=[],
            extracted_at=time.time(),
            entropy=ent,
            trap_flag=trap
        )

    def _hash(self, path):
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            while chunk := f.read(8192):
                h.update(chunk)
        return h.hexdigest()

    def _mime(self, path):
        try:
            return magic.from_file(str(path), mime=True)
        except:
            return "unknown"

    def _entropy(self, path):
        try:
            data = path.read_bytes()
            if not data:
                return 0.0
            prob = [float(data.count(b)) / len(data) for b in set(data)]
            return -sum(p * math.log2(p) for p in prob)
        except:
            return None

    def _detect_trap(self, name, entropy):
        if entropy and (entropy > 7.95 or entropy < 1.0):
            return True
        self.seen_filenames[name] = self.seen_filenames.get(name, 0) + 1
        if self.seen_filenames[name] > 3:
            return True
        return False

    async def _run_binwalk(self, file_path: Path, out_dir: Path) -> List[Dict[str, Any]]:
        out_dir.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(['binwalk', '--extract', '--directory', str(out_dir), str(file_path)], capture_output=True)
        return self._collect_files(out_dir, 'binwalk')

    async def _run_steghide(self, file_path: Path, out_dir: Path) -> List[Dict[str, Any]]:
        out_dir.mkdir(parents=True, exist_ok=True)
        passwords = ["", "hunter", "password"]
        found = []
        for i, pwd in enumerate(passwords):
            out = out_dir / f"steghide_{i}.out"
            cmd = ['steghide', 'extract', '-sf', str(file_path), '-xf', str(out), '-p', pwd]
            result = subprocess.run(cmd, capture_output=True)
            if out.exists():
                found.append({'file_path': str(out), 'method': 'steghide', 'tool': 'steghide'})
        return found

    async def _run_zsteg(self, file_path: Path, out_dir: Path) -> List[Dict[str, Any]]:
        out_dir.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(['zsteg', '--all', str(file_path)], capture_output=True, text=True)
        lines = result.stdout.splitlines()
        found = []
        for i, line in enumerate(lines[:5]):
            if ':' not in line:
                continue
            param = line.split(':')[0].strip()
            out = out_dir / f"zsteg_{i}.dat"
            cmd = ['zsteg', '-E', param, str(file_path)]
            r = subprocess.run(cmd, capture_output=True)
            if r.stdout:
                with open(out, 'wb') as f:
                    f.write(r.stdout)
                if out.exists():
                    found.append({'file_path': str(out), 'method': f'zsteg[{param}]', 'tool': 'zsteg'})
        return found

    async def _run_foremost(self, file_path: Path, out_dir: Path) -> List[Dict[str, Any]]:
        out_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(['foremost', '-i', str(file_path), '-o', str(out_dir)], capture_output=True)
        return self._collect_files(out_dir, 'foremost')

    def _collect_files(self, root: Path, tool: str) -> List[Dict[str, Any]]:
        files = []
        for f in root.rglob('*'):
            if f.is_file() and f.stat().st_size > 0:
                files.append({'file_path': str(f), 'method': f'{tool}_extraction', 'tool': tool})
        return files

    async def _save_tree(self):
        if not self.extraction_tree:
            return
        out = self.output_dir / "extraction_tree.json"
        def flatten(node):
            d = asdict(node)
            d['file_path'] = str(node.file_path)
            d['parent_path'] = str(node.parent_path) if node.parent_path else None
            d['children'] = [flatten(c) for c in node.children]
            return d
        with open(out, 'w') as f:
            json.dump(flatten(self.extraction_tree), f, indent=2)
        print(f"💾 Extraction tree saved to {out}")
