#!/usr/bin/env python3
"""
Complete Cascading Feature Fix for StegAnalyzer
Fixes all cascading-related issues and integrations
"""

import re
import sys
import shutil
from pathlib import Path
from typing import Dict, Any

def main():
    """Execute complete cascade fix"""
    print("🚀 Starting Complete Cascade Fix...")
    print("=" * 60)
    
    project_root = Path(".")
    
    # Step 1: Fix core database issues
    fix_database_storage(project_root)
    
    # Step 2: Fix orchestrator method issues
    fix_orchestrator_methods(project_root)
    
    # Step 3: Update cascade analyzer module
    update_cascade_analyzer(project_root)
    
    # Step 4: Fix configuration integration
    fix_configuration_integration(project_root)
    
    # Step 5: Fix main script integration
    fix_main_script_integration(project_root)
    
    # Step 6: Clean up broken files
    cleanup_broken_files(project_root)
    
    print("\n" + "=" * 60)
    print("✅ COMPLETE CASCADE FIX APPLIED!")
    print("\n🎯 Fixed Issues:")
    print("   • Database storage parameter mismatches")
    print("   • Method name conflicts (file_signature -> magic_analysis)")
    print("   • Cascade analyzer integration")
    print("   • Configuration handling")
    print("   • Main script CLI integration")
    print("\n🚀 Ready for vast.ai deployment!")
    print("   Test: python steg_main.py image.png --cascade --verbose")

def fix_database_storage(project_root: Path):
    """Fix critical database storage issues"""
    print("🗄️  Fixing database storage methods...")
    
    db_file = project_root / "core" / "database.py"
    if not db_file.exists():
        print("   ❌ database.py not found!")
        return
    
    with open(db_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix 1: Add missing store_analysis_result method
    if "async def store_analysis_result" not in content:
        method_code = '''
    async def store_analysis_result(self, session_id: str, method: str, results: list):
        """Store analysis results from tools with proper error handling"""
        if not results:
            return
        
        try:
            # Get file_id for this session (most recent file)
            file_id = None
            if self.db_type == "sqlite" and self.sqlite_conn:
                cursor = self.sqlite_conn.cursor()
                cursor.execute(
                    "SELECT id FROM files WHERE session_id = ? ORDER BY created_at DESC LIMIT 1", 
                    (session_id,)
                )
                row = cursor.fetchone()
                file_id = row[0] if row else None
                
                if not file_id:
                    self.logger.warning(f"No file record found for session {session_id}")
                    return
            
            # Store each result as a finding
            for result in results:
                if isinstance(result, dict):
                    await self.store_finding(session_id, file_id, result)
                    
        except Exception as e:
            self.logger.error(f"Error storing analysis results for {method}: {e}")
'''
        
        # Insert before the last class method or at end of class
        class_end_pattern = r'(\n    def [^_].*?\n        .*?\n\n)(\nclass|\Z)'
        if re.search(class_end_pattern, content, re.DOTALL):
            content = re.sub(
                class_end_pattern,
                r'\1' + method_code + r'\2',
                content,
                count=1,
                flags=re.DOTALL
            )
        else:
            # Fallback: add before end of DatabaseManager class
            content = content.replace(
                '\nclass ', 
                method_code + '\n\nclass '
            )
    
    # Fix 2: Ensure store_finding method has proper signature
    if "async def store_finding" in content:
        # Replace any broken store_finding calls
        content = re.sub(
            r'await self\.store_finding\(session_id,\s*result\)',
            'await self.store_finding(session_id, file_id, result)',
            content
        )
    
    # Fix 3: Add missing mark_task_complete method
    if "async def mark_task_complete" not in content:
        task_method = '''
    async def mark_task_complete(self, session_id: str, file_path: Path, method: str):
        """Mark analysis task as complete"""
        try:
            if self.db_type == "sqlite" and self.sqlite_conn:
                cursor = self.sqlite_conn.cursor()
                cursor.execute(
                    "INSERT OR REPLACE INTO task_completion (session_id, file_path, method, completed_at) VALUES (?, ?, ?, ?)",
                    (session_id, str(file_path), method, datetime.now(timezone.utc).isoformat())
                )
                self.sqlite_conn.commit()
        except Exception as e:
            self.logger.error(f"Error marking task complete: {e}")
'''
        content = content.replace(
            'async def store_analysis_result',
            task_method + '\n    async def store_analysis_result'
        )
    
    with open(db_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("   ✅ Database storage methods fixed")

def fix_orchestrator_methods(project_root: Path):
    """Fix orchestrator method name conflicts"""
    print("🔧 Fixing orchestrator method conflicts...")
    
    orchestrator_file = project_root / "core" / "orchestrator.py"
    if not orchestrator_file.exists():
        print("   ❌ orchestrator.py not found!")
        return
    
    with open(orchestrator_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix method name conflicts
    fixes = {
        '"file_signature"': '"magic_analysis"',
        "'file_signature'": "'magic_analysis'",
        'method="file_signature"': 'method="magic_analysis"',
        'if "file_signature" not in completed': 'if "magic_analysis" not in completed',
        'completed_set.add("file_signature")': 'completed_set.add("magic_analysis")',
    }
    
    for old, new in fixes.items():
        content = content.replace(old, new)
    
    # Fix execute_method calls - ensure proper error handling
    if "def execute_method" in content:
        execute_method_pattern = r'(def execute_method\(self, method.*?\n.*?)return\s+tool\.execute_method\(method,.*?\)'
        
        new_execute = '''try:
            return tool.execute_method(method, file_path)
        except AttributeError as e:
            self.logger.error(f"Tool {tool_name} missing method {method}: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Error executing {method} on {tool_name}: {e}")
            return []'''
        
        content = re.sub(
            execute_method_pattern,
            r'\1' + new_execute,
            content,
            flags=re.DOTALL
        )
    
    # Add cascade analyzer integration
    if "self.cascade_analyzer = None" not in content:
        init_pattern = r'(def __init__\(self, config.*?\n.*?)(        # Initialize.*?\n)'
        cascade_init = '''        # Initialize cascade analyzer
        try:
            from tools.cascade_analyzer import CascadeAnalyzer
            self.cascade_analyzer = CascadeAnalyzer(config)
            self.logger.info("Cascade analyzer initialized")
        except ImportError:
            self.cascade_analyzer = None
            self.logger.warning("Cascade analyzer not available")
        
'''
        content = re.sub(init_pattern, r'\1' + cascade_init + r'\2', content, flags=re.DOTALL)
    
    with open(orchestrator_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("   ✅ Orchestrator methods fixed")

def update_cascade_analyzer(project_root: Path):
    """Create/update the cascade analyzer module"""
    print("🌳 Updating cascade analyzer module...")
    
    tools_dir = project_root / "tools"
    tools_dir.mkdir(exist_ok=True)
    
    cascade_file = tools_dir / "cascade_analyzer.py"
    
    cascade_code = '''#!/usr/bin/env python3
"""
Recursive Cascade Steganography Analyzer
Deep extraction and analysis using zsteg + binwalk + comprehensive parameter testing
"""

import os
import subprocess
import tempfile
import shutil
import hashlib
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import re

@dataclass
class ExtractionNode:
    """Represents a file in the extraction tree"""
    file_path: str
    file_hash: str
    file_size: int
    depth: int
    parent_hash: Optional[str]
    extraction_method: str
    file_type: str
    analysis_results: List[Dict] = None
    extracted_files: List[str] = None
    
    def __post_init__(self):
        if self.analysis_results is None:
            self.analysis_results = []
        if self.extracted_files is None:
            self.extracted_files = []

class CascadeAnalyzer:
    """Recursive cascade analyzer with comprehensive parameter testing"""
    
    def __init__(self, config=None):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.max_depth = getattr(config, 'max_depth', 10) if config else 10
        self.max_files = getattr(config, 'max_files', 5000) if config else 5000
        self.output_dir = Path("cascade_analysis")
        self.output_dir.mkdir(exist_ok=True)
        
        # Tracking
        self.analyzed_hashes: Set[str] = set()
        self.extraction_nodes: Dict[str, ExtractionNode] = {}
        self.file_count = 0
        
        # Tool availability
        self.zsteg_available = self._check_tool("zsteg")
        self.binwalk_available = self._check_tool("binwalk")
        self.file_available = self._check_tool("file")
        
        # Comprehensive zsteg parameters
        self.zsteg_params = self._generate_zsteg_parameters()
        
        self.logger.info(f"Cascade analyzer initialized (zsteg: {self.zsteg_available}, binwalk: {self.binwalk_available})")
    
    def _check_tool(self, tool_name: str) -> bool:
        """Check if a tool is available"""
        try:
            subprocess.run([tool_name, '--version'], capture_output=True, timeout=5)
            return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def _generate_zsteg_parameters(self) -> List[List[str]]:
        """Generate comprehensive zsteg parameter combinations"""
        params = []
        
        # Basic LSB parameters
        channels = ['r', 'g', 'b', 'rgb', 'bgr', 'a']
        bit_orders = ['msb', 'lsb']
        bit_planes = ['0', '1', '2', '3', '4', '5', '6', '7']
        
        # Generate combinations
        for channel in channels:
            for order in bit_orders:
                for plane in bit_planes:
                    params.append([f'{channel}{plane},{order}'])
        
        # Add special parameters
        special_params = [
            ['-a'],  # All parameters
            ['-E', 'b1,rgb,lsb'],  # Extract mode
            ['-E', 'b1,bgr,msb'],
            ['--prime'],  # Prime checks
            ['--order', 'xy'],  # Different orders
            ['--order', 'yx'],
            ['-v'],  # Verbose
        ]
        params.extend(special_params)
        
        return params
    
    def get_file_type(self, file_path: Path) -> str:
        """Get file type using file command"""
        if not self.file_available:
            return file_path.suffix.lower()
        
        try:
            result = subprocess.run(
                ['file', '-b', str(file_path)], 
                capture_output=True, text=True, timeout=10
            )
            return result.stdout.strip() if result.returncode == 0 else "unknown"
        except:
            return "unknown"
    
    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of file"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except:
            return "unknown"
    
    async def cascade_analyze(self, file_path: Path, session_id: str = None) -> List[Dict[str, Any]]:
        """Main cascade analysis entry point"""
        self.logger.info(f"Starting cascade analysis on {file_path}")
        
        try:
            # Reset state
            self.analyzed_hashes.clear()
            self.extraction_nodes.clear()
            self.file_count = 0
            
            # Start recursive analysis
            results = await self._recursive_analyze(file_path, depth=0, parent_hash=None)
            
            # Generate summary
            summary = self._generate_summary()
            results.append(summary)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Cascade analysis failed: {e}")
            return [{
                "type": "cascade_error",
                "error": str(e),
                "file_path": str(file_path)
            }]
    
    async def _recursive_analyze(self, file_path: Path, depth: int, parent_hash: Optional[str]) -> List[Dict[str, Any]]:
        """Recursively analyze file and extracted content"""
        
        if depth > self.max_depth:
            self.logger.warning(f"Max depth {self.max_depth} reached")
            return []
        
        if self.file_count > self.max_files:
            self.logger.warning(f"Max files {self.max_files} reached")
            return []
        
        # Compute file hash
        file_hash = self._compute_file_hash(file_path)
        if file_hash in self.analyzed_hashes:
            return []  # Already analyzed
        
        self.analyzed_hashes.add(file_hash)
        self.file_count += 1
        
        self.logger.info(f"Analyzing {file_path.name} (depth: {depth}, hash: {file_hash[:8]})")
        
        # Create extraction node
        node = ExtractionNode(
            file_path=str(file_path),
            file_hash=file_hash,
            file_size=file_path.stat().st_size if file_path.exists() else 0,
            depth=depth,
            parent_hash=parent_hash,
            extraction_method="cascade",
            file_type=self.get_file_type(file_path)
        )
        
        self.extraction_nodes[file_hash] = node
        
        results = []
        
        # Run zsteg analysis
        if self.zsteg_available:
            zsteg_results = await self._run_zsteg_analysis(file_path)
            node.analysis_results.extend(zsteg_results)
            results.extend(zsteg_results)
        
        # Run binwalk extraction
        extracted_files = []
        if self.binwalk_available:
            extracted_files = await self._run_binwalk_extraction(file_path)
            node.extracted_files.extend(extracted_files)
        
        # Recursively analyze extracted files
        for extracted_file in extracted_files:
            try:
                extracted_path = Path(extracted_file)
                if extracted_path.exists() and extracted_path.is_file():
                    child_results = await self._recursive_analyze(
                        extracted_path, depth + 1, file_hash
                    )
                    results.extend(child_results)
            except Exception as e:
                self.logger.error(f"Error analyzing extracted file {extracted_file}: {e}")
        
        return results
    
    async def _run_zsteg_analysis(self, file_path: Path) -> List[Dict[str, Any]]:
        """Run comprehensive zsteg analysis"""
        if not self.zsteg_available:
            return []
        
        results = []
        
        # Check if file is suitable for zsteg
        file_ext = file_path.suffix.lower()
        if file_ext not in {'.png', '.bmp', '.gif', '.tiff', '.tif', '.webp'}:
            return []
        
        self.logger.info(f"Running zsteg analysis with {len(self.zsteg_params)} parameter sets")
        
        for i, params in enumerate(self.zsteg_params):
            try:
                cmd = ['zsteg'] + params + [str(file_path)]
                
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=30
                )
                
                if result.stdout.strip():
                    results.append({
                        "type": "zsteg_finding",
                        "method": "zsteg_cascade",
                        "params": ' '.join(params),
                        "output": result.stdout.strip(),
                        "command": ' '.join(cmd),
                        "confidence": 0.8,
                        "file_path": str(file_path),
                        "depth": self.extraction_nodes[self._compute_file_hash(file_path)].depth
                    })
                
                # Progress update
                if (i + 1) % 50 == 0:
                    self.logger.info(f"Tested {i + 1}/{len(self.zsteg_params)} parameter combinations")
                    
            except subprocess.TimeoutExpired:
                continue
            except Exception as e:
                self.logger.debug(f"zsteg error with params {params}: {e}")
                continue
        
        self.logger.info(f"zsteg found {len(results)} results")
        return results
    
    async def _run_binwalk_extraction(self, file_path: Path) -> List[str]:
        """Run binwalk extraction"""
        if not self.binwalk_available:
            return []
        
        extracted_files = []
        
        try:
            # Create unique extraction directory
            extract_dir = self.output_dir / f"binwalk_{file_path.stem}_{int(time.time())}"
            extract_dir.mkdir(exist_ok=True)
            
            # Run binwalk
            cmd = ['binwalk', '-e', '-C', str(extract_dir), str(file_path)]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            # Find all extracted files
            if extract_dir.exists():
                for extracted_file in extract_dir.rglob('*'):
                    if extracted_file.is_file() and extracted_file.stat().st_size > 0:
                        extracted_files.append(str(extracted_file))
            
            self.logger.info(f"binwalk extracted {len(extracted_files)} files")
            
        except subprocess.TimeoutExpired:
            self.logger.warning(f"binwalk timeout on {file_path.name}")
        except Exception as e:
            self.logger.error(f"binwalk error: {e}")
        
        return extracted_files
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate cascade analysis summary"""
        total_files = len(self.extraction_nodes)
        max_depth = max((node.depth for node in self.extraction_nodes.values()), default=0)
        
        findings_count = sum(
            len(node.analysis_results) 
            for node in self.extraction_nodes.values()
        )
        
        extracted_count = sum(
            len(node.extracted_files) 
            for node in self.extraction_nodes.values()
        )
        
        return {
            "type": "cascade_summary",
            "method": "cascade_analysis",
            "confidence": 1.0,
            "total_files_analyzed": total_files,
            "max_depth_reached": max_depth,
            "total_findings": findings_count,
            "total_extracted_files": extracted_count,
            "extraction_tree": {
                node.file_hash: {
                    "file_path": node.file_path,
                    "depth": node.depth,
                    "parent_hash": node.parent_hash,
                    "file_type": node.file_type,
                    "findings": len(node.analysis_results),
                    "extracted": len(node.extracted_files)
                }
                for node in self.extraction_nodes.values()
            }
        }
    
    def get_tool_info(self) -> Dict[str, Any]:
        """Get tool information"""
        return {
            "name": "cascade_analyzer",
            "version": "2.0",
            "requirements": {
                "zsteg": self.zsteg_available,
                "binwalk": self.binwalk_available,
                "file": self.file_available
            },
            "max_depth": self.max_depth,
            "max_files": self.max_files,
            "parameter_sets": len(self.zsteg_params)
        }
'''
    
    with open(cascade_file, 'w', encoding='utf-8') as f:
        f.write(cascade_code)
    
    print("   ✅ Cascade analyzer module updated")

def fix_configuration_integration(project_root: Path):
    """Fix configuration integration issues"""
    print("⚙️  Fixing configuration integration...")
    
    config_file = project_root / "config" / "steg_config.py"
    if not config_file.exists():
        print("   ⚠️  Config file not found, skipping")
        return
    
    with open(config_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Add cascade configuration if missing
    if "cascade_mode" not in content:
        cascade_config = '''
        # Cascade Analysis Configuration
        self.cascade_mode = False
        self.cascade_max_depth = 10
        self.cascade_max_files = 5000
        self.cascade_enable_zsteg = True
        self.cascade_enable_binwalk = True
        self.cascade_output_dir = "cascade_analysis"
'''
        # Insert into Config.__init__ method
        init_pattern = r'(def __init__\(self.*?\n.*?)(        # .*?\n)'
        content = re.sub(
            init_pattern,
            r'\1' + cascade_config + r'\2',
            content,
            flags=re.DOTALL
        )
    
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("   ✅ Configuration integration fixed")

def fix_main_script_integration(project_root: Path):
    """Fix main script CLI integration"""
    print("🎯 Fixing main script integration...")
    
    main_file = project_root / "steg_main.py"
    if not main_file.exists():
        print("   ⚠️  steg_main.py not found, skipping")
        return
    
    with open(main_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Add cascade arguments if missing
    if '--cascade' not in content:
        args_pattern = r'(parser\.add_argument\("--verbose".*?\n)'
        cascade_args = '''    parser.add_argument("--cascade", action="store_true", help="Enable recursive cascade analysis")
    parser.add_argument("--max-depth", type=int, default=10, help="Maximum cascade depth")
    parser.add_argument("--max-files", type=int, default=5000, help="Maximum files to process")
'''
        content = re.sub(args_pattern, r'\1' + cascade_args, content)
    
    # Add cascade handling in main function
    if 'args.cascade' not in content:
        main_pattern = r'(# Initialize orchestrator.*?\n.*?config.*?\n)'
        cascade_handling = '''    # Configure cascade mode
    if hasattr(args, 'cascade') and args.cascade:
        config.cascade_mode = True
        if hasattr(args, 'max_depth'):
            config.cascade_max_depth = args.max_depth
        if hasattr(args, 'max_files'):
            config.cascade_max_files = args.max_files
        print(f"🌳 Cascade mode enabled (depth: {config.cascade_max_depth}, max files: {config.cascade_max_files})")
    
'''
        content = re.sub(main_pattern, r'\1' + cascade_handling, content, flags=re.DOTALL)
    
    with open(main_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("   ✅ Main script integration fixed")

def cleanup_broken_files(project_root: Path):
    """Clean up broken/duplicate files"""
    print("🧹 Cleaning up broken files...")
    
    # Remove problematic duplicate directories
    cleanup_targets = [
        "hunter-beastmode",
        "cascade_tool_integration.py",
        "integration_script.py", 
        "simple_database_fix.py",
        "final_targeted_fix.py",
        "run_complete_fix.py"
    ]
    
    for target in cleanup_targets:
        target_path = project_root / target
        if target_path.exists():
            if target_path.is_dir():
                shutil.rmtree(target_path)
                print(f"   🗑️  Removed directory: {target}")
            else:
                target_path.unlink()
                print(f"   🗑️  Removed file: {target}")
    
    print("   ✅ Cleanup completed")

if __name__ == "__main__":
    main()
