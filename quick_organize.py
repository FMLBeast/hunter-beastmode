#!/usr/bin/env python3
"""
Quick StegAnalyzer Project Organizer
Run this script to quickly organize your project structure and fix imports
"""

import os
import shutil
import re
from pathlib import Path

def organize_steganalyzer():
    """Quick organization of StegAnalyzer project"""
    project_root = Path(".")
    
    print("🔧 Quick StegAnalyzer Project Organization")
    print("=" * 50)
    
    # 1. Remove duplicate files
    print("1. Removing duplicate files...")
    duplicates = [
        "reports/reporter.py",
        "tools/graph_tracker.py", 
        "utils/ml_detector.py",
        "steg_main.py.bak",
        "organize.py"
    ]
    
    for dup in duplicates:
        dup_path = project_root / dup
        if dup_path.exists():
            print(f"   Removing: {dup}")
            dup_path.unlink()
    
    # 2. Clean __pycache__
    print("2. Cleaning __pycache__ directories...")
    for pycache in project_root.rglob("__pycache__"):
        print(f"   Removing: {pycache}")
        shutil.rmtree(pycache)
    
    # 3. Rename files to standard naming
    print("3. Renaming files to standard naming...")
    renames = {
        "core/steg_orchestrator.py": "core/orchestrator.py",
        "core/steg_database.py": "core/database.py",
        "tools/classic_stego_tools.py": "tools/classic_stego.py",
        "tools/image_forensics_tools.py": "tools/image_forensics.py", 
        "tools/audio_analysis_tools.py": "tools/audio_analysis.py",
        "tools/file_forensics_tools.py": "tools/file_forensics.py",
        "tools/crypto_analysis_tools.py": "tools/crypto_analysis.py",
        "cloud/cloud_integrations.py": "cloud/integrations.py",
        "utils/checkpoint_manager.py": "utils/checkpoint.py",
        "config/default_config.json": "config/default.json"
    }
    
    for old_path, new_path in renames.items():
        old_file = project_root / old_path
        new_file = project_root / new_path
        
        if old_file.exists() and old_file.resolve() != new_file.resolve():
            print(f"   Renaming: {old_path} -> {new_path}")
            if new_file.exists():
                new_file.unlink()
            old_file.rename(new_file)
    
    # 4. Create __init__.py files
    print("4. Creating __init__.py files...")
    
    init_files = {
        "ai/__init__.py": 'from .llm_analyzer import LLMAnalyzer\nfrom .ml_detector import MLStegDetector\n',
        "cloud/__init__.py": 'from .integrations import CloudIntegrations\n',
        "config/__init__.py": 'from .steg_config import Config\n',
        "core/__init__.py": '''from .file_analyzer import FileAnalyzer
from .orchestrator import StegOrchestrator
from .database import DatabaseManager
from .dashboard import Dashboard
from .reporter import ReportGenerator
from .graph_tracker import GraphTracker
''',
        "tools/__init__.py": '''from .classic_stego import ClassicStegoTools
from .image_forensics import ImageForensicsTools
from .audio_analysis import AudioAnalysisTools
from .file_forensics import FileForensicsTools
from .crypto_analysis import CryptoAnalysisTools
from .metadata_carving import MetadataCarving
''',
        "utils/__init__.py": '''from .checkpoint import CheckpointManager
from .gpu_manager import GPUManager
from .system_check import SystemChecker
from .logger import setup_logging
'''
    }
    
    for init_path, content in init_files.items():
        init_file = project_root / init_path
        init_file.parent.mkdir(exist_ok=True)
        print(f"   Creating: {init_path}")
        with open(init_file, 'w') as f:
            f.write(content)
    
    # 5. Fix imports in all Python files
    print("5. Fixing imports...")
    
    import_fixes = [
        (r'from core\.orchestrator import', 'from core.orchestrator import'),
        (r'from core\.database import', 'from core.database import'),
        (r'from tools\.classic_stego import', 'from tools.classic_stego import'),
        (r'from tools\.image_forensics import', 'from tools.image_forensics import'),
        (r'from tools\.audio_analysis import', 'from tools.audio_analysis import'),
        (r'from tools\.file_forensics import', 'from tools.file_forensics import'),
        (r'from tools\.crypto_analysis import', 'from tools.crypto_analysis import'),
        (r'from cloud\.integrations import', 'from cloud.integrations import'),
        (r'from utils\.checkpoint import', 'from utils.checkpoint import'),
        (r'import core\.orchestrator', 'import core.orchestrator'),
        (r'import core\.database', 'import core.database'),
        (r'\.orchestrator', '.orchestrator'),
        (r'\.database', '.database'),
        (r'\.classic_stego', '.classic_stego'),
        (r'\.image_forensics', '.image_forensics'),
        (r'\.audio_analysis', '.audio_analysis'),
        (r'\.file_forensics', '.file_forensics'),
        (r'\.crypto_analysis', '.crypto_analysis'),
        (r'\.integrations', '.integrations'),
        (r'\.checkpoint', '.checkpoint'),
    ]
    
    python_files = []
    for py_file in project_root.rglob("*.py"):
        if "__pycache__" not in str(py_file):
            python_files.append(py_file)
    
    for py_file in python_files:
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            for old_pattern, new_pattern in import_fixes:
                content = re.sub(old_pattern, new_pattern, content)
            
            if content != original_content:
                print(f"   Fixed imports in: {py_file.relative_to(project_root)}")
                with open(py_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                    
        except Exception as e:
            print(f"   ⚠️ Could not fix imports in {py_file}: {e}")
    
    # 6. Create missing directories
    print("6. Creating missing directories...")
    dirs_to_create = ['static', 'templates', 'data', 'logs', 'models', 'reports', 'wordlists']
    for directory in dirs_to_create:
        dir_path = project_root / directory
        if not dir_path.exists():
            print(f"   Creating: {directory}/")
            dir_path.mkdir(exist_ok=True)
    
    # 7. Validate key files exist
    print("7. Validating structure...")
    required_files = [
        'steg_main.py',
        'core/orchestrator.py',
        'core/database.py', 
        'core/file_analyzer.py',
        'tools/classic_stego.py',
        'utils/checkpoint.py',
        'config/default.json'
    ]
    
    missing = []
    for req_file in required_files:
        if not (project_root / req_file).exists():
            missing.append(req_file)
    
    if missing:
        print("   ❌ Missing files:")
        for m in missing:
            print(f"      - {m}")
    else:
        print("   ✅ All key files present")
    
    print("\n" + "=" * 50)
    print("✅ Quick organization complete!")
    print("\nNext steps:")
    print("1. Test imports: python -c 'from core import StegOrchestrator'")
    print("2. Run system check: python steg_main.py --check-system")
    print("3. Install deps: pip install -r requirements.txt")

if __name__ == "__main__":
    organize_steganalyzer()
