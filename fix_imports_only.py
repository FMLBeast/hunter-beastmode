#!/usr/bin/env python3
"""
StegAnalyzer Import Fixer
Specifically fixes import issues without moving files
"""

import re
from pathlib import Path

def fix_imports_only():
    """Fix only import statements without reorganizing files"""
    
    print("🔧 Fixing StegAnalyzer imports...")
    
    project_root = Path(".")
    
    # Find all Python files
    python_files = []
    for py_file in project_root.rglob("*.py"):
        if "__pycache__" not in str(py_file) and "backup" not in str(py_file):
            python_files.append(py_file)
    
    print(f"Found {len(python_files)} Python files to check")
    
    # Define import fixes based on current structure
    import_fixes = [
        # Handle class names that might have changed
        (r'from core\.orchestrator import StegOrchestrator', 'from core.orchestrator import StegOrchestrator'),
        (r'from core\.database import DatabaseManager', 'from core.database import DatabaseManager'),
        
        # Handle tools imports
        (r'from tools\.classic_stego import ClassicStegoTools', 'from tools.classic_stego import ClassicStegoTools'),
        (r'from tools\.image_forensics import ImageForensicsTools', 'from tools.image_forensics import ImageForensicsTools'),
        (r'from tools\.audio_analysis import AudioAnalysisTools', 'from tools.audio_analysis import AudioAnalysisTools'),
        (r'from tools\.file_forensics import FileForensicsTools', 'from tools.file_forensics import FileForensicsTools'),
        (r'from tools\.crypto_analysis import CryptoAnalysisTools', 'from tools.crypto_analysis import CryptoAnalysisTools'),
        
        # Handle utils imports
        (r'from utils\.checkpoint import CheckpointManager', 'from utils.checkpoint import CheckpointManager'),
        
        # Handle cloud imports  
        (r'from cloud\.integrations import CloudIntegrations', 'from cloud.integrations import CloudIntegrations'),
        
        # Handle ai imports
        (r'from ai\.ml_detector import MLStegDetector', 'from ai.ml_detector import MLStegDetector'),
        (r'from ai\.llm_analyzer import LLMAnalyzer', 'from ai.llm_analyzer import LLMAnalyzer'),
        
        # Fix any imports that reference files in wrong locations
        (r'from utils\.ml_detector import', 'from ai.ml_detector import'),
        (r'from tools\.graph_tracker import', 'from core.graph_tracker import'),
        (r'from reports\.reporter import', 'from core.reporter import'),
        (r'from tools\.system_check import', 'from utils.system_check import'),
        
        # Standard relative import fixes
        (r'from \.orchestrator import', 'from .orchestrator import'),
        (r'from \.database import', 'from .database import'),
        (r'from \.classic_stego import', 'from .classic_stego import'),
        (r'from \.image_forensics import', 'from .image_forensics import'),
        (r'from \.audio_analysis import', 'from .audio_analysis import'),
        (r'from \.file_forensics import', 'from .file_forensics import'),
        (r'from \.crypto_analysis import', 'from .crypto_analysis import'),
        (r'from \.integrations import', 'from .integrations import'),
        (r'from \.checkpoint import', 'from .checkpoint import'),
        
        # Import statements (not from)
        (r'import core\.orchestrator', 'import core.orchestrator'),
        (r'import core\.database', 'import core.database'),
        (r'import tools\.classic_stego', 'import tools.classic_stego'),
        (r'import tools\.image_forensics', 'import tools.image_forensics'),
        (r'import tools\.audio_analysis', 'import tools.audio_analysis'),
        (r'import tools\.file_forensics', 'import tools.file_forensics'),
        (r'import tools\.crypto_analysis', 'import tools.crypto_analysis'),
        (r'import cloud\.integrations', 'import cloud.integrations'),
        (r'import utils\.checkpoint', 'import utils.checkpoint'),
    ]
    
    fixed_files = []
    
    for py_file in python_files:
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Apply all import fixes
            for old_pattern, new_pattern in import_fixes:
                content = re.sub(old_pattern, new_pattern, content)
            
            # Check if anything changed
            if content != original_content:
                print(f"✅ Fixed imports in: {py_file.relative_to(project_root)}")
                with open(py_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                fixed_files.append(py_file)
                
        except Exception as e:
            print(f"❌ Error fixing {py_file}: {e}")
    
    print(f"\n🎉 Fixed imports in {len(fixed_files)} files")
    
    # Now check for syntax errors
    print("\n🔍 Checking for syntax errors...")
    syntax_errors = []
    
    for py_file in python_files:
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                compile(f.read(), str(py_file), 'exec')
        except SyntaxError as e:
            syntax_errors.append((py_file, e))
        except Exception:
            # Other compilation errors, probably import errors
            pass
    
    if syntax_errors:
        print(f"❌ Found {len(syntax_errors)} syntax errors:")
        for py_file, error in syntax_errors:
            print(f"   {py_file.relative_to(project_root)}: {error}")
    else:
        print("✅ No syntax errors found")
    
    # Test basic imports
    print("\n🧪 Testing key imports...")
    test_imports = [
        "import core.orchestrator",
        "import core.database", 
        "import tools.classic_stego",
        "import ai.ml_detector",
        "import utils.checkpoint"
    ]
    
    import_success = 0
    for test_import in test_imports:
        try:
            exec(test_import)
            print(f"✅ {test_import}")
            import_success += 1
        except Exception as e:
            print(f"❌ {test_import}: {e}")
    
    print(f"\n📊 Import test results: {import_success}/{len(test_imports)} successful")
    
    if import_success == len(test_imports):
        print("🎉 All key imports working!")
    else:
        print("⚠️  Some imports still failing - may need file reorganization")

if __name__ == "__main__":
    fix_imports_only()
