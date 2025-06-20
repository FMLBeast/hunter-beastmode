#!/usr/bin/env python3
"""
Diagnose project structure and fix imports based on actual files
"""

from pathlib import Path
import ast

def diagnose_project():
    """Diagnose the current project structure"""
    
    print("🔍 Diagnosing project structure...")
    print("=" * 50)
    
    project_root = Path(".")
    
    # Check what Python files exist in each directory
    directories = ['ai', 'cloud', 'config', 'core', 'tools', 'utils']
    
    file_structure = {}
    
    for directory in directories:
        dir_path = project_root / directory
        if dir_path.exists():
            py_files = [f.name for f in dir_path.glob("*.py") if f.name != "__init__.py"]
            file_structure[directory] = py_files
            print(f"📁 {directory}/")
            for py_file in py_files:
                print(f"   📄 {py_file}")
        else:
            print(f"❌ {directory}/ - Directory missing")
            file_structure[directory] = []
    
    return file_structure

def check_file_imports(file_path):
    """Check what a file is trying to import"""
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(f"import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"from {module} import {alias.name}")
        
        return imports
    except Exception as e:
        print(f"   ⚠️ Could not parse {file_path}: {e}")
        return []

def diagnose_import_issues(file_structure):
    """Diagnose import issues in specific files"""
    
    print("\n🔍 Diagnosing import issues...")
    print("=" * 50)
    
    project_root = Path(".")
    
    # Check problematic files
    problematic_files = [
        "core/file_analyzer.py",
        "core/steg_database.py", 
        "core/graph_tracker.py",
        "utils/checkpoint_manager.py",
        "utils/system_check.py"
    ]
    
    for file_path in problematic_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"\n📄 {file_path}:")
            imports = check_file_imports(full_path)
            for imp in imports[:10]:  # Show first 10 imports
                print(f"   {imp}")
        else:
            print(f"\n❌ {file_path} - File missing")

def create_minimal_init_files(file_structure):
    """Create minimal __init__.py files that should work"""
    
    print("\n🔧 Creating minimal __init__.py files...")
    print("=" * 50)
    
    project_root = Path(".")
    
    # Create very basic __init__.py files that don't import everything
    init_contents = {
        'ai': '''# AI components - import individually as needed
# from .llm_analyzer import LLMAnalyzer
# from .ml_detector import MLStegDetector
# from .multimodal_classifier import MultimodalClassifier
''',
        'cloud': '''# Cloud components - import individually as needed  
# from .cloud_integrations import CloudIntegrations
''',
        'config': '''# Configuration
from .steg_config import Config
''',
        'core': '''# Core components - import individually to avoid circular imports
# Use: from core.file_analyzer import FileAnalyzer
# Use: from core.steg_database import DatabaseManager
# Use: from core.graph_tracker import GraphTracker
''',
        'tools': '''# Tools - import individually as needed
# from .classic_stego_tools import ClassicStegoTools
# from .image_forensics_tools import ImageForensicsTools  
# from .audio_analysis_tools import AudioAnalysisTools
# from .file_forensics_tools import FileForensicsTools
# from .crypto_analysis_tools import CryptoAnalysisTools
# from .metadata_carving import MetadataCarving
''',
        'utils': '''# Utilities - import individually as needed
# from .checkpoint_manager import CheckpointManager
# from .gpu_manager import GPUManager
# from .system_check import SystemChecker
# from .logger import setup_logging
'''
    }
    
    for directory, content in init_contents.items():
        if directory in file_structure and file_structure[directory]:
            init_file = project_root / directory / "__init__.py"
            print(f"   Creating minimal {directory}/__init__.py")
            with open(init_file, 'w') as f:
                f.write(content)

def test_individual_imports(file_structure):
    """Test importing individual files"""
    
    print("\n🧪 Testing individual file imports...")
    print("=" * 50)
    
    # Test individual files that should be safe to import
    safe_tests = [
        ("config.steg_config", "Config"),
    ]
    
    # Add tests based on what files exist
    if 'core/file_analyzer.py' in str(file_structure.get('core', [])):
        safe_tests.append(("core.file_analyzer", "FileAnalyzer"))
    
    if 'core/steg_database.py' in str(file_structure.get('core', [])):
        safe_tests.append(("core.steg_database", "DatabaseManager"))
    
    if 'utils/system_check.py' in str(file_structure.get('utils', [])):
        safe_tests.append(("utils.system_check", "SystemChecker"))
    
    working_imports = []
    
    for module_name, class_name in safe_tests:
        try:
            exec(f"from {module_name} import {class_name}")
            print(f"   ✅ {module_name}.{class_name}")
            working_imports.append((module_name, class_name))
        except Exception as e:
            print(f"   ❌ {module_name}.{class_name}: {e}")
    
    return working_imports

def create_working_init_files(working_imports):
    """Create __init__.py files based on what actually works"""
    
    print("\n🔧 Creating working __init__.py files...")
    print("=" * 50)
    
    project_root = Path(".")
    
    # Create config __init__.py (this should work)
    config_init = project_root / "config" / "__init__.py"
    with open(config_init, 'w') as f:
        f.write('from .steg_config import Config\n__all__ = ["Config"]')
    print("   ✅ config/__init__.py")
    
    # Create core __init__.py with working imports only
    core_imports = [imp for imp in working_imports if imp[0].startswith('core')]
    if core_imports:
        core_init = project_root / "core" / "__init__.py"
        with open(core_init, 'w') as f:
            f.write('# Core components\n')
            for module, class_name in core_imports:
                module_file = module.split('.')[-1]
                f.write(f'from .{module_file} import {class_name}\n')
            
            all_classes = [imp[1] for imp in core_imports]
            f.write(f'\n__all__ = {all_classes}')
        print("   ✅ core/__init__.py")
    
    # Create utils __init__.py with working imports only  
    utils_imports = [imp for imp in working_imports if imp[0].startswith('utils')]
    if utils_imports:
        utils_init = project_root / "utils" / "__init__.py"
        with open(utils_init, 'w') as f:
            f.write('# Utility components\n')
            for module, class_name in utils_imports:
                module_file = module.split('.')[-1]
                f.write(f'from .{module_file} import {class_name}\n')
                
            all_classes = [imp[1] for imp in utils_imports]
            f.write(f'\n__all__ = {all_classes}')
        print("   ✅ utils/__init__.py")

def final_test():
    """Final test of imports"""
    
    print("\n🧪 Final import test...")
    print("=" * 50)
    
    tests = [
        "from config import Config",
        "from core.file_analyzer import FileAnalyzer", 
        "from core.steg_database import DatabaseManager",
        "from utils.system_check import SystemChecker"
    ]
    
    working = 0
    for test in tests:
        try:
            exec(test)
            print(f"   ✅ {test}")
            working += 1
        except Exception as e:
            print(f"   ❌ {test}: {e}")
    
    print(f"\n📊 Final results: {working}/{len(tests)} imports working")
    
    if working >= 2:
        print("\n🎉 Enough imports working to proceed!")
        print("Next steps:")
        print("   python3 -c 'from config import Config'")
        print("   python3 steg_main.py --help")
    else:
        print("\n⚠️  Still having import issues. May need manual fixes.")

def main():
    """Main diagnosis and fix process"""
    
    print("🩺 StegAnalyzer Project Diagnosis & Fix")
    print("=" * 60)
    
    # Step 1: Diagnose current structure
    file_structure = diagnose_project()
    
    # Step 2: Check import issues in key files
    diagnose_import_issues(file_structure)
    
    # Step 3: Create minimal __init__.py files
    create_minimal_init_files(file_structure)
    
    # Step 4: Test individual imports
    working_imports = test_individual_imports(file_structure)
    
    # Step 5: Create working __init__.py files
    create_working_init_files(working_imports)
    
    # Step 6: Final test
    final_test()

if __name__ == "__main__":
    main()
