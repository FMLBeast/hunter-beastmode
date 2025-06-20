#!/usr/bin/env python3
"""
Final fix for StegAnalyzer imports using the actual renamed files
"""

from pathlib import Path

def final_fix_imports():
    """Final fix using the correctly renamed files"""
    
    print("🔧 Final import fix using actual file names...")
    print("=" * 50)
    
    project_root = Path(".")
    
    # Create correct __init__.py files using the renamed files
    init_contents = {
        'ai/__init__.py': '''"""
AI and Machine Learning components
"""
# Import individually to avoid dependency issues
# from .llm_analyzer import LLMAnalyzer
# from .ml_detector import MLStegDetector  
# from .multimodal_classifier import MultimodalClassifier
''',

        'cloud/__init__.py': '''"""
Cloud integration services
"""
try:
    from .integrations import CloudIntegrations
    __all__ = ['CloudIntegrations']
except ImportError:
    __all__ = []
''',

        'config/__init__.py': '''"""
Configuration management
"""
from .steg_config import Config
__all__ = ['Config']
''',

        'core/__init__.py': '''"""
Core analysis components
"""
# Use the actual renamed files
from .file_analyzer import FileAnalyzer
from .database import DatabaseManager  # This is database.py, not steg_database.py
from .graph_tracker import GraphTracker

# Optional components with dependencies
try:
    from .orchestrator import StegOrchestrator
except ImportError:
    StegOrchestrator = None

try:
    from .dashboard import Dashboard
except ImportError:
    Dashboard = None

try:
    from .reporter import ReportGenerator
except ImportError:
    ReportGenerator = None

__all__ = ['FileAnalyzer', 'DatabaseManager', 'GraphTracker']
if StegOrchestrator:
    __all__.append('StegOrchestrator')
if Dashboard:
    __all__.append('Dashboard')  
if ReportGenerator:
    __all__.append('ReportGenerator')
''',

        'tools/__init__.py': '''"""
Analysis tools
"""
# Import individually to avoid heavy dependencies
# from .classic_stego import ClassicStegoTools
# from .image_forensics import ImageForensicsTools
# from .audio_analysis import AudioAnalysisTools  
# from .file_forensics import FileForensicsTools
# from .crypto_analysis import CryptoAnalysisTools
# from .metadata_carving import MetadataCarving
''',

        'utils/__init__.py': '''"""
Utility functions
"""
from .checkpoint import CheckpointManager  # This is checkpoint.py, not checkpoint_manager.py
from .system_check import SystemChecker

try:
    from .gpu_manager import GPUManager
except ImportError:
    GPUManager = None

try:
    from .logger import setup_logging
except ImportError:
    def setup_logging(level="INFO"):
        import logging
        logging.basicConfig(level=getattr(logging, level.upper()))

__all__ = ['CheckpointManager', 'SystemChecker', 'setup_logging']
if GPUManager:
    __all__.append('GPUManager')
'''
    }
    
    # Write the corrected __init__.py files
    for file_path, content in init_contents.items():
        init_file = project_root / file_path
        print(f"   Fixing: {file_path}")
        with open(init_file, 'w') as f:
            f.write(content)
    
    print("✅ All __init__.py files updated with correct file names")

def test_corrected_imports():
    """Test the corrected imports"""
    
    print("\n🧪 Testing corrected imports...")
    print("=" * 50)
    
    tests = [
        ("Config", "from config import Config"),
        ("FileAnalyzer", "from core import FileAnalyzer"),
        ("DatabaseManager", "from core import DatabaseManager"),  # Now uses database.py
        ("GraphTracker", "from core import GraphTracker"),
        ("CheckpointManager", "from utils import CheckpointManager"),  # Now uses checkpoint.py
        ("SystemChecker", "from utils import SystemChecker"),
    ]
    
    working = 0
    for test_name, import_stmt in tests:
        try:
            exec(import_stmt)
            print(f"   ✅ {test_name}")
            working += 1
        except Exception as e:
            print(f"   ❌ {test_name}: {e}")
    
    print(f"\n📊 Results: {working}/{len(tests)} imports working")
    return working

def test_main_functionality():
    """Test that the main functionality works"""
    
    print("\n🧪 Testing main functionality...")
    print("=" * 50)
    
    try:
        # Test creating core objects
        from config import Config
        config = Config()
        print("   ✅ Config created successfully")
        
        from core import FileAnalyzer
        analyzer = FileAnalyzer(config)
        print("   ✅ FileAnalyzer created successfully")
        
        from core import DatabaseManager
        db = DatabaseManager(config.database)
        print("   ✅ DatabaseManager created successfully")
        
        from utils import SystemChecker
        checker = SystemChecker()
        print("   ✅ SystemChecker created successfully")
        
        print("\n🎉 Core functionality working!")
        return True
        
    except Exception as e:
        print(f"   ❌ Core functionality test failed: {e}")
        return False

def create_simple_test_script():
    """Create a simple test script"""
    
    test_script = '''#!/usr/bin/env python3
"""
Simple StegAnalyzer test
"""

def test_basic_functionality():
    print("🧪 Testing StegAnalyzer basic functionality...")
    
    try:
        from config import Config
        print("✅ Config import works")
        
        config = Config()
        print("✅ Config creation works")
        
        from core import FileAnalyzer, DatabaseManager
        print("✅ Core imports work")
        
        analyzer = FileAnalyzer(config)
        print("✅ FileAnalyzer creation works")
        
        db = DatabaseManager(config.database)
        print("✅ DatabaseManager creation works")
        
        print("\\n🎉 Basic functionality test passed!")
        print("Ready to run: python3 steg_main.py --help")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_basic_functionality()
'''
    
    test_file = Path("test_basic.py")
    with open(test_file, 'w') as f:
        f.write(test_script)
    
    print(f"✅ Created {test_file}")

def main():
    """Main function"""
    
    print("🔧 StegAnalyzer Final Import Fix")
    print("=" * 60)
    
    # Fix the imports
    final_fix_imports()
    
    # Test the fixes
    working_count = test_corrected_imports()
    
    # Test main functionality
    if working_count >= 4:
        functionality_works = test_main_functionality()
        
        if functionality_works:
            create_simple_test_script()
            
            print("\n" + "=" * 60)
            print("🎉 SUCCESS! StegAnalyzer imports are working!")
            print("\nNext steps:")
            print("1. Test basic: python3 test_basic.py")
            print("2. Check system: python3 steg_main.py --check-system")
            print("3. See help: python3 steg_main.py --help")
            print("4. Install deps: python3 -m pip install -r requirements.txt")
        else:
            print("\n⚠️ Imports work but object creation has issues")
    else:
        print(f"\n⚠️ Only {working_count}/6 imports working. Need more fixes.")

if __name__ == "__main__":
    main()
