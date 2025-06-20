#!/usr/bin/env python3
"""
Fix __init__.py files to match actual file names that exist
"""

from pathlib import Path

def fix_init_files_actual_names():
    """Fix __init__.py files to import from actual file names"""
    
    print("🔧 Fixing __init__.py files to match actual file names...")
    
    project_root = Path(".")
    
    # Check what files actually exist
    def check_file_exists(path):
        return (project_root / path).exists()
    
    # Fixed __init__.py contents using actual file names
    init_contents = {
        'ai/__init__.py': '''"""
AI and Machine Learning components for steganography detection
"""

# Core AI components
try:
    from .llm_analyzer import LLMAnalyzer
except ImportError as e:
    LLMAnalyzer = None

try:
    from .ml_detector import MLStegDetector
except ImportError as e:
    MLStegDetector = None

try:
    from .multimodal_classifier import MultimodalClassifier
except ImportError as e:
    MultimodalClassifier = None

# Export only available components
__all__ = []
if LLMAnalyzer:
    __all__.append('LLMAnalyzer')
if MLStegDetector:
    __all__.append('MLStegDetector') 
if MultimodalClassifier:
    __all__.append('MultimodalClassifier')
''',

        'cloud/__init__.py': '''"""
Cloud integration services
"""

try:
    from .cloud_integrations import CloudIntegrations
except ImportError as e:
    CloudIntegrations = None

__all__ = ['CloudIntegrations'] if CloudIntegrations else []
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

# Essential core components (should always work)
from .file_analyzer import FileAnalyzer
from .steg_database import DatabaseManager
from .graph_tracker import GraphTracker

# Components that might have optional dependencies
try:
    from .steg_orchestrator import StegOrchestrator
except ImportError as e:
    StegOrchestrator = None

try:
    from .dashboard import Dashboard
except ImportError as e:
    Dashboard = None

try:
    from .reporter import ReportGenerator
except ImportError as e:
    ReportGenerator = None

# Export available components
__all__ = ['FileAnalyzer', 'DatabaseManager', 'GraphTracker']
if StegOrchestrator:
    __all__.append('StegOrchestrator')
if Dashboard:
    __all__.append('Dashboard')
if ReportGenerator:
    __all__.append('ReportGenerator')
''',

        'tools/__init__.py': '''"""
Analysis tools and integrations
"""

# Always available tools
try:
    from .classic_stego_tools import ClassicStegoTools
except ImportError as e:
    ClassicStegoTools = None

try:
    from .file_forensics_tools import FileForensicsTools
except ImportError as e:
    FileForensicsTools = None

try:
    from .crypto_analysis_tools import CryptoAnalysisTools
except ImportError as e:
    CryptoAnalysisTools = None

try:
    from .metadata_carving import MetadataCarving
except ImportError as e:
    MetadataCarving = None

# Tools with heavy dependencies
try:
    from .image_forensics_tools import ImageForensicsTools
except ImportError as e:
    ImageForensicsTools = None

try:
    from .audio_analysis_tools import AudioAnalysisTools
except ImportError as e:
    AudioAnalysisTools = None

# Export available tools
__all__ = []
if ClassicStegoTools:
    __all__.append('ClassicStegoTools')
if ImageForensicsTools:
    __all__.append('ImageForensicsTools')
if AudioAnalysisTools:
    __all__.append('AudioAnalysisTools')
if FileForensicsTools:
    __all__.append('FileForensicsTools')
if CryptoAnalysisTools:
    __all__.append('CryptoAnalysisTools')
if MetadataCarving:
    __all__.append('MetadataCarving')
''',

        'utils/__init__.py': '''"""
Utility functions and helpers
"""

from .checkpoint_manager import CheckpointManager
from .system_check import SystemChecker

try:
    from .gpu_manager import GPUManager
except ImportError as e:
    GPUManager = None

try:
    from .logger import setup_logging
except ImportError:
    # Fallback logging setup
    def setup_logging(level="INFO"):
        import logging
        logging.basicConfig(level=getattr(logging, level.upper()))
    
__all__ = ['CheckpointManager', 'SystemChecker', 'setup_logging']
if GPUManager:
    __all__.append('GPUManager')
'''
    }
    
    # Write the fixed __init__.py files
    for file_path, content in init_contents.items():
        init_file = project_root / file_path
        init_file.parent.mkdir(exist_ok=True)
        
        print(f"   Fixing: {file_path}")
        with open(init_file, 'w') as f:
            f.write(content)
    
    # Also create a minimal root __init__.py
    root_init = project_root / "__init__.py"
    root_content = '''"""
StegAnalyzer - Advanced Steganography Detection Framework
"""

__version__ = "1.0.0"
__author__ = "StegAnalyzer Team"

# Core functionality should always be available
try:
    from .config import Config
except ImportError:
    Config = None

__all__ = []
if Config:
    __all__.append('Config')
'''
    
    with open(root_init, 'w') as f:
        f.write(root_content)
    
    print("✅ Fixed all __init__.py files to match actual file names")

def test_core_imports():
    """Test the basic imports"""
    
    print("\n🧪 Testing core imports...")
    
    tests = [
        ("Config", "from config import Config"),
        ("FileAnalyzer", "from core import FileAnalyzer"),
        ("DatabaseManager", "from core import DatabaseManager"),
        ("GraphTracker", "from core import GraphTracker"),
        ("CheckpointManager", "from utils import CheckpointManager"),
        ("SystemChecker", "from utils import SystemChecker"),
    ]
    
    passed = 0
    for test_name, import_stmt in tests:
        try:
            exec(import_stmt)
            print(f"   ✅ {test_name}")
            passed += 1
        except Exception as e:
            print(f"   ❌ {test_name}: {e}")
    
    print(f"\n📊 Core imports: {passed}/{len(tests)} working")
    return passed == len(tests)

def create_test_script():
    """Create a test script to verify everything works"""
    
    test_script = '''#!/usr/bin/env python3
"""
Test StegAnalyzer core functionality
"""

def test_core_functionality():
    """Test basic StegAnalyzer functionality"""
    
    print("🧪 Testing StegAnalyzer Core Functionality")
    print("=" * 50)
    
    # Test 1: Config
    try:
        from config import Config
        config = Config()
        print("✅ Config loading works")
    except Exception as e:
        print(f"❌ Config failed: {e}")
        return False
    
    # Test 2: FileAnalyzer
    try:
        from core import FileAnalyzer
        analyzer = FileAnalyzer(config)
        print("✅ FileAnalyzer initialization works")
    except Exception as e:
        print(f"❌ FileAnalyzer failed: {e}")
        return False
    
    # Test 3: DatabaseManager
    try:
        from core import DatabaseManager
        db = DatabaseManager(config.database)
        print("✅ DatabaseManager initialization works")
    except Exception as e:
        print(f"❌ DatabaseManager failed: {e}")
        return False
    
    # Test 4: System check
    try:
        from utils import SystemChecker
        checker = SystemChecker()
        print("✅ SystemChecker initialization works")
    except Exception as e:
        print(f"❌ SystemChecker failed: {e}")
        return False
    
    print("\\n🎉 All core components working!")
    return True

if __name__ == "__main__":
    success = test_core_functionality()
    if success:
        print("\\n✅ Ready to run StegAnalyzer!")
        print("Next: python steg_main.py --check-system")
    else:
        print("\\n❌ Some components need attention")
'''
    
    test_file = Path("test_core.py")
    with open(test_file, 'w') as f:
        f.write(test_script)
    
    print(f"✅ Created {test_file} for testing core functionality")

if __name__ == "__main__":
    fix_init_files_actual_names()
    success = test_core_imports()
    create_test_script()
    
    print("\n🎉 Init files fixed for actual file names!")
    
    if success:
        print("\n✅ Core imports working! Try:")
        print("   python test_core.py")
        print("   python steg_main.py --check-system")
    else:
        print("\n⚠️  Some imports still failing. Check individual files.")
