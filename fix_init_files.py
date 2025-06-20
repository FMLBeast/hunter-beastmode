#!/usr/bin/env python3
"""
Fix __init__.py files to handle optional dependencies gracefully
"""

from pathlib import Path

def fix_init_files():
    """Fix __init__.py files to handle missing optional dependencies"""
    
    print("🔧 Fixing __init__.py files for optional dependencies...")
    
    project_root = Path(".")
    
    # Fixed __init__.py contents with proper error handling
    init_contents = {
        'ai/__init__.py': '''"""
AI and Machine Learning components for steganography detection
"""

# Core AI components
try:
    from .llm_analyzer import LLMAnalyzer
except ImportError as e:
    print(f"Warning: LLMAnalyzer not available: {e}")
    LLMAnalyzer = None

try:
    from .ml_detector import MLStegDetector
except ImportError as e:
    print(f"Warning: MLStegDetector not available: {e}")
    MLStegDetector = None

try:
    from .multimodal_classifier import MultimodalClassifier
except ImportError as e:
    print(f"Warning: MultimodalClassifier not available: {e}")
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
    from .integrations import CloudIntegrations
except ImportError as e:
    print(f"Warning: CloudIntegrations not available: {e}")
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
    print(f"Warning: StegOrchestrator not available: {e}")
    StegOrchestrator = None

try:
    from .dashboard import Dashboard
except ImportError as e:
    print(f"Warning: Dashboard not available: {e}")
    Dashboard = None

try:
    from .reporter import ReportGenerator
except ImportError as e:
    print(f"Warning: ReportGenerator not available: {e}")
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
    from .classic_stego import ClassicStegoTools
except ImportError as e:
    print(f"Warning: ClassicStegoTools not available: {e}")
    ClassicStegoTools = None

try:
    from .file_forensics import FileForensicsTools
except ImportError as e:
    print(f"Warning: FileForensicsTools not available: {e}")
    FileForensicsTools = None

try:
    from .crypto_analysis import CryptoAnalysisTools
except ImportError as e:
    print(f"Warning: CryptoAnalysisTools not available: {e}")
    CryptoAnalysisTools = None

try:
    from .metadata_carving import MetadataCarving
except ImportError as e:
    print(f"Warning: MetadataCarving not available: {e}")
    MetadataCarving = None

# Tools with heavy dependencies
try:
    from .image_forensics import ImageForensicsTools
except ImportError as e:
    print(f"Warning: ImageForensicsTools not available (missing image dependencies): {e}")
    ImageForensicsTools = None

try:
    from .audio_analysis import AudioAnalysisTools
except ImportError as e:
    print(f"Warning: AudioAnalysisTools not available (missing audio dependencies): {e}")
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
    print(f"Warning: GPUManager not available: {e}")
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
    
    # Also create a root __init__.py that's safe
    root_init = project_root / "__init__.py"
    root_content = '''"""
StegAnalyzer - Advanced Steganography Detection Framework
"""

__version__ = "1.0.0"
__author__ = "StegAnalyzer Team"

# Import core components safely
try:
    from .core import FileAnalyzer, DatabaseManager
    from .config import Config
except ImportError:
    # Fallback if some dependencies are missing
    pass

__all__ = ['FileAnalyzer', 'DatabaseManager', 'Config']
'''
    
    with open(root_init, 'w') as f:
        f.write(root_content)
    
    print("✅ Fixed all __init__.py files for optional dependencies")

def create_minimal_requirements():
    """Create a minimal requirements file for core functionality"""
    
    minimal_reqs = '''# Minimal requirements for StegAnalyzer core functionality
numpy>=1.21.0
Pillow>=9.0.0
pathlib2>=2.3.0
psutil>=5.8.0
python-magic>=0.4.24

# Optional requirements (install as needed):
# librosa>=0.8.1           # For audio analysis
# opencv-python>=4.5.0     # For advanced image analysis
# torch>=1.10.0            # For ML analysis
# tensorflow>=2.7.0        # Alternative ML framework
# scipy>=1.7.0             # For statistical analysis
# scikit-learn>=1.0.0      # For ML algorithms
# matplotlib>=3.5.0        # For plotting
# aiohttp>=3.8.0           # For web dashboard
# fastapi>=0.70.0          # For API server
# jinja2>=3.0.0            # For report generation
'''
    
    minimal_file = Path("requirements-minimal.txt")
    with open(minimal_file, 'w') as f:
        f.write(minimal_reqs)
    
    print(f"✅ Created {minimal_file} for minimal installation")

def test_imports():
    """Test imports after fixing"""
    
    print("\n🧪 Testing imports after fixes...")
    
    tests = [
        ("Core config", "from config import Config"),
        ("Core file analyzer", "from core import FileAnalyzer"),
        ("Core database", "from core import DatabaseManager"), 
        ("Core graph tracker", "from core import GraphTracker"),
        ("Utils checkpoint", "from utils import CheckpointManager"),
        ("Utils system check", "from utils import SystemChecker"),
    ]
    
    passed = 0
    for test_name, import_stmt in tests:
        try:
            exec(import_stmt)
            print(f"   ✅ {test_name}")
            passed += 1
        except Exception as e:
            print(f"   ❌ {test_name}: {e}")
    
    print(f"\n📊 Basic imports: {passed}/{len(tests)} working")
    
    # Test optional imports
    optional_tests = [
        ("Tools (may have missing deps)", "from tools import *"),
        ("AI (may have missing deps)", "from ai import *"),
        ("Cloud (may have missing deps)", "from cloud import *"),
    ]
    
    print("\n🔧 Testing optional components:")
    for test_name, import_stmt in optional_tests:
        try:
            exec(import_stmt)
            print(f"   ✅ {test_name}")
        except Exception as e:
            print(f"   ⚠️  {test_name}: {e}")

if __name__ == "__main__":
    fix_init_files()
    create_minimal_requirements()
    test_imports()
    
    print("\n🎉 Init files fixed!")
    print("\nNext steps:")
    print("1. Install minimal deps: pip install -r requirements-minimal.txt")
    print("2. Test core import: python -c 'from core import FileAnalyzer'")
    print("3. Install optional deps as needed for specific features")
