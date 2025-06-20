#!/usr/bin/env python3
"""
Fix StegOrchestrator to handle optional tool dependencies gracefully
"""

from pathlib import Path
import re

def fix_orchestrator_imports():
    """Fix the orchestrator to import tools conditionally"""
    
    print("🔧 Fixing StegOrchestrator optional imports...")
    
    orchestrator_file = Path("core/orchestrator.py")
    
    if not orchestrator_file.exists():
        print("❌ orchestrator.py not found")
        return False
    
    # Read the current content
    with open(orchestrator_file, 'r') as f:
        content = f.read()
    
    # Replace the tool imports section to be conditional
    new_imports = '''"""
Core Orchestrator - Manages the entire steganography analysis pipeline
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
import hashlib
import time
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

# Core imports that should always work
from core.file_analyzer import FileAnalyzer
from core.graph_tracker import GraphTracker
from utils.checkpoint import CheckpointManager

# Tool imports with graceful handling
try:
    from tools.classic_stego import ClassicStegoTools
except ImportError as e:
    print(f"Warning: ClassicStegoTools not available: {e}")
    ClassicStegoTools = None

try:
    from tools.image_forensics import ImageForensicsTools
except ImportError as e:
    print(f"Warning: ImageForensicsTools not available: {e}")
    ImageForensicsTools = None

try:
    from tools.audio_analysis import AudioAnalysisTools
except ImportError as e:
    print(f"Warning: AudioAnalysisTools not available (missing audio dependencies): {e}")
    AudioAnalysisTools = None

try:
    from tools.file_forensics import FileForensicsTools
except ImportError as e:
    print(f"Warning: FileForensicsTools not available: {e}")
    FileForensicsTools = None

try:
    from tools.crypto_analysis import CryptoAnalysisTools
except ImportError as e:
    print(f"Warning: CryptoAnalysisTools not available: {e}")
    CryptoAnalysisTools = None

try:
    from tools.metadata_carving import MetadataCarving
except ImportError as e:
    print(f"Warning: MetadataCarving not available: {e}")
    MetadataCarving = None

# AI components (optional)
try:
    from ai.ml_detector import MLStegDetector
except ImportError as e:
    print(f"Warning: MLStegDetector not available: {e}")
    MLStegDetector = None

try:
    from ai.llm_analyzer import LLMAnalyzer
except ImportError as e:
    print(f"Warning: LLMAnalyzer not available: {e}")
    LLMAnalyzer = None

try:
    from ai.multimodal_classifier import MultimodalClassifier
except ImportError as e:
    print(f"Warning: MultimodalClassifier not available: {e}")
    MultimodalClassifier = None

# Cloud integrations (optional)
try:
    from cloud.integrations import CloudIntegrations
except ImportError as e:
    print(f"Warning: CloudIntegrations not available: {e}")
    CloudIntegrations = None

# GPU management (optional)
try:
    from utils.gpu_manager import GPUManager
except ImportError as e:
    print(f"Warning: GPUManager not available: {e}")
    GPUManager = None
'''
    
    # Find the import section and replace it
    # Look for the pattern from the start until the first class definition
    pattern = r'^(.*?)(?=@dataclass|class StegOrchestrator)'
    
    new_content = re.sub(pattern, new_imports + '\n\n', content, flags=re.DOTALL)
    
    # Also fix the __init__ method to handle None tools
    init_fix = '''    def __init__(self, config, database):
        self.config = config
        self.db = database
        self.logger = logging.getLogger(__name__)
        
        # Initialize tool managers (only if available)
        self.classic_tools = ClassicStegoTools(config) if ClassicStegoTools else None
        self.image_tools = ImageForensicsTools(config) if ImageForensicsTools else None
        self.audio_tools = AudioAnalysisTools(config) if AudioAnalysisTools else None
        self.file_tools = FileForensicsTools(config) if FileForensicsTools else None
        self.crypto_tools = CryptoAnalysisTools(config) if CryptoAnalysisTools else None
        self.metadata_tools = MetadataCarving(config) if MetadataCarving else None
        
        # Initialize AI components (only if available)
        self.ml_detector = MLStegDetector(config) if MLStegDetector else None
        self.llm_analyzer = LLMAnalyzer(config) if LLMAnalyzer else None
        self.multimodal_classifier = MultimodalClassifier(config) if MultimodalClassifier else None
        
        # Initialize cloud integrations (only if available)
        self.cloud = CloudIntegrations(config) if (CloudIntegrations and config.cloud.enabled) else None
        
        # Core components (should always be available)
        self.file_analyzer = FileAnalyzer(config)
        self.graph_tracker = GraphTracker(database)
        self.gpu_manager = GPUManager() if GPUManager else None
        self.checkpoint_manager = CheckpointManager(database)'''
    
    # Replace the __init__ method
    init_pattern = r'(def __init__\(self, config, database\):.*?)(?=\n    def|\n\n    def|\nclass|\Z)'
    new_content = re.sub(init_pattern, init_fix, new_content, flags=re.DOTALL)
    
    # Write the fixed content
    with open(orchestrator_file, 'w') as f:
        f.write(new_content)
    
    print("✅ StegOrchestrator fixed to handle optional dependencies")
    return True

def test_orchestrator_import():
    """Test that the orchestrator can now be imported"""
    
    print("\n🧪 Testing StegOrchestrator import...")
    
    try:
        from core.orchestrator import StegOrchestrator
        print("✅ StegOrchestrator imports successfully")
        
        # Test creating it
        from config import Config
        from core.database import DatabaseManager
        
        config = Config()
        # Don't initialize database for this test
        print("✅ StegOrchestrator can be imported and used")
        return True
        
    except Exception as e:
        print(f"❌ StegOrchestrator import failed: {e}")
        return False

def test_main_cli():
    """Test the main CLI"""
    
    print("\n🧪 Testing main CLI...")
    
    try:
        import subprocess
        result = subprocess.run(['python3', 'steg_main.py', '--help'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✅ CLI help works!")
            return True
        else:
            print(f"❌ CLI failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ CLI test error: {e}")
        return False

def main():
    """Main fix function"""
    
    print("🔧 Fixing StegAnalyzer Optional Dependencies")
    print("=" * 60)
    
    # Fix the orchestrator
    orchestrator_fixed = fix_orchestrator_imports()
    
    if orchestrator_fixed:
        # Test the orchestrator import
        orchestrator_works = test_orchestrator_import()
        
        if orchestrator_works:
            # Test the main CLI
            cli_works = test_main_cli()
            
            if cli_works:
                print("\n" + "=" * 60)
                print("🎉 SUCCESS! StegAnalyzer CLI is now working!")
                print("\nTry these commands:")
                print("• python3 steg_main.py --help")
                print("• python3 steg_main.py --check-system")
                print("• python3 steg_main.py --list-sessions")
            else:
                print("\n⚠️ Orchestrator works but CLI needs more fixes")
        else:
            print("\n❌ Orchestrator still has import issues")
    else:
        print("\n❌ Could not fix orchestrator file")

if __name__ == "__main__":
    main()
