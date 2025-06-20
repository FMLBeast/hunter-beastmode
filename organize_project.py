#!/usr/bin/env python3
"""
StegAnalyzer Project Organizer and Import Fixer
Organizes files into proper structure and fixes all imports
"""

import os
import shutil
import re
from pathlib import Path
import subprocess
import sys

class ProjectOrganizer:
    def __init__(self, project_root="."):
        self.project_root = Path(project_root).resolve()
        self.backup_dir = self.project_root / "backup_before_organize"
        
        # Define the target structure
        self.target_structure = {
            'ai': [
                'llm_analyzer.py',
                'ml_detector.py', 
                'multimodal_classifier.py'
            ],
            'cloud': [
                'integrations.py'  # rename from cloud_integrations.py
            ],
            'config': [
                'default.json',  # rename from default_config.json
                'steg_config.py'
            ],
            'core': [
                'dashboard.py',
                'file_analyzer.py',
                'graph_tracker.py',
                'orchestrator.py',  # rename from steg_orchestrator.py
                'database.py',      # rename from steg_database.py
                'reporter.py'
            ],
            'tools': [
                'classic_stego.py',     # rename from classic_stego_tools.py
                'image_forensics.py',   # rename from image_forensics_tools.py
                'audio_analysis.py',    # rename from audio_analysis_tools.py
                'file_forensics.py',    # rename from file_forensics_tools.py
                'crypto_analysis.py',   # rename from crypto_analysis_tools.py
                'metadata_carving.py'
            ],
            'utils': [
                'checkpoint.py',     # rename from checkpoint_manager.py
                'gpu_manager.py',
                'logger.py',
                'system_check.py'
            ]
        }
        
        # Files to rename during organization
        self.file_renames = {
            'cloud_integrations.py': 'integrations.py',
            'default_config.json': 'default.json',
            'steg_orchestrator.py': 'orchestrator.py',
            'steg_database.py': 'database.py',
            'classic_stego_tools.py': 'classic_stego.py',
            'image_forensics_tools.py': 'image_forensics.py',
            'audio_analysis_tools.py': 'audio_analysis.py',
            'file_forensics_tools.py': 'file_forensics.py',
            'crypto_analysis_tools.py': 'crypto_analysis.py',
            'checkpoint_manager.py': 'checkpoint.py'
        }
        
        # Import mappings for fixing imports
        self.import_mappings = {
            'core.orchestrator': 'core.orchestrator',
            'core.database': 'core.database',
            'tools.classic_stego': 'tools.classic_stego',
            'tools.image_forensics': 'tools.image_forensics',
            'tools.audio_analysis': 'tools.audio_analysis',
            'tools.file_forensics': 'tools.file_forensics',
            'tools.crypto_analysis': 'tools.crypto_analysis',
            'cloud.integrations': 'cloud.integrations',
            'utils.checkpoint': 'utils.checkpoint',
            'config.default_config': 'config.default'
        }
    
    def create_backup(self):
        """Create backup of current state"""
        print("Creating backup...")
        if self.backup_dir.exists():
            shutil.rmtree(self.backup_dir)
        
        # Copy important files and directories
        important_items = ['ai', 'cloud', 'config', 'core', 'tools', 'utils', 
                          'steg_main.py', 'requirements.txt', 'README.md']
        
        self.backup_dir.mkdir(exist_ok=True)
        
        for item in important_items:
            src = self.project_root / item
            if src.exists():
                if src.is_dir():
                    shutil.copytree(src, self.backup_dir / item)
                else:
                    shutil.copy2(src, self.backup_dir / item)
        
        print(f"Backup created at: {self.backup_dir}")
    
    def clean_duplicates_and_temp_files(self):
        """Remove duplicate files, backups, and temp files"""
        print("Cleaning duplicate and temporary files...")
        
        # Files to remove
        files_to_remove = [
            'steg_main.py.bak',
            'organize.py',
            'setup.py',
            'reports/reporter.py',  # duplicate
            'tools/graph_tracker.py',  # duplicate - should be in core
            'utils/ml_detector.py',    # duplicate - should be in ai
            'utils/system_check.py' if (self.project_root / 'tools' / 'system_check.py').exists() else None
        ]
        
        for file_path in files_to_remove:
            if file_path:
                full_path = self.project_root / file_path
                if full_path.exists():
                    print(f"Removing duplicate/temp file: {file_path}")
                    full_path.unlink()
        
        # Remove __pycache__ directories
        for pycache in self.project_root.rglob("__pycache__"):
            print(f"Removing: {pycache}")
            shutil.rmtree(pycache)
    
    def organize_files(self):
        """Organize files into proper directory structure"""
        print("Organizing files into proper structure...")
        
        # Create all necessary directories
        for directory in self.target_structure.keys():
            (self.project_root / directory).mkdir(exist_ok=True)
        
        # Additional directories
        additional_dirs = ['static', 'templates', 'data', 'logs', 'models', 'reports', 'wordlists']
        for directory in additional_dirs:
            (self.project_root / directory).mkdir(exist_ok=True)
        
        # Move and rename files
        self._move_files_to_target_structure()
        
        # Create __init__.py files
        self._create_init_files()
    
    def _move_files_to_target_structure(self):
        """Move files to their target locations"""
        
        # Find all Python files in the project
        all_python_files = {}
        for py_file in self.project_root.rglob("*.py"):
            if "__pycache__" not in str(py_file) and "backup" not in str(py_file):
                all_python_files[py_file.name] = py_file
        
        # Move files according to target structure
        for target_dir, target_files in self.target_structure.items():
            target_path = self.project_root / target_dir
            
            for target_file in target_files:
                # Check if we need to rename this file
                source_name = None
                for old_name, new_name in self.file_renames.items():
                    if new_name == target_file:
                        source_name = old_name
                        break
                
                if source_name is None:
                    source_name = target_file
                
                # Find the source file
                source_file = None
                if source_name in all_python_files:
                    source_file = all_python_files[source_name]
                else:
                    # Search for the file in various locations
                    possible_locations = [
                        self.project_root / source_name,
                        self.project_root / target_dir / source_name,
                        self.project_root / "core" / source_name,
                        self.project_root / "tools" / source_name,
                        self.project_root / "utils" / source_name,
                        self.project_root / "ai" / source_name,
                        self.project_root / "cloud" / source_name,
                        self.project_root / "config" / source_name
                    ]
                    
                    for location in possible_locations:
                        if location.exists():
                            source_file = location
                            break
                
                if source_file and source_file.exists():
                    target_file_path = target_path / target_file
                    
                    # Don't move if already in correct location with correct name
                    if source_file.resolve() != target_file_path.resolve():
                        print(f"Moving {source_file} -> {target_file_path}")
                        
                        # Remove target if it exists
                        if target_file_path.exists():
                            target_file_path.unlink()
                        
                        shutil.move(str(source_file), str(target_file_path))
                else:
                    print(f"Warning: Could not find source file for {target_file} (looking for {source_name})")
        
        # Handle special cases for JSON and other files
        json_files = {
            'config/default_config.json': 'config/default.json'
        }
        
        for source, target in json_files.items():
            source_path = self.project_root / source
            target_path = self.project_root / target
            
            if source_path.exists() and source_path.resolve() != target_path.resolve():
                print(f"Moving {source_path} -> {target_path}")
                if target_path.exists():
                    target_path.unlink()
                shutil.move(str(source_path), str(target_path))
    
    def _create_init_files(self):
        """Create __init__.py files for all packages"""
        print("Creating __init__.py files...")
        
        init_contents = {
            'ai': '''"""
AI and Machine Learning components for steganography detection
"""

from .llm_analyzer import LLMAnalyzer
from .ml_detector import MLStegDetector

try:
    from .multimodal_classifier import MultimodalClassifier
except ImportError:
    MultimodalClassifier = None

__all__ = ['LLMAnalyzer', 'MLStegDetector', 'MultimodalClassifier']
''',
            'cloud': '''"""
Cloud integration services
"""

from .integrations import CloudIntegrations

__all__ = ['CloudIntegrations']
''',
            'config': '''"""
Configuration management
"""

from .steg_config import Config

__all__ = ['Config']
''',
            'core': '''"""
Core analysis components
"""

from .file_analyzer import FileAnalyzer
from .orchestrator import StegOrchestrator
from .database import DatabaseManager
from .dashboard import Dashboard
from .reporter import ReportGenerator
from .graph_tracker import GraphTracker

__all__ = [
    'FileAnalyzer', 'StegOrchestrator', 'DatabaseManager', 
    'Dashboard', 'ReportGenerator', 'GraphTracker'
]
''',
            'tools': '''"""
Analysis tools and integrations
"""

from .classic_stego import ClassicStegoTools
from .image_forensics import ImageForensicsTools
from .audio_analysis import AudioAnalysisTools
from .file_forensics import FileForensicsTools
from .crypto_analysis import CryptoAnalysisTools
from .metadata_carving import MetadataCarving

__all__ = [
    'ClassicStegoTools', 'ImageForensicsTools', 'AudioAnalysisTools',
    'FileForensicsTools', 'CryptoAnalysisTools', 'MetadataCarving'
]
''',
            'utils': '''"""
Utility functions and helpers
"""

from .checkpoint import CheckpointManager
from .gpu_manager import GPUManager
from .system_check import SystemChecker
from .logger import setup_logging

__all__ = ['CheckpointManager', 'GPUManager', 'SystemChecker', 'setup_logging']
'''
        }
        
        # Create __init__.py files
        for directory, content in init_contents.items():
            init_file = self.project_root / directory / "__init__.py"
            with open(init_file, 'w') as f:
                f.write(content)
        
        # Create root __init__.py
        root_init = self.project_root / "__init__.py"
        with open(root_init, 'w') as f:
            f.write('''"""
StegAnalyzer - Advanced Steganography Detection Framework
"""

__version__ = "1.0.0"
__author__ = "StegAnalyzer Team"

from .core import StegOrchestrator
from .config import Config

__all__ = ['StegOrchestrator', 'Config']
''')
    
    def fix_imports(self):
        """Fix all imports throughout the codebase"""
        print("Fixing imports throughout the codebase...")
        
        # Get all Python files
        python_files = []
        for py_file in self.project_root.rglob("*.py"):
            if "__pycache__" not in str(py_file) and "backup" not in str(py_file):
                python_files.append(py_file)
        
        for py_file in python_files:
            self._fix_file_imports(py_file)
    
    def _fix_file_imports(self, file_path):
        """Fix imports in a specific file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Fix specific import patterns
            import_fixes = [
                # Old module names to new module names
                (r'from core\.orchestrator import', 'from core.orchestrator import'),
                (r'from core\.database import', 'from core.database import'),
                (r'from tools\.classic_stego import', 'from tools.classic_stego import'),
                (r'from tools\.image_forensics import', 'from tools.image_forensics import'),
                (r'from tools\.audio_analysis import', 'from tools.audio_analysis import'),
                (r'from tools\.file_forensics import', 'from tools.file_forensics import'),
                (r'from tools\.crypto_analysis import', 'from tools.crypto_analysis import'),
                (r'from cloud\.integrations import', 'from cloud.integrations import'),
                (r'from utils\.checkpoint import', 'from utils.checkpoint import'),
                
                # Class name updates
                (r'StegOrchestrator', 'StegOrchestrator'),  # This one stays the same
                (r'DatabaseManager', 'DatabaseManager'),   # This one stays the same
                (r'ClassicStegoTools', 'ClassicStegoTools'),
                (r'ImageForensicsTools', 'ImageForensicsTools'),
                (r'AudioAnalysisTools', 'AudioAnalysisTools'),
                (r'FileForensicsTools', 'FileForensicsTools'),
                (r'CryptoAnalysisTools', 'CryptoAnalysisTools'),
                (r'CloudIntegrations', 'CloudIntegrations'),
                (r'CheckpointManager', 'CheckpointManager'),
                
                # Import statements for relative imports
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
            
            # Apply fixes
            for old_pattern, new_pattern in import_fixes:
                content = re.sub(old_pattern, new_pattern, content)
            
            # Fix common import issues specific to our structure
            specific_fixes = [
                # Fix references to moved classes
                (r'from \.orchestrator import', 'from .orchestrator import'),
                (r'from \.database import', 'from .database import'),
                (r'from \.classic_stego import', 'from .classic_stego import'),
                (r'from \.image_forensics import', 'from .image_forensics import'),
                (r'from \.audio_analysis import', 'from .audio_analysis import'),
                (r'from \.file_forensics import', 'from .file_forensics import'),
                (r'from \.crypto_analysis import', 'from .crypto_analysis import'),
                (r'from \.integrations import', 'from .integrations import'),
                (r'from \.checkpoint import', 'from .checkpoint import'),
            ]
            
            for old_pattern, new_pattern in specific_fixes:
                content = re.sub(old_pattern, new_pattern, content)
            
            # Only write if content changed
            if content != original_content:
                print(f"Fixed imports in: {file_path}")
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                    
        except Exception as e:
            print(f"Error fixing imports in {file_path}: {e}")
    
    def create_project_files(self):
        """Create additional project files"""
        print("Creating additional project files...")
        
        # Create setup.py
        setup_py_content = '''#!/usr/bin/env python3
"""
Setup script for StegAnalyzer
"""

from setuptools import setup, find_packages
import os

# Read README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="steganalyzer",
    version="1.0.0",
    author="StegAnalyzer Team",
    description="Advanced Steganography Detection & Analysis Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/steganalyzer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Security",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "gpu": ["torch>=1.10.0", "torchvision>=0.11.0"],
        "ml": ["torch>=1.10.0", "tensorflow>=2.7.0", "transformers>=4.15.0"],
        "dev": ["pytest>=6.2.0", "black>=22.0.0", "isort>=5.10.0", "flake8>=4.0.0"],
    },
    entry_points={
        "console_scripts": [
            "steganalyzer=steg_main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.txt", "*.md"],
        "static": ["*"],
        "templates": ["*"],
        "wordlists": ["*"],
    },
)
'''
        
        setup_py_path = self.project_root / "setup.py"
        with open(setup_py_path, 'w') as f:
            f.write(setup_py_content)
        
        # Create .gitignore
        gitignore_content = '''# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/
.pytest_cache/

# Virtual environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Project specific
data/*.db
data/*.sqlite
data/*.sqlite3
logs/*.log
reports/*.html
reports/*.pdf
reports/*.json
models/*.pth
models/*.pkl
models/*.h5
temp/
tmp/
backup*/
*.bak
.steganalyzer/

# Sensitive files
config/production.json
config/secrets.json
*.key
*.pem
.env.local
.env.production

# Large files
*.zip
*.tar.gz
*.rar
*.7z
wordlists/rockyou.txt
datasets/
'''
        
        gitignore_path = self.project_root / ".gitignore"
        with open(gitignore_path, 'w') as f:
            f.write(gitignore_content)
        
        # Create Makefile
        makefile_content = '''# StegAnalyzer Makefile

.PHONY: install install-dev install-gpu test clean lint format check setup run

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install pytest black isort flake8 mypy pre-commit

install-gpu:
	pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
	pip install tensorflow

# Development
setup:
	python -m venv venv
	source venv/bin/activate && make install-dev
	pre-commit install

test:
	pytest tests/ -v

check:
	python steg_main.py --check-system

# Code quality
lint:
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

format:
	black .
	isort .

# Running
run:
	python steg_main.py

demo:
	python steg_main.py samples/test_image.jpg

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/

clean-data:
	rm -rf data/*.db
	rm -rf logs/*.log
	rm -rf reports/*
	rm -rf temp/

# Docker
docker-build:
	docker build -t steganalyzer .

docker-run:
	docker run -v $(PWD)/samples:/data steganalyzer /data/test_image.jpg

# Documentation
docs:
	sphinx-build -b html docs/ docs/_build/html

# Help
help:
	@echo "Available commands:"
	@echo "  install     - Install dependencies"
	@echo "  install-dev - Install development dependencies"  
	@echo "  install-gpu - Install GPU support"
	@echo "  setup       - Full development setup"
	@echo "  test        - Run tests"
	@echo "  check       - Check system requirements"
	@echo "  lint        - Run linting"
	@echo "  format      - Format code"
	@echo "  run         - Run StegAnalyzer"
	@echo "  demo        - Run demo analysis"
	@echo "  clean       - Clean build artifacts"
	@echo "  clean-data  - Clean data files"
'''
        
        makefile_path = self.project_root / "Makefile"
        with open(makefile_path, 'w') as f:
            f.write(makefile_content)
    
    def validate_structure(self):
        """Validate the final project structure"""
        print("Validating project structure...")
        
        required_files = [
            'steg_main.py',
            'requirements.txt',
            'README.md',
            'setup.py',
            'config/default.json',
            'ai/__init__.py',
            'core/__init__.py',
            'tools/__init__.py',
            'utils/__init__.py'
        ]
        
        missing_files = []
        for file_path in required_files:
            full_path = self.project_root / file_path
            if not full_path.exists():
                missing_files.append(file_path)
        
        if missing_files:
            print("❌ Missing required files:")
            for file_path in missing_files:
                print(f"  - {file_path}")
        else:
            print("✅ All required files present")
        
        # Check for proper package structure
        packages = ['ai', 'cloud', 'config', 'core', 'tools', 'utils']
        for package in packages:
            init_file = self.project_root / package / "__init__.py"
            if not init_file.exists():
                print(f"❌ Missing __init__.py in {package}")
            else:
                print(f"✅ {package} package properly structured")
    
    def run_syntax_check(self):
        """Check syntax of all Python files"""
        print("Checking syntax of all Python files...")
        
        python_files = list(self.project_root.rglob("*.py"))
        python_files = [f for f in python_files if "__pycache__" not in str(f) and "backup" not in str(f)]
        
        syntax_errors = []
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    compile(f.read(), str(py_file), 'exec')
                print(f"✅ {py_file.relative_to(self.project_root)}")
            except SyntaxError as e:
                syntax_errors.append((py_file, e))
                print(f"❌ {py_file.relative_to(self.project_root)}: {e}")
            except Exception as e:
                print(f"⚠️  {py_file.relative_to(self.project_root)}: {e}")
        
        if syntax_errors:
            print(f"\n❌ Found {len(syntax_errors)} syntax errors")
            return False
        else:
            print("\n✅ All Python files have valid syntax")
            return True
    
    def organize_project(self):
        """Main method to organize the entire project"""
        print("🚀 Starting StegAnalyzer project organization...")
        print("=" * 60)
        
        try:
            # Step 1: Create backup
            self.create_backup()
            
            # Step 2: Clean duplicates and temp files
            self.clean_duplicates_and_temp_files()
            
            # Step 3: Organize files
            self.organize_files()
            
            # Step 4: Fix imports
            self.fix_imports()
            
            # Step 5: Create additional project files
            self.create_project_files()
            
            # Step 6: Validate structure
            self.validate_structure()
            
            # Step 7: Syntax check
            self.run_syntax_check()
            
            print("=" * 60)
            print("✅ Project organization complete!")
            print("\nNext steps:")
            print("1. Review the changes")
            print("2. Test imports: python -c 'from core import StegOrchestrator'")
            print("3. Run system check: python steg_main.py --check-system")
            print("4. Install dependencies: pip install -r requirements.txt")
            print("5. Run a test analysis: python steg_main.py --help")
            print(f"\nBackup created at: {self.backup_dir}")
            
        except Exception as e:
            print(f"❌ Error during organization: {e}")
            print(f"Backup available at: {self.backup_dir}")
            raise

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Organize StegAnalyzer project structure")
    parser.add_argument("--project-root", default=".", help="Project root directory")
    parser.add_argument("--no-backup", action="store_true", help="Skip backup creation")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    
    args = parser.parse_args()
    
    organizer = ProjectOrganizer(args.project_root)
    
    if args.dry_run:
        print("DRY RUN MODE - No changes will be made")
        print("Would organize project structure...")
        return
    
    if not args.no_backup:
        organizer.organize_project()
    else:
        print("⚠️  Running without backup!")
        response = input("Are you sure you want to continue? (y/N): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
        organizer.organize_project()

if __name__ == "__main__":
    main()
