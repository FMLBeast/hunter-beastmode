#!/usr/bin/env python3
"""
Fix all missing dependencies and undefined variables in StegAnalyzer
"""

import os
import re
import subprocess
import sys
from pathlib import Path

def create_requirements_txt():
    """Create requirements.txt with all missing dependencies"""
    
    requirements = """# Core dependencies
numpy>=1.21.0
Pillow>=9.0.0
pathlib2>=2.3.0
psutil>=5.8.0
python-magic>=0.4.24
requests>=2.25.0

# AI/ML dependencies
anthropic>=0.25.0
torch>=1.10.0
tensorflow>=2.7.0
scikit-learn>=1.0.0
scipy>=1.7.0

# Image analysis
opencv-python>=4.5.0
easyocr>=1.6.0

# Audio analysis  
librosa>=0.8.1
aubio>=0.4.9
deepspeech>=0.9.3

# File analysis
PyPDF2>=3.0.0
pdfplumber>=0.7.0
python-docx>=0.8.11
oletools>=0.60

# Cryptography
cryptography>=36.0.0
pycryptodome>=3.15.0

# Web/API
aiohttp>=3.8.0
fastapi>=0.70.0
jinja2>=3.0.0

# Visualization
matplotlib>=3.5.0
plotly>=5.0.0

# Database
sqlalchemy>=1.4.0
sqlite3  # Built-in

# Development
pytest>=6.0.0
black>=21.0.0
flake8>=3.9.0
"""
    
    with open("requirements.txt", "w") as f:
        f.write(requirements)
    
    print("✅ Created requirements.txt")

def fix_audio_analysis():
    """Fix audio_analysis.py missing imports and undefined variables"""
    
    file_path = Path("tools/audio_analysis.py")
    if not file_path.exists():
        print(f"❌ {file_path} not found")
        return
    
    with open(file_path, "r") as f:
        content = f.read()
    
    # Add missing re import at the top
    if "import re" not in content:
        # Find where imports start and add re
        lines = content.split('\n')
        import_section = []
        code_section = []
        in_imports = True
        
        for line in lines:
            if line.startswith('import ') or line.startswith('from ') or line.strip().startswith('#') or line.strip() == '':
                if in_imports:
                    import_section.append(line)
                else:
                    code_section.append(line)
            else:
                if in_imports:
                    in_imports = False
                    # Add missing imports here
                    if "import re" not in '\n'.join(import_section):
                        import_section.append("import re")
                code_section.append(line)
        
        content = '\n'.join(import_section + code_section)
    
    # Fix undefined file_path variables by adding proper parameter
    # Line 273: def some_function(): -> def some_function(file_path):
    content = re.sub(
        r'(def \w+\([^)]*)\)(\s*:\s*\n.*?file_path)',
        r'\1, file_path)\2',
        content,
        flags=re.DOTALL
    )
    
    # Alternative approach - look for specific patterns
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'file_path' in line and i > 0:
            # Look for the function definition above
            j = i - 1
            while j >= 0 and not lines[j].strip().startswith('def '):
                j -= 1
            
            if j >= 0 and 'file_path' not in lines[j]:
                # Add file_path parameter
                if '()' in lines[j]:
                    lines[j] = lines[j].replace('():', '(file_path):')
                elif lines[j].endswith('):'):
                    lines[j] = lines[j][:-2] + ', file_path):'
    
    content = '\n'.join(lines)
    
    with open(file_path, "w") as f:
        f.write(content)
    
    print(f"✅ Fixed {file_path}")

def fix_image_forensics():
    """Fix image_forensics.py undefined file_path variables"""
    
    file_path = Path("tools/image_forensics.py") 
    if not file_path.exists():
        print(f"❌ {file_path} not found")
        return
    
    with open(file_path, "r") as f:
        content = f.read()
    
    # Fix undefined file_path by adding to function signatures
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'file_path' in line and '"file_path" is not defined' in str(line):
            # Find the function definition
            j = i - 1
            while j >= 0 and not lines[j].strip().startswith('def '):
                j -= 1
            
            if j >= 0 and 'file_path' not in lines[j]:
                # Add file_path parameter
                if '()' in lines[j]:
                    lines[j] = lines[j].replace('():', '(file_path):')
                elif lines[j].endswith('):'):
                    lines[j] = lines[j][:-2] + ', file_path):'
    
    content = '\n'.join(lines)
    
    with open(file_path, "w") as f:
        f.write(content)
    
    print(f"✅ Fixed {file_path}")

def fix_import_statements():
    """Fix import statements throughout codebase"""
    
    # Pattern replacements for common import issues
    replacements = [
        # Fix relative imports
        (r'from \.(\w+) import', r'from \1 import'),
        # Fix specific tool imports
        (r'from tools\.(\w+)_tools import', r'from tools.\1 import'),
        # Fix core imports
        (r'from core\.steg_(\w+) import', r'from core.\1 import'),
        # Fix utils imports  
        (r'from utils\.(\w+)_manager import', r'from utils.\1 import'),
    ]
    
    # Files to fix
    python_files = []
    for pattern in ["**/*.py"]:
        python_files.extend(Path(".").glob(pattern))
    
    for file_path in python_files:
        if file_path.name.startswith('.'):
            continue
            
        try:
            with open(file_path, "r") as f:
                content = f.read()
            
            original_content = content
            
            # Apply replacements
            for pattern, replacement in replacements:
                content = re.sub(pattern, replacement, content)
            
            # Only write if changed
            if content != original_content:
                with open(file_path, "w") as f:
                    f.write(content)
                print(f"✅ Fixed imports in {file_path}")
                
        except Exception as e:
            print(f"❌ Error fixing {file_path}: {e}")

def create_optional_import_wrapper():
    """Create a wrapper for optional imports"""
    
    wrapper_content = '''"""
Optional import wrapper - handles missing dependencies gracefully
"""

import logging
import warnings

def optional_import(module_name, package=None, min_version=None):
    """
    Import a module optionally, returning None if not available
    
    Args:
        module_name: Name of module to import
        package: Package name if different from module
        min_version: Minimum version required
    
    Returns:
        Module if available, None otherwise
    """
    try:
        module = __import__(module_name, fromlist=[''])
        
        if min_version and hasattr(module, '__version__'):
            from packaging import version
            if version.parse(module.__version__) < version.parse(min_version):
                warnings.warn(f"{module_name} version {module.__version__} < {min_version}")
                
        return module
        
    except ImportError as e:
        logging.warning(f"Optional dependency {module_name} not available: {e}")
        return None

# Pre-import common optional dependencies
anthropic = optional_import('anthropic')
easyocr = optional_import('easyocr')
aubio = optional_import('aubio')
deepspeech = optional_import('deepspeech')
PyPDF2 = optional_import('PyPDF2')
pdfplumber = optional_import('pdfplumber')

# Cryptography imports
try:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.backends import default_backend
    cryptography_available = True
except ImportError:
    cryptography_available = False
    Cipher = algorithms = modes = default_backend = None

# OLE tools
try:
    from oletools import olevba, rtfobj
    oletools_available = True
except ImportError:
    oletools_available = False
    olevba = rtfobj = None
'''

    with open("utils/optional_imports.py", "w") as f:
        f.write(wrapper_content)
    
    print("✅ Created optional_imports.py wrapper")

def fix_specific_files():
    """Fix specific files with targeted fixes"""
    
    # Fix audio_analysis.py line 273 and 707
    audio_file = Path("tools/audio_analysis.py")
    if audio_file.exists():
        with open(audio_file, "r") as f:
            lines = f.readlines()
        
        # Add re import at top if missing
        has_re_import = any("import re" in line for line in lines[:20])
        if not has_re_import:
            # Find good place to insert import
            insert_idx = 0
            for i, line in enumerate(lines):
                if line.startswith("import ") or line.startswith("from "):
                    insert_idx = i + 1
            lines.insert(insert_idx, "import re\n")
        
        # Fix function definitions that use file_path without parameter
        for i, line in enumerate(lines):
            if i == 272:  # Line 273 (0-indexed)
                # Look backwards for function definition
                j = i
                while j >= 0 and not lines[j].strip().startswith('def '):
                    j -= 1
                if j >= 0 and 'file_path' not in lines[j]:
                    if '():' in lines[j]:
                        lines[j] = lines[j].replace('():', '(file_path):')
                    elif lines[j].rstrip().endswith('):'):
                        lines[j] = lines[j].rstrip()[:-2] + ', file_path):\n'
        
        with open(audio_file, "w") as f:
            f.writelines(lines)
        
        print(f"✅ Fixed specific issues in {audio_file}")
    
    # Fix image_forensics.py lines 558, 569  
    image_file = Path("tools/image_forensics.py")
    if image_file.exists():
        with open(image_file, "r") as f:
            lines = f.readlines()
        
        # Fix function definitions for lines that use file_path
        for line_num in [557, 568]:  # 0-indexed for lines 558, 569
            if line_num < len(lines):
                j = line_num
                while j >= 0 and not lines[j].strip().startswith('def '):
                    j -= 1
                if j >= 0 and 'file_path' not in lines[j]:
                    if '():' in lines[j]:
                        lines[j] = lines[j].replace('():', '(file_path):')
                    elif lines[j].rstrip().endswith('):'):
                        lines[j] = lines[j].rstrip()[:-2] + ', file_path):\n'
        
        with open(image_file, "w") as f:
            f.writelines(lines)
        
        print(f"✅ Fixed specific issues in {image_file}")

def install_dependencies():
    """Install dependencies with error handling"""
    
    print("📦 Installing dependencies...")
    
    # Install in groups to handle failures gracefully
    core_deps = [
        "numpy", "Pillow", "psutil", "requests", "pathlib2"
    ]
    
    ai_deps = [
        "anthropic", "torch", "scikit-learn", "scipy"
    ]
    
    analysis_deps = [
        "opencv-python", "librosa", "PyPDF2", "cryptography"
    ]
    
    optional_deps = [
        "easyocr", "aubio", "deepspeech", "pdfplumber", "oletools"
    ]
    
    def install_group(deps, group_name):
        print(f"\n📥 Installing {group_name}...")
        for dep in deps:
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", dep], 
                             check=True, capture_output=True)
                print(f"  ✅ {dep}")
            except subprocess.CalledProcessError:
                print(f"  ❌ {dep} (will continue without it)")
    
    install_group(core_deps, "core dependencies")
    install_group(ai_deps, "AI dependencies") 
    install_group(analysis_deps, "analysis dependencies")
    install_group(optional_deps, "optional dependencies")

def main():
    """Main fix function"""
    
    print("🔧 StegAnalyzer Dependency and Import Fixer")
    print("=" * 50)
    
    # 1. Create requirements.txt
    create_requirements_txt()
    
    # 2. Create optional import wrapper
    os.makedirs("utils", exist_ok=True)
    create_optional_import_wrapper()
    
    # 3. Fix specific undefined variable issues
    fix_specific_files()
    
    # 4. Fix audio analysis
    fix_audio_analysis()
    
    # 5. Fix image forensics
    fix_image_forensics()
    
    # 6. Fix import statements
    fix_import_statements()
    
    print("\n" + "=" * 50)
    print("✅ All fixes applied!")
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Test imports: python -c 'from tools.audio_analysis import AudioAnalysisTools'")
    print("3. Run system check: python steg_main.py --check-system")
    
    # Optionally install dependencies
    install_choice = input("\nInstall dependencies now? (y/N): ").lower().strip()
    if install_choice == 'y':
        install_dependencies()

if __name__ == "__main__":
    main()
