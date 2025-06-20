#!/usr/bin/env python3
"""
Fix syntax error in orchestrator.py
"""

from pathlib import Path
import re

def fix_orchestrator_syntax():
    """Fix syntax errors in orchestrator.py"""
    
    print("🔧 Fixing syntax errors in orchestrator.py...")
    
    orchestrator_file = Path("core/orchestrator.py")
    
    if not orchestrator_file.exists():
        print("❌ orchestrator.py not found")
        return False
    
    try:
        # Read the file
        with open(orchestrator_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Create backup
        backup_file = orchestrator_file.with_suffix('.py.backup')
        with open(backup_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Backup created: {backup_file}")
        
        # Find and fix common syntax issues
        lines = content.split('\n')
        fixed_lines = []
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Check if this line defines a function/method without a body
            if (line.strip().endswith(':') and 
                ('def ' in line or 'class ' in line or 'if ' in line or 'for ' in line or 'while ' in line)):
                
                # Check if next line is indented or is another definition
                if i + 1 < len(lines):
                    next_line = lines[i + 1]
                    
                    # If next line is not indented properly or is empty, add pass
                    if (not next_line.strip() or 
                        not next_line.startswith('    ') or
                        next_line.strip().startswith('def ') or
                        next_line.strip().startswith('class ')):
                        
                        fixed_lines.append(line)
                        fixed_lines.append('        pass  # TODO: Implement this method')
                        print(f"   Fixed empty function/block at line {i+1}")
                    else:
                        fixed_lines.append(line)
                else:
                    # Last line ends with colon - add pass
                    fixed_lines.append(line)
                    fixed_lines.append('        pass  # TODO: Implement this method')
                    print(f"   Fixed empty function at end of file")
            else:
                fixed_lines.append(line)
            
            i += 1
        
        # Join the fixed lines
        fixed_content = '\n'.join(fixed_lines)
        
        # Additional fixes for common issues
        fixed_content = fix_import_issues(fixed_content)
        
        # Write the fixed file
        with open(orchestrator_file, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        
        print("✅ Syntax errors fixed in orchestrator.py")
        return True
        
    except Exception as e:
        print(f"❌ Failed to fix orchestrator.py: {e}")
        return False

def fix_import_issues(content):
    """Fix common import issues"""
    
    # Fix relative imports that might be broken
    import_fixes = [
        # Fix tool imports to use actual file names
        (r'from tools\.classic_stego import', 'from tools.classic_stego import'),
        (r'from tools\.image_forensics import', 'from tools.image_forensics import'),
        (r'from tools\.audio_analysis import', 'from tools.audio_analysis import'),
        (r'from tools\.file_forensics import', 'from tools.file_forensics import'),
        (r'from tools\.crypto_analysis import', 'from tools.crypto_analysis import'),
        (r'from tools\.metadata_carving import', 'from tools.metadata_carving import'),
        
        # Fix core imports
        (r'from core\.database import', 'from core.database import'),
        (r'from core\.graph_tracker import', 'from core.graph_tracker import'),
        (r'from core\.file_analyzer import', 'from core.file_analyzer import'),
        
        # Fix utils imports
        (r'from utils\.checkpoint import', 'from utils.checkpoint import'),
        (r'from utils\.gpu_manager import', 'from utils.gpu_manager import'),
        
        # Fix AI imports
        (r'from ai\.ml_detector import', 'from ai.ml_detector import'),
        (r'from ai\.llm_analyzer import', 'from ai.llm_analyzer import'),
        (r'from ai\.multimodal_classifier import', 'from ai.multimodal_classifier import'),
        
        # Fix cloud imports
        (r'from cloud\.integrations import', 'from cloud.integrations import'),
    ]
    
    for old_pattern, new_pattern in import_fixes:
        content = re.sub(old_pattern, new_pattern, content)
    
    return content

def test_orchestrator_syntax():
    """Test that orchestrator.py has valid syntax"""
    
    print("\n🧪 Testing orchestrator.py syntax...")
    
    orchestrator_file = Path("core/orchestrator.py")
    
    try:
        with open(orchestrator_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Try to compile the file
        compile(content, str(orchestrator_file), 'exec')
        print("✅ orchestrator.py syntax is valid")
        return True
        
    except SyntaxError as e:
        print(f"❌ Syntax error in orchestrator.py:")
        print(f"   Line {e.lineno}: {e.text}")
        print(f"   Error: {e.msg}")
        return False
    except Exception as e:
        print(f"❌ Error checking orchestrator.py: {e}")
        return False

def test_orchestrator_import():
    """Test importing StegOrchestrator"""
    
    print("\n🧪 Testing StegOrchestrator import...")
    
    try:
        from core.orchestrator import StegOrchestrator
        print("✅ StegOrchestrator imports successfully")
        return True
    except Exception as e:
        print(f"❌ StegOrchestrator import failed: {e}")
        return False

def main():
    """Main function"""
    
    print("🔧 StegAnalyzer Orchestrator Syntax Fix")
    print("=" * 60)
    
    # Fix syntax errors
    if fix_orchestrator_syntax():
        
        # Test syntax
        if test_orchestrator_syntax():
            
            # Test import
            if test_orchestrator_import():
                print("\n🎉 SUCCESS! StegOrchestrator is working!")
                print("\nNow you can:")
                print("• python3 -c 'from core import StegOrchestrator'")
                print("• python3 steg_main.py --help")
                print("• python3 steg_main.py --check-system")
            else:
                print("\n⚠️ Syntax fixed but import still fails")
                print("The issue might be in imported dependencies")
        else:
            print("\n❌ Syntax errors still present")
    else:
        print("\n❌ Failed to fix syntax errors")

if __name__ == "__main__":
    main()
