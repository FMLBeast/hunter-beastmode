#!/usr/bin/env python3
"""
Quick fix for syntax error in orchestrator.py
"""

from pathlib import Path

def quick_fix_syntax():
    """Quick fix for the specific syntax error"""
    
    print("🔧 Quick syntax fix for orchestrator.py...")
    
    orchestrator_file = Path("core/orchestrator.py")
    
    if not orchestrator_file.exists():
        print("❌ orchestrator.py not found")
        return
    
    try:
        # Read the file
        with open(orchestrator_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Find the problematic line (around line 102-103)
        for i, line in enumerate(lines):
            line_num = i + 1
            
            # Look for function definitions without bodies
            if (line.strip().endswith(':') and 'def ' in line):
                # Check if next line exists and is properly indented
                if i + 1 < len(lines):
                    next_line = lines[i + 1]
                    
                    # If next line is not indented or is empty, add pass
                    if (not next_line.strip() or 
                        not next_line.startswith('    ') or
                        next_line.strip().startswith('def ') or
                        next_line.strip().startswith('class ') or
                        next_line.strip().startswith('async def ') or
                        next_line.strip().startswith('return')):
                        
                        print(f"   Found empty function at line {line_num}: {line.strip()}")
                        # Insert a pass statement
                        lines.insert(i + 1, '        pass  # TODO: Implement\n')
                        print(f"   Added 'pass' statement after line {line_num}")
                        break
                else:
                    # Function at end of file without body
                    print(f"   Found empty function at end of file: {line.strip()}")
                    lines.append('        pass  # TODO: Implement\n')
                    print("   Added 'pass' statement at end of file")
                    break
        
        # Write the fixed file
        with open(orchestrator_file, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        print("✅ Quick syntax fix applied")
        
        # Test the fix
        try:
            with open(orchestrator_file, 'r', encoding='utf-8') as f:
                content = f.read()
            compile(content, str(orchestrator_file), 'exec')
            print("✅ Syntax is now valid")
            
            # Test import
            try:
                from core.orchestrator import StegOrchestrator
                print("✅ StegOrchestrator imports successfully!")
            except Exception as e:
                print(f"⚠️ Import issue (but syntax is fixed): {e}")
                
        except SyntaxError as e:
            print(f"❌ Syntax error still exists: {e}")
            print(f"   Line {e.lineno}: {e.text}")
            
    except Exception as e:
        print(f"❌ Error fixing file: {e}")

if __name__ == "__main__":
    quick_fix_syntax()
