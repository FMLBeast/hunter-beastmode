#!/usr/bin/env python3
"""
Final targeted fix for the remaining issues:
1. Fix method name: file_signature -> magic_analysis
2. Add missing database method: store_analysis_result
"""

import re
from pathlib import Path

def main():
    """Apply final targeted fixes"""
    print("🎯 Applying final targeted fixes...")
    
    project_root = Path(".")
    
    # Fix 1: Orchestrator method names
    fix_orchestrator_method_names(project_root)
    
    # Fix 2: Database missing method
    fix_database_missing_method(project_root)
    
    print("✅ Final fixes applied!")
    print("🚀 Test now with: python3 steg_main.py image.png --verbose")

def fix_orchestrator_method_names(project_root):
    """Fix incorrect method names in orchestrator"""
    print("🔧 Fixing orchestrator method names...")
    
    orchestrator_file = project_root / "core" / "orchestrator.py"
    
    if not orchestrator_file.exists():
        print("   ❌ orchestrator.py not found!")
        return
    
    with open(orchestrator_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix the method name in _create_generic_analysis_tasks
    content = re.sub(
        r'"file_signature"',
        '"magic_analysis"',
        content
    )
    
    # Also fix the task method name
    content = re.sub(
        r'if "file_signature" not in completed:',
        'if "magic_analysis" not in completed:',
        content
    )
    
    content = re.sub(
        r'method="file_signature"',
        'method="magic_analysis"',
        content
    )
    
    with open(orchestrator_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("   ✅ Method names fixed: file_signature -> magic_analysis")

def fix_database_missing_method(project_root):
    """Add missing store_analysis_result method to database"""
    print("🗄️  Adding missing database method...")
    
    db_file = project_root / "core" / "database.py"
    
    if not db_file.exists():
        print("   ❌ database.py not found!")
        return
    
    with open(db_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if store_analysis_result method exists
    if 'def store_analysis_result' in content:
        print("   ✅ store_analysis_result method already exists")
        return
    
    # Find a good place to add the method (after store_finding)
    insertion_point = content.find('    async def store_finding(')
    if insertion_point == -1:
        # Fallback: add at end of class
        insertion_point = content.rfind('\n\n')
        if insertion_point == -1:
            insertion_point = len(content) - 100
    else:
        # Find the end of store_finding method
        method_start = insertion_point
        brace_count = 0
        in_method = False
        i = method_start
        
        while i < len(content):
            if content[i:i+4] == 'def ' and in_method:
                # Found next method
                insertion_point = i - 8  # Back up to add before next method
                break
            elif content[i:i+8] == 'async def' and in_method:
                # Found next async method
                insertion_point = i - 8
                break
            elif content[i:i+5] == '    #' and in_method and content[i-1] == '\n':
                # Found comment at class level, likely end of method
                insertion_point = i
                break
            elif i > method_start + 20:
                in_method = True
            i += 1
        
        if insertion_point == method_start:
            # Couldn't find end, add at end of file
            insertion_point = len(content) - 50
    
    # Method to add
    missing_method = '''
    async def store_analysis_result(self, session_id: str, method: str, results: list):
        """Store analysis results from tools"""
        if not results:
            return
        
        try:
            for result in results:
                if isinstance(result, dict):
                    await self.store_finding(session_id, result)
                else:
                    self.logger.warning(f"Invalid result format: {type(result)}")
        except Exception as e:
            self.logger.error(f"Error storing analysis results for method {method}: {e}")
'''
    
    # Insert the method
    new_content = content[:insertion_point] + missing_method + content[insertion_point:]
    
    with open(db_file, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("   ✅ Added store_analysis_result method to database")

if __name__ == "__main__":
    main()
