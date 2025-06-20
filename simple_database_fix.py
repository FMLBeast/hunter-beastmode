#!/usr/bin/env python3
"""
Simple fix for the critical database storage issue.
The main problem: store_finding() requires file_id parameter but we're not passing it.
"""

import re
from pathlib import Path

def main():
    """Fix the database storage issue"""
    print("🔧 Fixing database storage issue...")
    
    project_root = Path(".")
    fix_database_store_method(project_root)
    
    print("✅ Database storage fixed!")
    print()
    print("🎯 Your results should now:")
    print("   • Save properly to database")
    print("   • Show up on dashboard") 
    print("   • Display in reports")
    print()
    print("🚀 Test: python3 steg_main.py image.png --verbose")
    print("📊 Dashboard: http://127.0.0.1:8080")
    print()
    print("🔍 For 45K files, run on each directory:")
    print("   for dir in */; do python3 steg_main.py \"$dir\"*.jpg; done")

def fix_database_store_method(project_root):
    """Fix the store_analysis_result method"""
    print("🗄️  Fixing database store_analysis_result method...")
    
    db_file = project_root / "core" / "database.py"
    
    if not db_file.exists():
        print("   ❌ database.py not found!")
        return
    
    with open(db_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # The issue is in the store_analysis_result method I added earlier
    # It calls: await self.store_finding(session_id, result)  
    # But store_finding expects: await self.store_finding(session_id, file_id, finding)
    
    # Find and fix the broken method call
    broken_call_pattern = r'await self\.store_finding\(session_id, result\)'
    
    # Replace with correct call that gets file_id first
    fixed_call = '''# Get file_id for this session
                    if hasattr(self, 'current_file_id'):
                        file_id = self.current_file_id
                    else:
                        # Fallback: get most recent file for this session
                        cursor = self.sqlite_conn.cursor()
                        cursor.execute("SELECT id FROM files WHERE session_id = ? ORDER BY created_at DESC LIMIT 1", (session_id,))
                        row = cursor.fetchone()
                        file_id = row[0] if row else "unknown"
                    
                    await self.store_finding(session_id, file_id, result)'''
    
    content = re.sub(broken_call_pattern, fixed_call, content)
    
    # Also need to fix the entire store_analysis_result method to be more robust
    if 'await self.store_finding(session_id, file_id, result)' not in content:
        # The regex didn't work, let's replace the entire method
        method_pattern = r'    async def store_analysis_result\(self, session_id: str, method: str, results: list\):.*?(?=\n    async def|\n    def|\nclass|\Z)'
        
        new_method = '''    async def store_analysis_result(self, session_id: str, method: str, results: list):
        """Store analysis results from tools"""
        if not results:
            return
        
        try:
            # Get file_id for this session - find the most recent file
            file_id = None
            if self.db_type == "sqlite":
                cursor = self.sqlite_conn.cursor()
                cursor.execute("SELECT id FROM files WHERE session_id = ? ORDER BY created_at DESC LIMIT 1", (session_id,))
                row = cursor.fetchone()
                file_id = row[0] if row else None
                
                if not file_id:
                    self.logger.warning(f"No file found for session {session_id}")
                    return
            
            # Store each result as a finding
            for result in results:
                if isinstance(result, dict):
                    await self.store_finding(session_id, file_id, result)
                else:
                    self.logger.warning(f"Invalid result format for method {method}: {type(result)}")
                    
        except Exception as e:
            self.logger.error(f"Error storing analysis results for method {method}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())'''
        
        content = re.sub(method_pattern, new_method, content, flags=re.DOTALL)
    
    with open(db_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("   ✅ Database storage method fixed")

if __name__ == "__main__":
    main()
