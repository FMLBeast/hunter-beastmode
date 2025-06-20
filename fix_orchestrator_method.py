#!/usr/bin/env python3
"""
Quick fix for the orchestrator method call
"""

from pathlib import Path

def fix_orchestrator_method():
    """Fix the FileAnalyzer method call in orchestrator"""
    
    print("🔧 Fixing Orchestrator Method Call")
    print("=" * 50)
    
    orchestrator_file = Path("core/orchestrator.py")
    
    if not orchestrator_file.exists():
        print("❌ core/orchestrator.py not found")
        return False
    
    # Read the current content
    with open(orchestrator_file, 'r') as f:
        content = f.read()
    
    # Fix the method call
    # Change: file_info = await self.file_analyzer.analyze(file_path)
    # To: file_info = await self.file_analyzer.analyze_file(file_path)
    
    old_line = "file_info = await self.file_analyzer.analyze(file_path)"
    new_line = "file_info = await self.file_analyzer.analyze_file(file_path)"
    
    if old_line in content:
        content = content.replace(old_line, new_line)
        print(f"✅ Fixed method call: analyze() -> analyze_file()")
    else:
        print("⚠️  Expected line not found, trying alternative patterns")
        
        # Try variations
        patterns = [
            ("await self.file_analyzer.analyze(", "await self.file_analyzer.analyze_file("),
            ("self.file_analyzer.analyze(", "self.file_analyzer.analyze_file(")
        ]
        
        fixed = False
        for old_pattern, new_pattern in patterns:
            if old_pattern in content:
                content = content.replace(old_pattern, new_pattern)
                print(f"✅ Fixed pattern: {old_pattern} -> {new_pattern}")
                fixed = True
        
        if not fixed:
            print("❌ Could not find method call to fix")
            return False
    
    # Write the fixed content
    with open(orchestrator_file, 'w') as f:
        f.write(content)
    
    print("✅ Orchestrator method call fixed")
    return True

def test_fix():
    """Test if the fix worked"""
    
    print("\n🧪 Testing Fix")
    print("=" * 50)
    
    try:
        import sys
        sys.path.insert(0, '.')
        
        from config.steg_config import Config
        from core.file_analyzer import FileAnalyzer
        from core.database import DatabaseManager
        
        config = Config()
        db = DatabaseManager(config.database)
        file_analyzer = FileAnalyzer(config)
        
        # Check if analyze_file method exists
        if hasattr(file_analyzer, 'analyze_file'):
            print("✅ FileAnalyzer.analyze_file() method exists")
        else:
            print("❌ FileAnalyzer.analyze_file() method not found")
            return False
        
        # Check if the method is callable
        if callable(getattr(file_analyzer, 'analyze_file')):
            print("✅ FileAnalyzer.analyze_file() is callable")
        else:
            print("❌ FileAnalyzer.analyze_file() is not callable")
            return False
        
        print("✅ Method signature is correct")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main function"""
    print("🔧 StegAnalyzer Orchestrator Method Fix")
    print("=" * 60)
    
    # Fix the method call
    fix_success = fix_orchestrator_method()
    
    # Test the fix
    test_success = test_fix()
    
    print("\n" + "=" * 60)
    
    if fix_success and test_success:
        print("🎉 SUCCESS! Orchestrator method call has been fixed.")
        print("\nNow try:")
        print("   python steg_main.py image.png")
    else:
        print("❌ Fix incomplete. Manual intervention may be needed.")
    
    return fix_success and test_success

if __name__ == "__main__":
    main()