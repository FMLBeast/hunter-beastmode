#!/usr/bin/env python3
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
    
    print("\n🎉 All core components working!")
    return True

if __name__ == "__main__":
    success = test_core_functionality()
    if success:
        print("\n✅ Ready to run StegAnalyzer!")
        print("Next: python steg_main.py --check-system")
    else:
        print("\n❌ Some components need attention")
