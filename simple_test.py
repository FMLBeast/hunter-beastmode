#!/usr/bin/env python3
"""
Simple synchronous test for StegAnalyzer
"""

def test_basic_imports():
    """Test that all basic imports work"""
    
    print("🧪 Testing StegAnalyzer Basic Imports")
    print("=" * 50)
    
    try:
        print("Testing config...")
        from config import Config
        config = Config()
        print("✅ Config: OK")
        
        print("Testing file analyzer...")
        from core import FileAnalyzer
        analyzer = FileAnalyzer(config)
        print("✅ FileAnalyzer: OK")
        
        print("Testing graph tracker...")
        from core import GraphTracker
        # Don't initialize with database for simple test
        print("✅ GraphTracker: OK")
        
        print("Testing checkpoint manager...")
        from utils import CheckpointManager
        # Don't initialize with database for simple test
        print("✅ CheckpointManager: OK")
        
        print("Testing system checker...")
        from utils import SystemChecker
        checker = SystemChecker()
        print("✅ SystemChecker: OK")
        
        print("Testing tools...")
        # Test individual tool imports
        from tools.classic_stego import ClassicStegoTools
        print("✅ ClassicStegoTools: OK")
        
        from tools.crypto_analysis import CryptoAnalysisTools
        print("✅ CryptoAnalysisTools: OK")
        
        print("\n🎉 All basic imports working!")
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def test_cli_help():
    """Test that the CLI help works"""
    
    print("\n🧪 Testing CLI Help")
    print("=" * 50)
    
    try:
        import subprocess
        result = subprocess.run(['python3', 'steg_main.py', '--help'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✅ CLI help works!")
            print("Sample output:")
            print(result.stdout[:200] + "..." if len(result.stdout) > 200 else result.stdout)
            return True
        else:
            print(f"❌ CLI help failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ CLI test failed: {e}")
        return False

def main():
    """Main test"""
    
    print("🚀 StegAnalyzer Simple Test")
    print("=" * 60)
    
    # Test basic imports
    imports_work = test_basic_imports()
    
    # Test CLI if imports work
    if imports_work:
        cli_works = test_cli_help()
        
        if cli_works:
            print("\n" + "=" * 60)
            print("🎉 SUCCESS! StegAnalyzer is working!")
            print("\nNext steps:")
            print("1. Run system check: python3 steg_main.py --check-system")
            print("2. Analyze a file: python3 steg_main.py path/to/file.jpg")
            print("3. See full help: python3 steg_main.py --help")
            print("\nOptional: Install additional dependencies:")
            print("• python3 -m pip install torch librosa opencv-python")
        else:
            print("\n⚠️ Imports work but CLI needs attention")
    else:
        print("\n❌ Basic imports need fixes")

if __name__ == "__main__":
    main()
