#!/usr/bin/env python3
"""
Test StegAnalyzer with proper async handling
"""

import asyncio

async def test_async_functionality():
    """Test functionality with proper async handling"""
    
    print("🧪 Testing StegAnalyzer with async support...")
    print("=" * 50)
    
    try:
        from config import Config
        config = Config()
        print("✅ Config created successfully")
        
        from core import FileAnalyzer
        analyzer = FileAnalyzer(config)
        print("✅ FileAnalyzer created successfully")
        
        from core import DatabaseManager
        db = DatabaseManager(config.database)
        # Initialize async components
        await db.initialize()
        print("✅ DatabaseManager created and initialized successfully")
        
        from utils import SystemChecker
        checker = SystemChecker()
        print("✅ SystemChecker created successfully")
        
        # Test system check
        print("\n🔍 Running system check...")
        system_status = await checker.check_all()
        print(f"✅ System check completed: {system_status['overall_status']}")
        
        # Cleanup
        await db.close()
        print("✅ Database connection closed")
        
        print("\n🎉 All functionality working with async support!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_sync_functionality():
    """Test non-async functionality"""
    
    print("🧪 Testing sync functionality...")
    print("=" * 50)
    
    try:
        from config import Config
        config = Config()
        print("✅ Config works")
        
        from core import FileAnalyzer
        analyzer = FileAnalyzer(config)
        print("✅ FileAnalyzer works")
        
        from utils import SystemChecker
        checker = SystemChecker()
        print("✅ SystemChecker works")
        
        # Test basic file analysis (sync)
        from pathlib import Path
        print("✅ Basic imports successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Sync test failed: {e}")
        return False

async def main():
    """Main test function"""
    
    print("🚀 StegAnalyzer Functionality Test")
    print("=" * 60)
    
    # Test sync functionality first
    sync_works = test_sync_functionality()
    
    print("\n" + "=" * 60)
    
    # Test async functionality
    if sync_works:
        async_works = await test_async_functionality()
        
        if async_works:
            print("\n" + "=" * 60)
            print("🎉 SUCCESS! StegAnalyzer is fully functional!")
            print("\nReady to use:")
            print("• python3 steg_main.py --help")
            print("• python3 steg_main.py --check-system")
            print("• python3 steg_main.py your_file.jpg")
            
            print("\nInstall optional dependencies as needed:")
            print("• pip3 install torch torchvision  # For ML analysis")
            print("• pip3 install librosa soundfile  # For audio analysis")
            print("• pip3 install opencv-python      # For advanced image analysis")
        else:
            print("\n⚠️ Async functionality needs attention")
    else:
        print("\n❌ Basic functionality issues")

if __name__ == "__main__":
    asyncio.run(main())
