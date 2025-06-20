#!/usr/bin/env python3
"""
Test script to verify the database fix works correctly
"""

import asyncio
import logging
import sys
from pathlib import Path

# Setup basic logging
logging.basicConfig(level=logging.INFO)

def test_imports():
    """Test that we can import the required modules"""
    try:
        from core.database import DatabaseManager
        print("✓ DatabaseManager import successful")
        
        from config.steg_config import Config
        print("✓ Config import successful")
        
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_database_basic():
    """Test basic database functionality"""
    try:
        from core.database import DatabaseManager
        
        # Test with dict config
        config = {
            "type": "sqlite",
            "path": "test_fix.db"
        }
        
        db = DatabaseManager(config)
        print("✓ DatabaseManager created successfully")
        
        # Check if store_file_analysis method exists
        if hasattr(db, 'store_file_analysis'):
            print("✓ store_file_analysis method exists")
        else:
            print("✗ store_file_analysis method missing")
            return False
        
        # Cleanup
        try:
            Path("test_fix.db").unlink()
        except:
            pass
        
        return True
    except Exception as e:
        print(f"✗ Database test failed: {e}")
        return False

async def test_async_operations():
    """Test async database operations"""
    try:
        from core.database import DatabaseManager
        
        config = {
            "type": "sqlite", 
            "path": "test_async.db"
        }
        
        db = DatabaseManager(config)
        
        # Test session creation
        session_id = await db.create_session("test.jpg", config={"test": True})
        print(f"✓ Created session: {session_id}")
        
        # Test file analysis storage
        file_info = {
            "file_path": "test.jpg",
            "hash": "test_hash",
            "size": 1024,
            "type": "image/jpeg",
            "metadata": {}
        }
        
        file_id = await db.store_file_analysis(session_id, file_info)
        print(f"✓ Stored file analysis: {file_id}")
        
        await db.close()
        
        # Cleanup
        try:
            Path("test_async.db").unlink()
        except:
            pass
        
        return True
    except Exception as e:
        print(f"✗ Async test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests"""
    print("🧪 Running StegAnalyzer fix verification tests...\n")
    
    print("1. Testing imports...")
    imports_ok = test_imports()
    
    print("\n2. Testing basic database functionality...")
    basic_ok = test_database_basic()
    
    print("\n3. Testing async operations...")
    async_ok = await test_async_operations()
    
    print("\n" + "="*50)
    if imports_ok and basic_ok and async_ok:
        print("🎉 All tests passed! The fix should work correctly.")
        return 0
    else:
        print("❌ Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)