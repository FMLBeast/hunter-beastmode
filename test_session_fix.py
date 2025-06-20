#!/usr/bin/env python3
"""
Test the database session creation fix
"""

import asyncio
import sys
import json
from pathlib import Path

async def test_session_creation():
    """Test if session creation now works properly"""
    
    print("🔧 Testing Database Session Creation Fix")
    print("=" * 50)
    
    try:
        # Import required modules
        sys.path.insert(0, '.')
        from config.steg_config import Config
        from core.database import DatabaseManager
        
        print("✅ Imports successful")
        
        # Load config
        config = Config()
        print("✅ Config loaded")
        
        # Initialize database
        db = DatabaseManager(config.database)
        await db.initialize()
        print("✅ Database initialized")
        
        # Test session creation with config parameter
        test_file = "image.png"
        
        # Test config.to_dict() method
        config_dict = config.to_dict()
        print(f"✅ Config.to_dict() works, has {len(config_dict)} sections")
        
        session_id = await db.create_session(
            target_path=test_file,
            config=config_dict,
            batch_mode=False,
            target_dir="/tmp"
        )
        
        print(f"✅ Session created successfully: {session_id}")
        
        # Verify session was stored
        session = await db.get_session(session_id)
        if session:
            print(f"✅ Session retrieved: {session['target_path']}")
            print(f"   Target dir: {session.get('target_dir', 'Not set')}")
            print(f"   Config sections: {len(json.loads(session.get('config', '{}')))}")
        else:
            print("❌ Session not found in database")
            return False
        
        # Cleanup
        await db.close()
        print("✅ Database connection closed")
        
        print("\n🎉 Session creation fix successful!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function"""
    print("🧪 StegAnalyzer Session Creation Test")
    print("=" * 60)
    
    success = await test_session_creation()
    
    print("\n" + "=" * 60)
    
    if success:
        print("🎉 SUCCESS! Session creation is now working.")
        print("\nYou can now run:")
        print("   python steg_main.py image.png")
        print("\nThe LLM error is separate and won't affect core analysis.")
    else:
        print("❌ Session creation still has issues.")
    
    return success

if __name__ == "__main__":
    asyncio.run(main())