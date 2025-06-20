#!/usr/bin/env python3
"""
Quick test to verify the database fix and setup
"""

import sys
import os
from pathlib import Path

def test_database_fix():
    """Test if the database configuration issue is fixed"""
    
    print("🔧 Testing Database Configuration Fix")
    print("=" * 50)
    
    try:
        # Test 1: Import config
        sys.path.insert(0, '.')
        from config.steg_config import Config
        print("✅ Config import works")
        
        # Test 2: Load config
        config = Config()
        print("✅ Config loading works")
        
        # Test 3: Check database config has 'type' attribute
        db_type = config.database.type
        print(f"✅ Database type: {db_type}")
        
        # Test 4: Import DatabaseManager
        from core.database import DatabaseManager
        print("✅ DatabaseManager import works")
        
        # Test 5: Initialize DatabaseManager (this was failing before)
        db_manager = DatabaseManager(config.database)
        print("✅ DatabaseManager initialization works")
        
        # Test 6: Verify database file creation
        if db_type == "sqlite":
            db_path = Path(config.database.path)
            if db_path.exists():
                print(f"✅ Database file created: {db_path}")
            else:
                print(f"⚠️  Database file not found: {db_path}")
        
        print("\n🎉 Database fix successful!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def setup_missing_files():
    """Create missing files and directories"""
    
    print("\n📁 Setting up missing files and directories")
    print("=" * 50)
    
    # Create wordlists directory and files
    wordlists_dir = Path("wordlists")
    wordlists_dir.mkdir(exist_ok=True)
    
    # Create common_passwords.txt if missing
    common_passwords = wordlists_dir / "common_passwords.txt"
    if not common_passwords.exists():
        with open(common_passwords, 'w') as f:
            f.write("# Common passwords for steganography tools\n")
            f.write("password\n123456\npassword123\nadmin\ntest\n")
            f.write("secret\nhidden\nsteganography\nstego\nimage\n")
            f.write("data\nfile\nhide\nembed\nextract\n")
        print(f"✅ Created {common_passwords}")
    
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    print(f"✅ Created {data_dir}")
    
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    print(f"✅ Created {logs_dir}")
    
    # Create reports directory
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    print(f"✅ Created {reports_dir}")

def check_optional_dependencies():
    """Check for optional dependencies and give warnings"""
    
    print("\n📦 Checking Optional Dependencies")
    print("=" * 50)
    
    # Check CUDA for GPU support
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print("✅ CUDA available for GPU acceleration")
        else:
            print("⚠️  CUDA not available - GPU acceleration disabled")
    except ImportError:
        print("⚠️  PyTorch not installed - GPU acceleration disabled")
    
    # Check Anthropic API key
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if api_key:
        print("✅ Anthropic API key found")
    else:
        print("⚠️  Anthropic API key not set - LLM features disabled")
        print("   Set ANTHROPIC_API_KEY environment variable to enable")
    
    # Check steganography tools
    tools_to_check = ['steghide', 'binwalk', 'exiftool', 'strings']
    for tool in tools_to_check:
        import shutil
        if shutil.which(tool):
            print(f"✅ {tool} found")
        else:
            print(f"⚠️  {tool} not found - some features may be limited")

def main():
    """Main test function"""
    print("🧪 StegAnalyzer Setup and Fix Test")
    print("=" * 60)
    
    # Setup missing files
    setup_missing_files()
    
    # Test the database fix
    success = test_database_fix()
    
    # Check dependencies
    check_optional_dependencies()
    
    print("\n" + "=" * 60)
    
    if success:
        print("🎉 SUCCESS! The database configuration issue has been fixed.")
        print("\nNext steps:")
        print("1. Run: python steg_main.py --check-system")
        print("2. Test with: python steg_main.py image.png")
        print("\nNote: Some warnings are normal for optional features.")
    else:
        print("❌ The fix didn't work. Please check the error messages above.")
    
    return success

if __name__ == "__main__":
    main()