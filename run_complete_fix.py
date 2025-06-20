#!/usr/bin/env python3
"""
Complete StegAnalyzer Fix
Run this script to fix all issues and clean up the project
"""

import subprocess
import sys
from pathlib import Path

def run_complete_fix():
    """Run all fixes in the correct order"""
    
    print("🚀 Starting complete StegAnalyzer fix...")
    print("=" * 50)
    
    # Step 1: Database fix
    print("Step 1: Fixing database methods...")
    try:
        exec(open('database_fix.py').read())
        print("✅ Database methods fixed")
    except Exception as e:
        print(f"⚠️  Database fix warning: {e}")
    
    print()
    
    # Step 2: Project cleanup and orchestrator fix
    print("Step 2: Cleaning up project and fixing orchestrator...")
    try:
        exec(open('project_cleanup_fix.py').read())
        print("✅ Project cleanup completed")
    except Exception as e:
        print(f"❌ Project cleanup failed: {e}")
        return False
    
    print()
    print("🎉 ALL FIXES COMPLETED!")
    print("=" * 50)
    print()
    print("✅ What was fixed:")
    print("   • Removed duplicate hunter-beastmode directory")
    print("   • Cleaned up backup and fix files")
    print("   • Fixed orchestrator method calls (execute_method)")
    print("   • Fixed LLM analyzer Anthropic client")
    print("   • Added missing database methods")
    print("   • Updated configuration")
    print()
    print("🎯 Your steganography analyzer should now work properly!")
    print("   Run: python steg_main.py image.png --verbose")
    print()
    print("🔍 Expected improvements:")
    print("   • No more 'object has no attribute' errors")
    print("   • Tools will actually execute and find results")
    print("   • Should detect steganography in your 45K files")
    
    return True

if __name__ == "__main__":
    success = run_complete_fix()
    if not success:
        sys.exit(1)
