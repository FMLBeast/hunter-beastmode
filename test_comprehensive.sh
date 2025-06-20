#!/bin/bash
# Quick test of comprehensive cascade system
echo "🧪 Testing comprehensive cascade system..."

# Test single file
if [ "$1" ]; then
    echo "🔍 Testing on: $1"
    python comprehensive_analysis.py "$1" --output-dir test_output
    echo "✅ Test complete - check test_output/ for results"
else
    echo "Usage: ./test_comprehensive.sh image.png"
fi
