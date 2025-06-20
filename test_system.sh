#!/bin/bash
# Test the fixed StegAnalyzer system
echo "🧪 Testing StegAnalyzer system..."

# Test dashboard
echo "📊 Testing dashboard..."
curl -s http://127.0.0.1:8080/api/status || echo "❌ Dashboard not running"

# Test tool availability
echo "🛠️  Testing tools..."
command -v zsteg >/dev/null && echo "✅ zsteg available" || echo "❌ zsteg missing"
command -v steghide >/dev/null && echo "✅ steghide available" || echo "❌ steghide missing"
command -v binwalk >/dev/null && echo "✅ binwalk available" || echo "❌ binwalk missing"

# Test Python packages
echo "🐍 Testing Python packages..."
python3 -c "import cv2; print('✅ OpenCV available')" 2>/dev/null || echo "❌ OpenCV missing"
python3 -c "import PIL; print('✅ Pillow available')" 2>/dev/null || echo "❌ Pillow missing"

echo "🎉 System test complete!"
