#!/bin/bash
# Installation script for StegAnalyzer fix

set -e  # Exit on any error

echo "🔧 Installing StegAnalyzer fix..."

# Backup existing files
echo "📁 Creating backup of existing files..."
if [ -f "core/database.py" ]; then
    cp core/database.py core/database.py.backup.$(date +%Y%m%d_%H%M%S)
    echo "✓ Backed up core/database.py"
fi

if [ -f "steg_main.py" ]; then
    cp steg_main.py steg_main.py.backup.$(date +%Y%m%d_%H%M%S)
    echo "✓ Backed up steg_main.py"
fi

if [ -f "requirements.txt" ]; then
    cp requirements.txt requirements.txt.backup.$(date +%Y%m%d_%H%M%S)
    echo "✓ Backed up requirements.txt"
fi

# Check Python version
echo "🐍 Checking Python version..."
python3 --version || {
    echo "❌ Python 3 is required but not found"
    exit 1
}

# Check if virtual environment is active
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✓ Virtual environment is active: $VIRTUAL_ENV"
else
    echo "⚠️  No virtual environment detected. It's recommended to use one."
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Please activate your virtual environment and run again."
        exit 1
    fi
fi

# Install/update dependencies
echo "📦 Installing dependencies..."
echo "First installing minimal requirements..."
if pip install -r requirements-minimal.txt; then
    echo "✓ Minimal dependencies installed successfully"
else
    echo "❌ Failed to install minimal dependencies"
    exit 1
fi

echo "Installing additional optional dependencies..."
echo "(Some may fail, this is normal for optional components)"

# Install optional dependencies individually to avoid failing on problematic ones
pip install opencv-python>=4.5.0 || echo "⚠️  opencv-python failed (optional)"
pip install scikit-learn>=1.0.0 || echo "⚠️  scikit-learn failed (optional)"
pip install torch>=1.10.0 || echo "⚠️  torch failed (optional)"
pip install torchvision>=0.11.0 || echo "⚠️  torchvision failed (optional)"
pip install fastapi>=0.70.0 || echo "⚠️  fastapi failed (optional)"
pip install uvicorn>=0.15.0 || echo "⚠️  uvicorn failed (optional)"
pip install librosa>=0.8.1 || echo "⚠️  librosa failed (optional)"
pip install soundfile>=0.10.0 || echo "⚠️  soundfile failed (optional)"

echo "✓ Dependency installation completed"

# Run the test to verify the fix
echo "🧪 Running verification tests..."
if python3 test_database_fix.py; then
    echo "✅ Fix verification successful!"
else
    echo "❌ Fix verification failed. Please check the output above."
    exit 1
fi

echo ""
echo "🎉 Installation complete!"
echo ""
echo "You can now run StegAnalyzer:"
echo "  python3 steg_main.py image.png --output reports/ --verbose"
echo ""
echo "To check system requirements:"
echo "  python3 steg_main.py --check-system"
echo ""
echo "If you encounter any issues, check the backup files created."