# StegAnalyzer 🔍

**Advanced Steganography Detection & Analysis Framework**  
*GPU-Powered, AI-Augmented, Massively Parallel Steganography Analysis Tool*

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GPU Accelerated](https://img.shields.io/badge/GPU-Accelerated-green.svg)](https://pytorch.org/)
[![Real-time Dashboard](https://img.shields.io/badge/Dashboard-Real--time-orange.svg)](http://localhost:8080)

## 🚀 Quick Start

```bash
# Clone and setup
git clone https://github.com/your-org/steganalyzer.git
cd steganalyzer
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Analyze a file
python steg_main.py suspicious_image.jpg

# Access real-time dashboard
open http://localhost:8080
```

## ✨ Features

### 🎯 **Core Detection Capabilities**
- **Classical Steganography**: Steghide, Outguess, Zsteg, Binwalk integration
- **AI-Powered Detection**: CNN, Noiseprint, Deep learning ensembles  
- **Statistical Analysis**: Chi-square, entropy, frequency domain analysis
- **Metadata Forensics**: Comprehensive EXIF, IPTC, XMP analysis
- **Cryptographic Analysis**: Pattern detection, key search, entropy analysis

### 🖼️ **Multi-Format Support**
- **Images**: JPEG, PNG, GIF, TIFF, BMP, WebP
- **Audio**: MP3, WAV, FLAC, OGG
- **Video**: MP4, AVI, MOV, MKV
- **Documents**: PDF, DOC/DOCX, PPT/PPTX, XLS/XLSX
- **Archives**: ZIP, RAR, 7Z, TAR

### ⚡ **Performance & Scalability**
- **GPU Acceleration**: CUDA-enabled ML models
- **Parallel Processing**: Multi-threaded analysis pipeline
- **Session Management**: Resume interrupted analyses
- **Batch Processing**: Analyze entire directories
- **Real-time Monitoring**: Live dashboard with progress tracking

### 🧠 **AI & Machine Learning**
- **Neural Networks**: CNN-based steganography detection
- **Ensemble Methods**: Multiple AI models voting
- **Anomaly Detection**: Statistical outlier identification
- **Transfer Learning**: Pre-trained model adaptation
- **Custom Models**: Train domain-specific detectors

## 📋 System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **OS** | Linux, macOS, Windows 10+ | Ubuntu 20.04+ |
| **Python** | 3.8+ | 3.10+ |
| **RAM** | 4GB | 16GB+ |
| **Storage** | 10GB | 50GB+ |
| **GPU** | None (CPU-only) | NVIDIA RTX series |
| **CUDA** | N/A | 11.0+ |

## 🛠️ Installation

### Method 1: Quick Install (Ubuntu/Debian)

```bash
# Install system dependencies
sudo apt-get update && sudo apt-get install -y \
    python3-pip python3-venv python3-dev build-essential \
    libmagic1 libmagic-dev steghide outguess binwalk foremost \
    exiftool yara clamav strings hexdump

# Setup Python environment
git clone https://github.com/your-org/steganalyzer.git
cd steganalyzer
python3 -m venv venv
source venv/bin/activate

# Install Python packages
pip install --upgrade pip
pip install -r requirements.txt

# Optional: GPU support
pip install torch torchvision tensorflow

# Verify installation
python steg_main.py --check-system
```

### Method 2: Docker

```bash
# CPU-only version
docker run -v $(pwd)/samples:/data steganalyzer:latest /data/image.jpg

# GPU-accelerated version
docker run --gpus all -v $(pwd)/samples:/data steganalyzer:gpu /data/image.jpg
```

### Method 3: Conda

```bash
conda env create -f environment.yml
conda activate steganalyzer
python steg_main.py --check-system
```

## 🎮 Usage Examples

### Single File Analysis
```bash
# Basic analysis
python steg_main.py image.jpg

# Verbose output with all tools
python steg_main.py -v --all-tools suspicious_file.pdf

# Quick scan mode
python steg_main.py --quick image.png
```

### Batch Analysis
```bash
# Analyze directory
python steg_main.py /path/to/images/

# Specific file types
python steg_main.py --pattern "*.jpg,*.png" /path/to/mixed_files/

# Recursive analysis
python steg_main.py -r /path/to/nested_dirs/
```

### Advanced Options
```bash
# Custom configuration
python steg_main.py --config custom_config.json file.jpg

# Specific output format
python steg_main.py --format pdf --output reports/ file.jpg

# Resume interrupted session
python steg_main.py --resume session_abc123
```

### Session Management
```bash
# List all sessions
python steg_main.py --list-sessions

# Get session results
python steg_main.py --session-results session_abc123

# Export session data
python steg_main.py --export session_abc123 --format json
```

## 📊 Real-time Dashboard

The web dashboard provides live monitoring of analysis progress:

![Dashboard Screenshot](docs/images/dashboard.png)

### Features:
- **Live Progress**: Real-time analysis status
- **Interactive Charts**: Confidence distribution, tool performance
- **Finding Browser**: Detailed finding inspection
- **Export Options**: JSON, CSV, PDF reports
- **Session History**: Previous analysis sessions

Access at: `http://localhost:8080` (auto-opens during analysis)

## 🔧 Configuration

### Basic Configuration (`config/default.json`)

```json
{
  "analysis": {
    "quick_mode": false,
    "parallel_analysis": true,
    "ml_analysis": true,
    "max_file_size_mb": 100
  },
  "ml": {
    "gpu_enabled": true,
    "confidence_threshold": 0.7,
    "ensemble_voting": true
  },
  "dashboard": {
    "enabled": true,
    "port": 8080,
    "auto_open": true
  },
  "tools": {
    "steghide": true,
    "binwalk": true,
    "exiftool": true,
    "yara": true
  }
}
```

### Tool-Specific Settings

```json
{
  "classic_stego": {
    "steghide_wordlist": "wordlists/common.txt",
    "outguess_quality": 75,
    "zsteg_all_bits": true
  },
  "crypto_analysis": {
    "entropy_window_size": 8192,
    "pattern_min_length": 4,
    "chi_square_threshold": 0.01
  },
  "image_forensics": {
    "jpeg_quality_analysis": true,
    "noise_analysis": true,
    "copy_move_detection": true
  }
}
```

## 🧪 Analysis Methods

### Classical Steganography Detection
- **LSB Analysis**: Least Significant Bit manipulation detection
- **DCT Coefficient**: JPEG coefficient analysis
- **Palette Analysis**: Color palette manipulation detection
- **Tool Signatures**: Known steganography tool detection

### Machine Learning Detection
- **CNN Classification**: Deep learning steganography detection
- **Ensemble Methods**: Multiple model consensus
- **Anomaly Detection**: Statistical outlier identification
- **Feature Engineering**: Hand-crafted statistical features

### Forensic Analysis
- **Metadata Extraction**: Comprehensive metadata analysis
- **File Carving**: Embedded file detection
- **Signature Analysis**: File format validation
- **Timeline Analysis**: Temporal inconsistency detection

### Cryptographic Analysis
- **Entropy Analysis**: Randomness measurement
- **Pattern Detection**: Repeated sequence identification
- **Key Material Search**: Cryptographic key detection
- **Encoding Detection**: Base64, hex, binary encodings

## 📈 Output & Reporting

### Finding Types

| Confidence | Type | Description |
|------------|------|-------------|
| **🔴 High (0.7-1.0)** | Confirmed steganography | Tool signatures, embedded files |
| **🟡 Medium (0.4-0.7)** | Suspicious patterns | Statistical anomalies, metadata issues |
| **🟢 Low (0.0-0.4)** | Potential indicators | Minor inconsistencies, traces |

### Report Formats

- **HTML**: Interactive web report with charts
- **JSON**: Machine-readable structured data
- **PDF**: Professional presentation format
- **Markdown**: Documentation-friendly format
- **CSV**: Tabular data for analysis

### Sample Output

```json
{
  "session_id": "session_2024_01_15_143022",
  "file_path": "suspicious_image.jpg",
  "findings": [
    {
      "type": "steganography_detected",
      "method": "steghide_extraction",
      "confidence": 0.95,
      "details": "Hidden file extracted: secret.txt (1.2KB)",
      "tool": "steghide",
      "evidence": {
        "extracted_file": "secret.txt",
        "extraction_method": "password_cracking",
        "password_found": "password123"
      }
    },
    {
      "type": "metadata_anomaly",
      "method": "exif_analysis", 
      "confidence": 0.72,
      "details": "GPS coordinates inconsistent with timestamp",
      "location": {"lat": 40.7128, "lon": -74.0060},
      "timestamp": "2024:01:15 14:30:22"
    }
  ],
  "statistics": {
    "total_tools_run": 15,
    "analysis_time": 42.3,
    "file_size": 2048576,
    "entropy": 7.23
  }
}
```

## 🔌 API Integration

### Python API

```python
import asyncio
from steg_analyzer import StegAnalyzer

async def analyze():
    analyzer = StegAnalyzer("config/production.json")
    
    # Single file
    results = await analyzer.analyze_file("image.jpg")
    
    # Batch processing
    batch_results = await analyzer.analyze_directory("samples/")
    
    # Get detailed findings
    for finding in results['findings']:
        if finding['confidence'] > 0.7:
            print(f"High confidence finding: {finding['type']}")
    
    await analyzer.cleanup()

asyncio.run(analyze())
```

### REST API

```bash
# Start API server
python api_server.py --port 8000

# Submit analysis
curl -X POST http://localhost:8000/analyze \
  -F "file=@image.jpg" \
  -F "config=@config.json"

# Check status
curl http://localhost:8000/status/session_123

# Get results
curl http://localhost:8000/results/session_123
```

## 🧩 Extending StegAnalyzer

### Adding Custom Tools

```python
from tools.base_tool import BaseTool

class CustomDetector(BaseTool):
    def execute_method(self, method: str, file_path: Path):
        # Your detection logic here
        results = []
        
        if method == "custom_analysis":
            # Implement custom analysis
            confidence = self.analyze_file(file_path)
            results.append({
                "type": "custom_detection",
                "confidence": confidence,
                "details": "Custom analysis result"
            })
        
        return results

# Register tool
orchestrator.register_tool("custom_detector", CustomDetector)
```

### Custom ML Models

```python
from ai.ml_detector import MLStegDetector

class CustomMLDetector(MLStegDetector):
    def load_custom_model(self, model_path):
        # Load your trained model
        self.custom_model = torch.load(model_path)
    
    def custom_prediction(self, image):
        # Run custom inference
        with torch.no_grad():
            prediction = self.custom_model(image)
        return prediction.item()
```

## 📚 Documentation

- **[User Guide](docs/user_guide.md)**: Comprehensive usage documentation
- **[API Reference](docs/api/README.md)**: Complete API documentation  
- **[Configuration Guide](docs/configuration.md)**: Detailed configuration options
- **[Tool Documentation](docs/tools/README.md)**: Individual tool documentation
- **[Development Guide](docs/development.md)**: Contributing and development setup

## 🐛 Troubleshooting

### Common Issues

**Issue**: `steghide: command not found`
```bash
# Ubuntu/Debian
sudo apt-get install steghide

# macOS  
brew install steghide
```

**Issue**: `CUDA out of memory`
```bash
# Reduce GPU workers
export CUDA_VISIBLE_DEVICES=0
# Or disable GPU in config
"gpu_enabled": false
```

**Issue**: Analysis hangs or times out
```bash
# Increase timeout
python steg_main.py --timeout 3600 file.jpg

# Use quick mode
python steg_main.py --quick file.jpg
```

### Getting Help

1. **Check System**: `python steg_main.py --check-system`
2. **Enable Debug**: `python steg_main.py -v --debug file.jpg`
3. **Review Logs**: Check `logs/steganalyzer.log`
4. **GitHub Issues**: Report bugs and request features
5. **Community Discord**: Real-time help and discussion

## 📊 Performance Benchmarks

| File Type | Size | Analysis Time | Tools Used | Findings |
|-----------|------|---------------|------------|----------|
| JPEG Image | 2MB | 15.3s | 12 | 3 findings |
| PNG Image | 5MB | 28.7s | 14 | 1 finding |
| PDF Document | 10MB | 45.2s | 8 | 5 findings |
| WAV Audio | 50MB | 62.1s | 6 | 2 findings |

*Benchmarks on Intel i7-10700K, 32GB RAM, RTX 3080*

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md).

### Development Setup

```bash
# Clone repository
git clone https://github.com/your-org/steganalyzer.git
cd steganalyzer

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/

# Code formatting
black steg_analyzer/
isort steg_analyzer/
```

### Areas for Contribution

- **New Detection Methods**: Implement novel steganography detection algorithms
- **Tool Integrations**: Add support for additional steganography tools
- **ML Models**: Contribute trained models or improve existing ones
- **File Format Support**: Add support for new file formats
- **Performance Optimization**: Improve analysis speed and efficiency
- **Documentation**: Improve documentation and examples

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Steganography Research Community**: For foundational research and techniques
- **Open Source Tools**: Steghide, Outguess, Binwalk, Foremost, ExifTool, and others
- **ML Frameworks**: PyTorch, TensorFlow, Scikit-learn
- **Contributors**: All the amazing people who have contributed to this project

## 📞 Support

- **GitHub Issues**: Bug reports and feature requests
- **Discord Server**: `discord.gg/steganalyzer`
- **Email**: support@steganalyzer.org
- **Wiki**: Community documentation and examples

## 🗺️ Roadmap

### Version 1.1 (Q2 2024)
- [ ] Enhanced ML models with transformer architectures
- [ ] Real-time streaming analysis
- [ ] Advanced visualization and heat maps
- [ ] Plugin system for external tools

### Version 1.2 (Q3 2024)
- [ ] Cloud-native deployment options
- [ ] Advanced threat intelligence integration
- [ ] Multi-language support
- [ ] Mobile app companion

### Version 2.0 (Q4 2024)
- [ ] Distributed analysis across multiple nodes
- [ ] Advanced AI-powered correlation engine
- [ ] Blockchain-based evidence integrity
- [ ] Quantum-resistant analysis methods

---

**⭐ Star this repository if you find it useful!**

**🔗 Share with the cybersecurity community**

Built with ❤️ by the StegAnalyzer team