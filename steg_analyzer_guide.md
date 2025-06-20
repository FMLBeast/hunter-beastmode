# StegAnalyzer - Advanced Steganography Detection Framework

## Overview

StegAnalyzer is a comprehensive, GPU-powered, AI-augmented steganography detection and analysis framework. It combines classical steganography tools, advanced machine learning models, cryptographic analysis, and forensic techniques to provide thorough analysis of potentially hidden content in files.

## Features

### Core Capabilities
- **Multi-format Support**: Images (JPEG, PNG, GIF, TIFF), Audio (MP3, WAV), Video (MP4, AVI), Documents (PDF, Office)
- **Classical Steganography Tools**: Steghide, Outguess, Zsteg, Binwalk, Foremost integration
- **AI-Powered Analysis**: CNN-based detection, Noiseprint, Deep learning ensembles
- **Cryptographic Analysis**: Entropy analysis, pattern detection, key search
- **Metadata Forensics**: Comprehensive metadata extraction and analysis
- **Real-time Dashboard**: Live analysis monitoring with web interface
- **Parallel Processing**: Multi-threaded and GPU-accelerated analysis
- **Session Management**: Checkpoint system for resuming long analyses

### Analysis Methods
- **Image Forensics**: LSB analysis, DCT coefficient analysis, noise analysis
- **Audio Analysis**: Spectral analysis, echo hiding detection
- **File Forensics**: Header/footer validation, embedded file extraction
- **ML Detection**: Neural networks, anomaly detection, ensemble methods
- **Cloud Integration**: VirusTotal, hash lookups, threat intelligence

## Installation

### System Requirements
- **OS**: Linux (Ubuntu 20.04+), macOS, Windows 10+
- **Python**: 3.8 or higher
- **Memory**: 8GB RAM recommended (4GB minimum)
- **Storage**: 10GB free space
- **GPU**: NVIDIA GPU with CUDA (optional, for ML acceleration)

### Quick Install (Ubuntu/Debian)

```bash
# Clone the repository
git clone https://github.com/your-org/steganalyzer.git
cd steganalyzer

# Install system dependencies
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv python3-dev build-essential \
    libmagic1 libmagic-dev steghide outguess binwalk foremost exiftool \
    yara clamav strings hexdump

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Optional: Install ML frameworks for GPU acceleration
pip install torch torchvision tensorflow transformers

# Check system compatibility
python steg_main.py --check-system
```

### Docker Installation

```bash
# Pull the Docker image
docker pull steganalyzer/steganalyzer:latest

# Run with GPU support (if available)
docker run --gpus all -v $(pwd)/samples:/data steganalyzer/steganalyzer:latest /data/image.jpg

# Run CPU-only
docker run -v $(pwd)/samples:/data steganalyzer/steganalyzer:latest /data/image.jpg
```

## Configuration

### Basic Configuration

Create `config/default.json`:

```json
{
  "database": {
    "type": "sqlite",
    "path": "steganalyzer.db"
  },
  "orchestrator": {
    "max_concurrent_files": 4,
    "max_cpu_workers": 8,
    "max_gpu_workers": 2,
    "task_timeout": 3600
  },
  "analysis": {
    "quick_mode": false,
    "deep_analysis": true,
    "ml_analysis": true,
    "parallel_analysis": true
  },
  "classic_stego": {
    "steghide_enabled": true,
    "outguess_enabled": true,
    "zsteg_enabled": true,
    "binwalk_enabled": true
  },
  "ml": {
    "gpu_enabled": true,
    "model_cache_size": 5,
    "load_pretrained": true
  },
  "dashboard": {
    "enabled": true,
    "host": "127.0.0.1",
    "port": 8080,
    "auto_open": true
  },
  "report": {
    "format": "html",
    "output_dir": "reports/",
    "auto_open": true
  }
}
```

### Advanced Configuration

For production environments or specialized analysis:

```json
{
  "database": {
    "type": "postgresql",
    "postgres_host": "localhost",
    "postgres_port": 5432,
    "postgres_db": "steganalyzer",
    "postgres_user": "steguser",
    "postgres_password": "secure_password"
  },
  "cloud": {
    "enabled": true,
    "virustotal_api_key": "your_vt_api_key",
    "hashlookup_enabled": true,
    "rate_limit": 1.0
  },
  "llm": {
    "provider": "anthropic",
    "api_key": "your_anthropic_key",
    "model_name": "claude-3-sonnet-20240229"
  }
}
```

## Usage Examples

### Basic File Analysis

```bash
# Analyze a single image
python steg_main.py /path/to/image.jpg

# Analyze with verbose output
python steg_main.py -v /path/to/suspicious_file.pdf

# Quick analysis mode
python steg_main.py --config config/quick.json /path/to/file.png
```

### Batch Analysis

```bash
# Analyze entire directory
python steg_main.py /path/to/directory/

# Analyze specific file types
python steg_main.py --pattern "*.jpg,*.png" /path/to/directory/

# Resume interrupted analysis
python steg_main.py --resume session_id_12345
```

### Advanced Analysis

```bash
# Deep analysis with all tools
python steg_main.py --config config/comprehensive.json /path/to/file.jpg

# Analysis with custom output directory
python steg_main.py -o /custom/output/path /path/to/file.png

# Generate specific report format
python steg_main.py --format pdf /path/to/file.jpg
```

### Session Management

```bash
# List all analysis sessions
python steg_main.py --list-sessions

# Get results for specific session
python steg_main.py --session-results session_id_12345

# Resume incomplete session
python steg_main.py --resume session_id_12345
```

## Dashboard Usage

### Accessing the Dashboard

1. Start analysis with dashboard enabled:
   ```bash
   python steg_main.py /path/to/file.jpg
   ```

2. Open browser to: `http://127.0.0.1:8080`

### Dashboard Features

- **Real-time Progress**: Live updates of analysis progress
- **Finding Visualization**: Interactive charts of confidence levels
- **Live Logs**: Real-time analysis logging
- **File Overview**: Status of all files being processed
- **Tool Performance**: Performance metrics for each analysis tool

### Dashboard Controls

- **Export Results**: Download analysis results as JSON/CSV
- **Session Management**: View and manage analysis sessions
- **Filter Findings**: Filter by confidence level, tool, or type
- **Detailed Views**: Click findings for detailed information

## Analysis Methods

### Image Analysis

#### Classical Methods
- **LSB Analysis**: Detects least significant bit steganography
- **DCT Analysis**: Analyzes discrete cosine transform coefficients
- **Chi-Square Test**: Statistical analysis for hidden data
- **Visual Attack**: Checks for visual artifacts

#### Advanced Methods
- **CNN Detection**: Deep learning steganography detection
- **Noiseprint**: Camera fingerprint and tampering detection
- **Texture Analysis**: GLCM, LBP, and Gabor filter analysis
- **Frequency Domain**: FFT and wavelet analysis

#### Metadata Analysis
- **EXIF Extraction**: Camera settings, GPS, timestamps
- **Thumbnail Analysis**: Hidden thumbnail examination
- **Color Profile**: ICC profile analysis
- **Software Signatures**: Creation tool identification

### Audio Analysis

#### Spectral Methods
- **Spectrogram Analysis**: Visual frequency domain inspection
- **Echo Hiding**: Echo-based steganography detection
- **Phase Coding**: Phase manipulation detection
- **LSB Audio**: Audio LSB steganography detection

#### Metadata Methods
- **ID3 Tags**: MP3 metadata analysis
- **Embedded Files**: Hidden file detection in audio
- **Format Analysis**: Audio format integrity checking

### Document Analysis

#### PDF Analysis
- **Stream Analysis**: PDF object stream examination
- **JavaScript Detection**: Embedded script analysis
- **Form Analysis**: Interactive form inspection
- **Embedded Files**: Attachment and embedded object detection

#### Office Documents
- **XML Metadata**: Document properties analysis
- **Embedded Objects**: OLE object examination
- **Macro Analysis**: VBA script detection
- **Custom Properties**: Non-standard metadata fields

### Cryptographic Analysis

#### Entropy Analysis
- **Shannon Entropy**: Randomness measurement
- **Windowed Entropy**: Localized entropy analysis
- **Entropy Variance**: Entropy distribution analysis

#### Pattern Detection
- **Repeated Sequences**: Pattern repetition analysis
- **Encoding Detection**: Base64, hex, and other encodings
- **Key Material**: Cryptographic key detection
- **Compression Analysis**: Data compression characteristics

## Output and Reporting

### Report Formats

#### HTML Report
- Interactive web-based report
- Charts and visualizations
- Expandable sections
- Export capabilities

#### JSON Report
- Machine-readable format
- Complete analysis data
- API integration friendly
- Structured findings

#### PDF Report
- Professional presentation format
- Executive summary
- Detailed technical findings
- Print-ready layout

#### Markdown Report
- Documentation-friendly format
- Version control compatible
- Plain text readable
- Easy integration

### Finding Types

#### High Confidence Findings
- **Embedded Files**: Confirmed hidden files
- **Tool Signatures**: Known steganography tool usage
- **Metadata Anomalies**: Unusual metadata patterns
- **Statistical Anomalies**: Significant statistical deviations

#### Medium Confidence Findings
- **Pattern Anomalies**: Suspicious patterns detected
- **Entropy Irregularities**: Unusual entropy distributions
- **Format Inconsistencies**: File format violations
- **Timestamp Discrepancies**: Time-based inconsistencies

#### Low Confidence Findings
- **Suspicious Strings**: Potentially relevant text
- **Minor Anomalies**: Small statistical variations
- **Metadata Traces**: User activity traces
- **Software Artifacts**: Tool usage indicators

## Troubleshooting

### Common Issues

#### Installation Problems

**Error**: `steghide: command not found`
```bash
# Ubuntu/Debian
sudo apt-get install steghide

# macOS
brew install steghide

# Check PATH
which steghide
```

**Error**: `CUDA out of memory`
```bash
# Reduce GPU workers in config
"max_gpu_workers": 1

# Or disable GPU
"gpu_enabled": false
```

#### Analysis Issues

**Error**: `Analysis timeout`
```bash
# Increase timeout in config
"task_timeout": 7200  # 2 hours

# Or use quick mode
--config config/quick.json
```

**Error**: `Permission denied`
```bash
# Check file permissions
chmod 644 target_file.jpg

# Run with appropriate user
sudo python steg_main.py file.jpg
```

### Performance Optimization

#### CPU Optimization
```json
{
  "orchestrator": {
    "max_cpu_workers": 16,  // Increase for more CPU cores
    "max_concurrent_files": 8,  // Parallel file processing
    "parallel_analysis": true
  }
}
```

#### GPU Optimization
```json
{
  "ml": {
    "gpu_enabled": true,
    "max_gpu_workers": 2,
    "batch_size": 32,
    "model_cache_size": 10
  }
}
```

#### Memory Optimization
```json
{
  "analysis": {
    "chunk_size": 1048576,  // 1MB chunks
    "max_file_size": 104857600,  // 100MB limit
    "cache_results": false  // Disable for low memory
  }
}
```

### Debugging

#### Verbose Logging
```bash
# Enable debug logging
python steg_main.py -v --log-level DEBUG /path/to/file.jpg

# Log to file
python steg_main.py -v --log-file analysis.log /path/to/file.jpg
```

#### System Check
```bash
# Comprehensive system check
python steg_main.py --check-system

# Generate system report
python steg_main.py --check-system > system_report.json
```

## API Integration

### Python API

```python
from steg_analyzer import StegAnalyzer
import asyncio

async def analyze_file():
    analyzer = StegAnalyzer("config/default.json")
    
    # Analyze single file
    results = await analyzer.analyze_file("suspicious_image.jpg")
    
    # Print findings
    for finding in results['findings']:
        print(f"Type: {finding['type']}")
        print(f"Confidence: {finding['confidence']}")
        print(f"Details: {finding['details']}")
    
    await analyzer.cleanup()

# Run analysis
asyncio.run(analyze_file())
```

### REST API

```python
# Start REST API server
python api_server.py --port 8000

# Submit analysis job
curl -X POST http://localhost:8000/analyze \
  -F "file=@suspicious_image.jpg" \
  -F "config=@config.json"

# Get results
curl http://localhost:8000/results/session_id_12345
```

## Best Practices

### Analysis Strategy

1. **Start with Quick Analysis**: Use quick mode for initial screening
2. **Progressive Analysis**: Increase depth based on initial findings
3. **Multiple Tools**: Use ensemble methods for higher confidence
4. **Context Matters**: Consider file source and expected content
5. **Validation**: Manually verify high-confidence findings

### Performance Tips

1. **Resource Management**: Monitor CPU and memory usage
2. **Batch Processing**: Process similar files together
3. **Tool Selection**: Disable unused tools to improve speed
4. **Caching**: Enable result caching for repeated analysis
5. **Parallel Processing**: Use multiple workers for large datasets

### Security Considerations

1. **Isolation**: Run analysis in sandboxed environment
2. **File Handling**: Validate input files before processing
3. **Temporary Files**: Secure cleanup of temporary data
4. **Network Access**: Control external tool network access
5. **Logging**: Secure handling of analysis logs

## Advanced Topics

### Custom Tool Integration

```python
# Add custom analysis tool
from tools.base_tool import BaseTool

class CustomTool(BaseTool):
    def execute_method(self, method, file_path):
        # Custom analysis logic
        return results

# Register tool
orchestrator.register_tool("custom_tool", CustomTool)
```

### Machine Learning Model Training

```python
# Train custom steganography detection model
from ai.model_trainer import ModelTrainer

trainer = ModelTrainer()
model = trainer.train_cnn_detector(
    clean_images_path="clean_dataset/",
    stego_images_path="stego_dataset/",
    epochs=100
)

# Deploy model
trainer.deploy_model(model, "custom_steg_detector")
```

### Database Customization

```python
# Custom database backend
from core.database import DatabaseInterface

class CustomDatabase(DatabaseInterface):
    async def store_finding(self, session_id, finding):
        # Custom storage logic
        pass

# Use custom database
analyzer = StegAnalyzer(database=CustomDatabase())
```

## Support and Resources

### Documentation
- API Reference: `/docs/api/`
- Tool Documentation: `/docs/tools/`
- Configuration Guide: `/docs/configuration/`
- Examples: `/examples/`

### Community
- GitHub Issues: Report bugs and feature requests
- Discord Server: Real-time community support
- Forum: Technical discussions and tips
- Wiki: Community-maintained documentation

### Professional Support
- Enterprise Support: Commercial support packages
- Training: Professional training programs
- Consulting: Custom implementation services
- Development: Custom tool development

## License and Legal

### License
StegAnalyzer is released under the MIT License. See `LICENSE` file for details.

### Legal Considerations
- **Authorized Use**: Only analyze files you own or have permission to analyze
- **Privacy**: Handle sensitive data according to applicable laws
- **Export Controls**: Be aware of cryptography export restrictions
- **Evidence Handling**: Follow forensic best practices for legal evidence

### Third-Party Tools
This software integrates with various third-party tools. Please review their individual licenses and terms of use.

## Changelog

### Version 1.0.0
- Initial release
- Core steganography detection tools
- Machine learning integration
- Web dashboard
- Comprehensive reporting

### Version 1.1.0 (Planned)
- Enhanced ML models
- Additional file format support
- Performance improvements
- Advanced visualization

---

For the latest updates and detailed documentation, visit: https://github.com/your-org/steganalyzer