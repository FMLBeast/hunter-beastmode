#!/usr/bin/env python3
"""
Comprehensive Tool Installation Checker for StegAnalyzer
Verifies all required tools, libraries, and dependencies for vast.ai deployment
"""

import subprocess
import sys
import importlib
import platform
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import os

class ToolChecker:
    """Comprehensive tool and dependency checker"""
    
    def __init__(self):
        self.results = {
            "system_tools": {},
            "python_packages": {},
            "gpu_support": {},
            "optional_tools": {},
            "configuration": {},
            "performance": {}
        }
        self.errors = []
        self.warnings = []
        self.missing_tools = []
        
    def run_full_check(self) -> Dict:
        """Run complete system check"""
        print("🔍 StegAnalyzer Tool Installation Checker")
        print("=" * 60)
        
        # Check system information
        self.check_system_info()
        
        # Check core system tools
        self.check_system_tools()
        
        # Check Python packages
        self.check_python_packages()
        
        # Check GPU support
        self.check_gpu_support()
        
        # Check optional tools
        self.check_optional_tools()
        
        # Check configuration
        self.check_configuration()
        
        # Check performance indicators
        self.check_performance()
        
        # Generate report
        self.generate_report()
        
        return self.results
    
    def check_system_info(self):
        """Check basic system information"""
        print("\n📋 System Information")
        print("-" * 30)
        
        info = {
            "platform": platform.system(),
            "architecture": platform.machine(),
            "python_version": platform.python_version(),
            "cpu_count": os.cpu_count()
        }
        
        self.results["system_info"] = info
        
        for key, value in info.items():
            print(f"   {key}: {value}")
    
    def check_system_tools(self):
        """Check core system tools required for steganography analysis"""
        print("\n🛠️  Core System Tools")
        print("-" * 30)
        
        core_tools = {
            # Steganography tools
            "steghide": {
                "command": ["steghide", "--version"],
                "required": True,
                "description": "Hide/extract data in image files"
            },
            "outguess": {
                "command": ["outguess", "-h"],
                "required": True,
                "description": "JPEG steganography tool"
            },
            "zsteg": {
                "command": ["zsteg", "--version"],
                "required": True,
                "description": "PNG/BMP steganography detection"
            },
            "binwalk": {
                "command": ["binwalk", "--version"],
                "required": True,
                "description": "Firmware analysis and file extraction"
            },
            "foremost": {
                "command": ["foremost", "-V"],
                "required": True,
                "description": "File carving tool"
            },
            "exiftool": {
                "command": ["exiftool", "-ver"],
                "required": True,
                "description": "Metadata extraction tool"
            },
            
            # System utilities
            "file": {
                "command": ["file", "--version"],
                "required": True,
                "description": "File type identification"
            },
            "strings": {
                "command": ["strings", "--version"],
                "required": True,
                "description": "Extract text strings from files"
            },
            "hexdump": {
                "command": ["hexdump", "-C", "/dev/null"],
                "required": True,
                "description": "Hex dump utility"
            },
            "ffmpeg": {
                "command": ["ffmpeg", "-version"],
                "required": False,
                "description": "Audio/video processing"
            },
            "tesseract": {
                "command": ["tesseract", "--version"],
                "required": False,
                "description": "OCR engine"
            }
        }
        
        for tool, config in core_tools.items():
            status = self._check_tool_availability(tool, config["command"])
            self.results["system_tools"][tool] = {
                "available": status["available"],
                "version": status["version"],
                "required": config["required"],
                "description": config["description"]
            }
            
            status_icon = "✅" if status["available"] else "❌"
            req_text = "[REQUIRED]" if config["required"] else "[OPTIONAL]"
            
            print(f"   {status_icon} {tool:12} {req_text:12} - {config['description']}")
            if status["version"]:
                print(f"      Version: {status['version']}")
            
            if not status["available"] and config["required"]:
                self.missing_tools.append(tool)
                self.errors.append(f"Missing required tool: {tool}")
    
    def check_python_packages(self):
        """Check Python package dependencies"""
        print("\n🐍 Python Packages")
        print("-" * 30)
        
        packages = {
            # Core dependencies
            "numpy": {"required": True, "description": "Numerical computing"},
            "opencv-python": {"required": True, "description": "Computer vision library"},
            "Pillow": {"required": True, "description": "Image processing"},
            "matplotlib": {"required": True, "description": "Plotting and visualization"},
            "scipy": {"required": True, "description": "Scientific computing"},
            "scikit-learn": {"required": True, "description": "Machine learning"},
            "librosa": {"required": True, "description": "Audio analysis"},
            
            # Database
            "sqlite3": {"required": True, "description": "SQLite database"},
            "asyncpg": {"required": False, "description": "PostgreSQL async driver"},
            
            # Web and API
            "fastapi": {"required": True, "description": "Web API framework"},
            "uvicorn": {"required": True, "description": "ASGI server"},
            "requests": {"required": True, "description": "HTTP client"},
            "aiohttp": {"required": True, "description": "Async HTTP client"},
            
            # ML and AI
            "torch": {"required": False, "description": "PyTorch deep learning"},
            "torchvision": {"required": False, "description": "PyTorch computer vision"},
            "tensorflow": {"required": False, "description": "TensorFlow deep learning"},
            "transformers": {"required": False, "description": "Transformer models"},
            "anthropic": {"required": False, "description": "Anthropic API client"},
            
            # Utilities
            "python-magic": {"required": True, "description": "File type detection"},
            "cryptography": {"required": True, "description": "Cryptographic utilities"},
            "tqdm": {"required": True, "description": "Progress bars"},
            "click": {"required": True, "description": "CLI framework"},
            "pydantic": {"required": True, "description": "Data validation"},
            "jinja2": {"required": True, "description": "Template engine"},
            
            # Optional performance
            "numba": {"required": False, "description": "JIT compilation"},
            "cython": {"required": False, "description": "C extensions"}
        }
        
        for package, config in packages.items():
            status = self._check_python_package(package)
            self.results["python_packages"][package] = {
                "available": status["available"],
                "version": status["version"],
                "required": config["required"],
                "description": config["description"]
            }
            
            status_icon = "✅" if status["available"] else "❌"
            req_text = "[REQUIRED]" if config["required"] else "[OPTIONAL]"
            
            print(f"   {status_icon} {package:20} {req_text:12} - {config['description']}")
            if status["version"]:
                print(f"      Version: {status['version']}")
            
            if not status["available"] and config["required"]:
                self.missing_tools.append(f"python:{package}")
                self.errors.append(f"Missing required Python package: {package}")
    
    def check_gpu_support(self):
        """Check GPU support and CUDA availability"""
        print("\n🎮 GPU Support")
        print("-" * 30)
        
        gpu_info = {
            "nvidia_smi": self._check_nvidia_smi(),
            "cuda_available": self._check_cuda(),
            "torch_cuda": self._check_torch_cuda(),
            "tensorflow_gpu": self._check_tensorflow_gpu()
        }
        
        self.results["gpu_support"] = gpu_info
        
        for component, available in gpu_info.items():
            status_icon = "✅" if available else "❌"
            print(f"   {status_icon} {component}")
    
    def check_optional_tools(self):
        """Check optional tools that enhance functionality"""
        print("\n🔧 Optional Enhancement Tools")
        print("-" * 30)
        
        optional_tools = {
            "yara": {
                "command": ["yara", "--version"],
                "description": "Pattern matching engine"
            },
            "clamav": {
                "command": ["clamscan", "--version"],
                "description": "Antivirus scanner"
            },
            "volatility": {
                "command": ["volatility", "--help"],
                "description": "Memory forensics"
            },
            "john": {
                "command": ["john", "--help"],
                "description": "Password cracking"
            },
            "hashcat": {
                "command": ["hashcat", "--version"],
                "description": "Password recovery"
            },
            "sqlmap": {
                "command": ["sqlmap", "--version"],
                "description": "SQL injection testing"
            }
        }
        
        for tool, config in optional_tools.items():
            status = self._check_tool_availability(tool, config["command"])
            self.results["optional_tools"][tool] = {
                "available": status["available"],
                "version": status["version"],
                "description": config["description"]
            }
            
            status_icon = "✅" if status["available"] else "❌"
            print(f"   {status_icon} {tool:15} - {config['description']}")
    
    def check_configuration(self):
        """Check configuration files and directories"""
        print("\n⚙️  Configuration")
        print("-" * 30)
        
        config_items = {
            "config_dir": Path("config").exists(),
            "models_dir": Path("models").exists(),
            "logs_dir": Path("logs").exists(),
            "reports_dir": Path("reports").exists(),
            "temp_dir": Path("temp").exists(),
            "static_dir": Path("static").exists(),
            "templates_dir": Path("templates").exists(),
            "wordlists_dir": Path("wordlists").exists(),
            "main_config": Path("config/default.json").exists(),
            "database_exists": Path("steganalyzer.db").exists()
        }
        
        self.results["configuration"] = config_items
        
        for item, exists in config_items.items():
            status_icon = "✅" if exists else "❌"
            print(f"   {status_icon} {item}")
            
            if not exists and item in ["config_dir", "main_config"]:
                self.warnings.append(f"Missing configuration: {item}")
    
    def check_performance(self):
        """Check performance-related settings"""
        print("\n⚡ Performance Indicators")
        print("-" * 30)
        
        # Check available memory
        try:
            with open('/proc/meminfo', 'r') as f:
                meminfo = f.read()
                total_mem = int([line for line in meminfo.split('\n') if 'MemTotal' in line][0].split()[1]) // 1024
        except:
            total_mem = "Unknown"
        
        # Check disk space
        try:
            disk_usage = shutil.disk_usage('.')
            free_space_gb = disk_usage.free // (1024**3)
        except:
            free_space_gb = "Unknown"
        
        performance_info = {
            "total_memory_mb": total_mem,
            "free_disk_space_gb": free_space_gb,
            "cpu_count": os.cpu_count(),
            "python_version": platform.python_version()
        }
        
        self.results["performance"] = performance_info
        
        for metric, value in performance_info.items():
            print(f"   {metric}: {value}")
        
        # Performance warnings
        if isinstance(total_mem, int) and total_mem < 4096:
            self.warnings.append("Low memory: Less than 4GB RAM available")
        
        if isinstance(free_space_gb, int) and free_space_gb < 10:
            self.warnings.append("Low disk space: Less than 10GB free")
    
    def _check_tool_availability(self, tool: str, command: List[str]) -> Dict:
        """Check if a system tool is available"""
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, timeout=10
            )
            
            # Extract version information
            version = "Available"
            output = result.stdout + result.stderr
            
            # Common version patterns
            version_patterns = [
                r'version\s+(\d+\.\d+(?:\.\d+)?)',
                r'v(\d+\.\d+(?:\.\d+)?)',
                r'(\d+\.\d+(?:\.\d+)?)',
                r'Version:\s*(\d+\.\d+(?:\.\d+)?)'
            ]
            
            import re
            for pattern in version_patterns:
                match = re.search(pattern, output, re.IGNORECASE)
                if match:
                    version = match.group(1)
                    break
            
            return {"available": True, "version": version}
            
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
            return {"available": False, "version": None}
    
    def _check_python_package(self, package: str) -> Dict:
        """Check if a Python package is available"""
        try:
            module = importlib.import_module(package.replace('-', '_'))
            version = getattr(module, '__version__', 'Unknown')
            return {"available": True, "version": version}
        except ImportError:
            return {"available": False, "version": None}
    
    def _check_nvidia_smi(self) -> bool:
        """Check if nvidia-smi is available"""
        try:
            subprocess.run(['nvidia-smi'], capture_output=True, timeout=10)
            return True
        except:
            return False
    
    def _check_cuda(self) -> bool:
        """Check CUDA availability"""
        try:
            subprocess.run(['nvcc', '--version'], capture_output=True, timeout=10)
            return True
        except:
            return False
    
    def _check_torch_cuda(self) -> bool:
        """Check PyTorch CUDA support"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def _check_tensorflow_gpu(self) -> bool:
        """Check TensorFlow GPU support"""
        try:
            import tensorflow as tf
            return len(tf.config.list_physical_devices('GPU')) > 0
        except:
            return False
    
    def generate_report(self):
        """Generate comprehensive report"""
        print("\n" + "=" * 60)
        print("📊 INSTALLATION REPORT")
        print("=" * 60)
        
        # Summary statistics
        total_system_tools = len(self.results["system_tools"])
        available_system_tools = sum(1 for tool in self.results["system_tools"].values() if tool["available"])
        
        total_python_packages = len(self.results["python_packages"])
        available_python_packages = sum(1 for pkg in self.results["python_packages"].values() if pkg["available"])
        
        print(f"\n📈 Summary:")
        print(f"   System Tools: {available_system_tools}/{total_system_tools} available")
        print(f"   Python Packages: {available_python_packages}/{total_python_packages} available")
        print(f"   GPU Support: {'✅ Yes' if any(self.results['gpu_support'].values()) else '❌ No'}")
        
        # Critical issues
        if self.errors:
            print(f"\n❌ Critical Issues ({len(self.errors)}):")
            for error in self.errors:
                print(f"   • {error}")
        
        # Warnings
        if self.warnings:
            print(f"\n⚠️  Warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"   • {warning}")
        
        # Installation suggestions
        if self.missing_tools:
            print(f"\n🔧 Installation Suggestions:")
            self._generate_installation_commands()
        
        # Vast.ai readiness
        self._check_vast_ai_readiness()
        
        # Save detailed report
        self._save_detailed_report()
    
    def _generate_installation_commands(self):
        """Generate installation commands for missing tools"""
        system = platform.system().lower()
        
        if system == "linux":
            print("\n   Ubuntu/Debian commands:")
            
            apt_tools = {
                "steghide": "steghide",
                "outguess": "outguess", 
                "binwalk": "binwalk",
                "foremost": "foremost",
                "exiftool": "exiftool",
                "file": "file",
                "strings": "binutils",
                "hexdump": "bsdutils",
                "ffmpeg": "ffmpeg",
                "tesseract": "tesseract-ocr"
            }
            
            missing_apt = [apt_tools[tool] for tool in self.missing_tools if tool in apt_tools]
            if missing_apt:
                print(f"   sudo apt update && sudo apt install -y {' '.join(missing_apt)}")
            
            # zsteg needs gem
            if "zsteg" in self.missing_tools:
                print("   sudo gem install zsteg")
            
            # Python packages
            missing_python = [tool.replace('python:', '') for tool in self.missing_tools if tool.startswith('python:')]
            if missing_python:
                print(f"   pip install {' '.join(missing_python)}")
        
        elif system == "darwin":  # macOS
            print("\n   macOS (Homebrew) commands:")
            print("   brew install steghide outguess binwalk foremost exiftool")
            print("   gem install zsteg")
    
    def _check_vast_ai_readiness(self):
        """Check readiness for vast.ai deployment"""
        print(f"\n🚀 Vast.ai Deployment Readiness:")
        
        readiness_score = 0
        max_score = 10
        
        # Core tools (40% weight)
        core_tools = ["steghide", "outguess", "zsteg", "binwalk", "foremost", "exiftool"]
        available_core = sum(1 for tool in core_tools if self.results["system_tools"].get(tool, {}).get("available", False))
        readiness_score += (available_core / len(core_tools)) * 4
        
        # Python packages (30% weight)
        required_packages = [pkg for pkg, info in self.results["python_packages"].items() if info["required"]]
        available_packages = sum(1 for pkg in required_packages if self.results["python_packages"][pkg]["available"])
        readiness_score += (available_packages / len(required_packages)) * 3
        
        # GPU support (20% weight)
        if any(self.results["gpu_support"].values()):
            readiness_score += 2
        
        # Configuration (10% weight)
        if self.results["configuration"]["config_dir"]:
            readiness_score += 1
        
        readiness_percentage = (readiness_score / max_score) * 100
        
        if readiness_percentage >= 90:
            status = "🟢 READY"
        elif readiness_percentage >= 70:
            status = "🟡 MOSTLY READY"
        else:
            status = "🔴 NOT READY"
        
        print(f"   {status} ({readiness_percentage:.1f}%)")
        print(f"   Recommendation: {'Deploy immediately' if readiness_percentage >= 90 else 'Install missing components first'}")
    
    def _save_detailed_report(self):
        """Save detailed JSON report"""
        report_path = Path("tool_check_report.json")
        
        report_data = {
            "timestamp": subprocess.run(['date'], capture_output=True, text=True).stdout.strip(),
            "results": self.results,
            "errors": self.errors,
            "warnings": self.warnings,
            "missing_tools": self.missing_tools
        }
        
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n💾 Detailed report saved: {report_path}")

def main():
    """Main function"""
    checker = ToolChecker()
    results = checker.run_full_check()
    
    # Return appropriate exit code
    if checker.errors:
        print(f"\n❌ Found {len(checker.errors)} critical issues. Please resolve before deployment.")
        sys.exit(1)
    elif checker.warnings:
        print(f"\n⚠️  Found {len(checker.warnings)} warnings. Consider addressing for optimal performance.")
        sys.exit(0)
    else:
        print(f"\n✅ All systems ready for deployment!")
        sys.exit(0)

if __name__ == "__main__":
    main()
