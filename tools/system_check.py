"""
System Requirements Checker - Validates system dependencies and capabilities
"""

import subprocess
import sys
import logging
import platform
import shutil
import importlib
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json
import time
import psutil

class SystemChecker:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.system_info = {}
        self.dependency_status = {}
        self.recommendations = []
        
        # Required and optional dependencies
        self.python_dependencies = {
            "required": [
                "numpy", "Pillow", "opencv-python", "matplotlib", 
                "scipy", "scikit-learn", "requests", "aiohttp",
                "asyncio", "pathlib", "hashlib", "magic", "psutil"
            ],
            "optional": [
                "torch", "torchvision", "tensorflow", "transformers",
                "pynvml", "nvidia-ml-py3", "anthropic", "openai",
                "weasyprint", "jinja2", "networkx", "neo4j",
                "asyncpg", "pymongo", "redis", "ssdeep"
            ],
            "crypto": [
                "cryptography", "pycryptodome", "hashlib", "hmac"
            ],
            "image": [
                "Pillow", "opencv-python", "imageio", "skimage"
            ],
            "audio": [
                "librosa", "soundfile", "wave", "scipy"
            ],
            "ml": [
                "torch", "torchvision", "tensorflow", "scikit-learn",
                "transformers", "huggingface-hub"
            ]
        }
        
        self.system_tools = {
            "required": [
                "python3", "pip"
            ],
            "steganography": [
                "steghide", "outguess", "zsteg", "stegseek", 
                "stegoveritas", "stegcracker"
            ],
            "forensics": [
                "binwalk", "foremost", "exiftool", "strings",
                "file", "hexdump", "xxd"
            ],
            "crypto": [
                "openssl", "hashcat", "john"
            ],
            "analysis": [
                "yara", "clamav", "volatility", "bulk_extractor"
            ],
            "optional": [
                "tesseract", "ffmpeg", "sox", "imagemagick",
                "ghostscript", "poppler-utils"
            ]
        }
        
        # Minimum system requirements
        self.min_requirements = {
            "python_version": (3, 8),
            "memory_gb": 4,
            "disk_space_gb": 10,
            "cpu_cores": 2
        }
    
    async def check_all(self) -> Dict[str, Any]:
        """Perform comprehensive system check"""
        start_time = time.time()
        
        try:
            self.logger.info("Starting comprehensive system check...")
            
            # System information
            self.system_info = self._get_system_info()
            
            # Check system requirements
            system_req_status = self._check_system_requirements()
            
            # Check Python environment
            python_status = self._check_python_environment()
            
            # Check Python dependencies
            python_deps_status = await self._check_python_dependencies()
            
            # Check system tools
            tools_status = await self._check_system_tools()
            
            # Check GPU availability
            gpu_status = self._check_gpu_availability()
            
            # Check database connectivity
            db_status = await self._check_database_connectivity()
            
            # Check network connectivity
            network_status = await self._check_network_connectivity()
            
            # Generate recommendations
            self._generate_recommendations()
            
            # Calculate overall status
            overall_status = self._calculate_overall_status([
                system_req_status, python_status, python_deps_status,
                tools_status, gpu_status, db_status, network_status
            ])
            
            check_time = time.time() - start_time
            
            result = {
                "timestamp": time.time(),
                "check_duration": check_time,
                "overall_status": overall_status,
                "system_info": self.system_info,
                "checks": {
                    "system_requirements": system_req_status,
                    "python_environment": python_status,
                    "python_dependencies": python_deps_status,
                    "system_tools": tools_status,
                    "gpu_availability": gpu_status,
                    "database_connectivity": db_status,
                    "network_connectivity": network_status
                },
                "recommendations": self.recommendations,
                "capabilities": self._assess_capabilities()
            }
            
            self.logger.info(f"System check completed in {check_time:.2f}s - Status: {overall_status}")
            return result
            
        except Exception as e:
            self.logger.error(f"System check failed: {e}")
            return {
                "timestamp": time.time(),
                "overall_status": "ERROR",
                "error": str(e),
                "recommendations": ["Fix system check errors before proceeding"]
            }
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Collect basic system information"""
        try:
            info = {
                "platform": platform.platform(),
                "system": platform.system(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "python_version": platform.python_version(),
                "python_implementation": platform.python_implementation(),
                "architecture": platform.architecture(),
                "hostname": platform.node(),
                "cpu_count": psutil.cpu_count(),
                "memory_total": psutil.virtual_memory().total,
                "memory_available": psutil.virtual_memory().available,
                "disk_usage": {}
            }
            
            # Disk usage for current directory
            try:
                current_disk = psutil.disk_usage('.')
                info["disk_usage"] = {
                    "total": current_disk.total,
                    "used": current_disk.used,
                    "free": current_disk.free
                }
            except Exception as e:
                info["disk_usage"] = {"error": str(e)}
            
            # Python executable path
            info["python_executable"] = sys.executable
            
            # Environment variables of interest
            env_vars = ["PATH", "PYTHONPATH", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS"]
            info["environment"] = {var: os.environ.get(var) for var in env_vars}
            
            return info
            
        except Exception as e:
            self.logger.error(f"Failed to get system info: {e}")
            return {"error": str(e)}
    
    def _check_system_requirements(self) -> Dict[str, Any]:
        """Check minimum system requirements"""
        status = {
            "status": "PASS",
            "checks": {},
            "issues": []
        }
        
        try:
            # Python version
            current_version = sys.version_info[:2]
            min_version = self.min_requirements["python_version"]
            
            if current_version >= min_version:
                status["checks"]["python_version"] = {
                    "status": "PASS",
                    "current": f"{current_version[0]}.{current_version[1]}",
                    "required": f"{min_version[0]}.{min_version[1]}"
                }
            else:
                status["checks"]["python_version"] = {
                    "status": "FAIL",
                    "current": f"{current_version[0]}.{current_version[1]}",
                    "required": f"{min_version[0]}.{min_version[1]}"
                }
                status["issues"].append(f"Python {min_version[0]}.{min_version[1]}+ required")
                status["status"] = "FAIL"
            
            # Memory
            memory_gb = psutil.virtual_memory().total / (1024**3)
            min_memory = self.min_requirements["memory_gb"]
            
            if memory_gb >= min_memory:
                status["checks"]["memory"] = {
                    "status": "PASS",
                    "current_gb": round(memory_gb, 1),
                    "required_gb": min_memory
                }
            else:
                status["checks"]["memory"] = {
                    "status": "WARN",
                    "current_gb": round(memory_gb, 1),
                    "required_gb": min_memory
                }
                status["issues"].append(f"Low memory: {memory_gb:.1f}GB (recommended: {min_memory}GB+)")
                if status["status"] == "PASS":
                    status["status"] = "WARN"
            
            # Disk space
            disk_free_gb = psutil.disk_usage('.').free / (1024**3)
            min_disk = self.min_requirements["disk_space_gb"]
            
            if disk_free_gb >= min_disk:
                status["checks"]["disk_space"] = {
                    "status": "PASS",
                    "free_gb": round(disk_free_gb, 1),
                    "required_gb": min_disk
                }
            else:
                status["checks"]["disk_space"] = {
                    "status": "WARN",
                    "free_gb": round(disk_free_gb, 1),
                    "required_gb": min_disk
                }
                status["issues"].append(f"Low disk space: {disk_free_gb:.1f}GB (recommended: {min_disk}GB+)")
                if status["status"] == "PASS":
                    status["status"] = "WARN"
            
            # CPU cores
            cpu_cores = psutil.cpu_count()
            min_cores = self.min_requirements["cpu_cores"]
            
            if cpu_cores >= min_cores:
                status["checks"]["cpu_cores"] = {
                    "status": "PASS",
                    "current": cpu_cores,
                    "required": min_cores
                }
            else:
                status["checks"]["cpu_cores"] = {
                    "status": "WARN",
                    "current": cpu_cores,
                    "required": min_cores
                }
                status["issues"].append(f"Few CPU cores: {cpu_cores} (recommended: {min_cores}+)")
                if status["status"] == "PASS":
                    status["status"] = "WARN"
            
        except Exception as e:
            status["status"] = "ERROR"
            status["error"] = str(e)
        
        return status
    
    def _check_python_environment(self) -> Dict[str, Any]:
        """Check Python environment setup"""
        status = {
            "status": "PASS",
            "checks": {},
            "issues": []
        }
        
        try:
            # Python executable
            python_exe = sys.executable
            status["checks"]["python_executable"] = {
                "path": python_exe,
                "exists": os.path.exists(python_exe)
            }
            
            # Virtual environment detection
            in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
            status["checks"]["virtual_environment"] = {
                "detected": in_venv,
                "prefix": sys.prefix,
                "base_prefix": getattr(sys, 'base_prefix', sys.prefix)
            }
            
            if not in_venv:
                status["issues"].append("Not running in virtual environment (recommended)")
                if status["status"] == "PASS":
                    status["status"] = "WARN"
            
            # Pip availability
            try:
                import pip
                pip_version = pip.__version__
                status["checks"]["pip"] = {
                    "available": True,
                    "version": pip_version
                }
            except ImportError:
                status["checks"]["pip"] = {"available": False}
                status["issues"].append("pip not available")
                status["status"] = "FAIL"
            
            # Site packages writability
            try:
                import site
                site_packages = site.getsitepackages()[0] if site.getsitepackages() else None
                if site_packages:
                    writable = os.access(site_packages, os.W_OK)
                    status["checks"]["site_packages"] = {
                        "path": site_packages,
                        "writable": writable
                    }
                    
                    if not writable:
                        status["issues"].append("Site-packages not writable (may need sudo for installs)")
                        if status["status"] == "PASS":
                            status["status"] = "WARN"
            except Exception:
                pass
            
        except Exception as e:
            status["status"] = "ERROR"
            status["error"] = str(e)
        
        return status
    
    async def _check_python_dependencies(self) -> Dict[str, Any]:
        """Check Python package dependencies"""
        status = {
            "status": "PASS",
            "categories": {},
            "missing_required": [],
            "missing_optional": [],
            "issues": []
        }
        
        try:
            for category, packages in self.python_dependencies.items():
                category_status = {
                    "available": [],
                    "missing": [],
                    "versions": {}
                }
                
                for package in packages:
                    try:
                        module = importlib.import_module(package.replace('-', '_'))
                        version = getattr(module, '__version__', 'unknown')
                        
                        category_status["available"].append(package)
                        category_status["versions"][package] = version
                        
                    except ImportError:
                        category_status["missing"].append(package)
                        
                        if category == "required":
                            status["missing_required"].append(package)
                        else:
                            status["missing_optional"].append(package)
                
                status["categories"][category] = category_status
            
            # Assess overall status
            if status["missing_required"]:
                status["status"] = "FAIL"
                status["issues"].append(f"Missing required packages: {', '.join(status['missing_required'])}")
            elif status["missing_optional"]:
                status["status"] = "WARN"
                status["issues"].append(f"Missing optional packages: {', '.join(status['missing_optional'][:5])}")
            
        except Exception as e:
            status["status"] = "ERROR"
            status["error"] = str(e)
        
        return status
    
    async def _check_system_tools(self) -> Dict[str, Any]:
        """Check system tool availability"""
        status = {
            "status": "PASS",
            "categories": {},
            "missing_required": [],
            "missing_optional": [],
            "issues": []
        }
        
        try:
            for category, tools in self.system_tools.items():
                category_status = {
                    "available": [],
                    "missing": [],
                    "versions": {}
                }
                
                for tool in tools:
                    tool_available, tool_version = await self._check_tool_availability(tool)
                    
                    if tool_available:
                        category_status["available"].append(tool)
                        if tool_version:
                            category_status["versions"][tool] = tool_version
                    else:
                        category_status["missing"].append(tool)
                        
                        if category == "required":
                            status["missing_required"].append(tool)
                        else:
                            status["missing_optional"].append(tool)
                
                status["categories"][category] = category_status
            
            # Assess overall status
            if status["missing_required"]:
                status["status"] = "FAIL"
                status["issues"].append(f"Missing required tools: {', '.join(status['missing_required'])}")
            elif len(status["missing_optional"]) > 10:
                status["status"] = "WARN"
                status["issues"].append(f"Many optional tools missing: {len(status['missing_optional'])} tools")
            
        except Exception as e:
            status["status"] = "ERROR"
            status["error"] = str(e)
        
        return status
    
    async def _check_tool_availability(self, tool: str) -> Tuple[bool, Optional[str]]:
        """Check if a system tool is available"""
        try:
            # First check if tool exists in PATH
            if shutil.which(tool):
                # Try to get version
                version_commands = [
                    [tool, '--version'],
                    [tool, '-V'],
                    [tool, 'version'],
                    [tool, '-v']
                ]
                
                for cmd in version_commands:
                    try:
                        result = subprocess.run(
                            cmd,
                            capture_output=True,
                            text=True,
                            timeout=10
                        )
                        
                        if result.returncode == 0:
                            # Extract version from output
                            output = result.stdout.strip()
                            if output:
                                return True, output.split('\n')[0][:100]  # First line, truncated
                            else:
                                return True, result.stderr.strip().split('\n')[0][:100]
                        
                    except subprocess.TimeoutExpired:
                        return True, "timeout"
                    except Exception:
                        continue
                
                return True, "available"
            else:
                return False, None
                
        except Exception:
            return False, None
    
    def _check_gpu_availability(self) -> Dict[str, Any]:
        """Check GPU availability and capabilities"""
        status = {
            "status": "INFO",
            "nvidia_available": False,
            "amd_available": False,
            "intel_available": False,
            "devices": [],
            "libraries": {},
            "issues": []
        }
        
        try:
            # Check NVIDIA GPUs
            nvidia_status = self._check_nvidia_gpu()
            status.update(nvidia_status)
            
            # Check PyTorch GPU support
            try:
                import torch
                status["libraries"]["pytorch"] = {
                    "available": True,
                    "version": torch.__version__,
                    "cuda_available": torch.cuda.is_available(),
                    "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
                }
            except ImportError:
                status["libraries"]["pytorch"] = {"available": False}
            
            # Check TensorFlow GPU support
            try:
                import tensorflow as tf
                gpu_devices = tf.config.list_physical_devices('GPU')
                status["libraries"]["tensorflow"] = {
                    "available": True,
                    "version": tf.__version__,
                    "gpu_devices": len(gpu_devices)
                }
            except ImportError:
                status["libraries"]["tensorflow"] = {"available": False}
            
            # Overall GPU status
            if status["nvidia_available"] or status["amd_available"]:
                status["status"] = "PASS"
            else:
                status["status"] = "WARN"
                status["issues"].append("No GPU detected - analysis will use CPU only")
            
        except Exception as e:
            status["status"] = "ERROR"
            status["error"] = str(e)
        
        return status
    
    def _check_nvidia_gpu(self) -> Dict[str, Any]:
        """Check NVIDIA GPU availability"""
        result = {
            "nvidia_available": False,
            "nvidia_driver_version": None,
            "cuda_version": None,
            "devices": []
        }
        
        try:
            # Try pynvml
            try:
                import pynvml
                pynvml.nvmlInit()
                
                driver_version = pynvml.nvmlSystemGetDriverVersion().decode()
                cuda_version = pynvml.nvmlSystemGetCudaDriverVersion()
                
                result["nvidia_available"] = True
                result["nvidia_driver_version"] = driver_version
                result["cuda_version"] = f"{cuda_version // 1000}.{(cuda_version % 1000) // 10}"
                
                device_count = pynvml.nvmlDeviceGetCount()
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    name = pynvml.nvmlDeviceGetName(handle).decode()
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    
                    result["devices"].append({
                        "id": i,
                        "name": name,
                        "memory_total": memory_info.total,
                        "memory_free": memory_info.free
                    })
                
                return result
                
            except ImportError:
                pass
            
            # Try nvidia-smi command
            try:
                result_nvidia_smi = subprocess.run(
                    ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if result_nvidia_smi.returncode == 0:
                    result["nvidia_available"] = True
                    
                    for i, line in enumerate(result_nvidia_smi.stdout.strip().split('\n')):
                        if line.strip():
                            parts = line.split(', ')
                            if len(parts) >= 2:
                                result["devices"].append({
                                    "id": i,
                                    "name": parts[0],
                                    "memory_total": int(parts[1]) * 1024 * 1024  # Convert MB to bytes
                                })
                
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
            
        except Exception as e:
            self.logger.debug(f"NVIDIA GPU check failed: {e}")
        
        return result
    
    async def _check_database_connectivity(self) -> Dict[str, Any]:
        """Check database connectivity"""
        status = {
            "status": "INFO",
            "sqlite": {"available": True, "version": None},
            "postgresql": {"available": False, "connectable": False},
            "neo4j": {"available": False, "connectable": False},
            "issues": []
        }
        
        try:
            # SQLite (always available in Python)
            import sqlite3
            status["sqlite"]["version"] = sqlite3.sqlite_version
            
            # PostgreSQL
            try:
                import asyncpg
                status["postgresql"]["available"] = True
                # Would need connection parameters to test connectivity
            except ImportError:
                pass
            
            # Neo4j
            try:
                from neo4j import GraphDatabase
                status["neo4j"]["available"] = True
                # Would need connection parameters to test connectivity
            except ImportError:
                pass
            
            # MongoDB
            try:
                import pymongo
                status["mongodb"] = {"available": True}
            except ImportError:
                status["mongodb"] = {"available": False}
            
            status["status"] = "PASS"
            
        except Exception as e:
            status["status"] = "ERROR"
            status["error"] = str(e)
        
        return status
    
    async def _check_network_connectivity(self) -> Dict[str, Any]:
        """Check network connectivity for cloud services"""
        status = {
            "status": "PASS",
            "internet": False,
            "services": {},
            "issues": []
        }
        
        try:
            import aiohttp
            
            # Test basic internet connectivity
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                    async with session.get('https://httpbin.org/ip') as response:
                        if response.status == 200:
                            status["internet"] = True
            except Exception:
                status["internet"] = False
                status["issues"].append("No internet connectivity detected")
                status["status"] = "WARN"
            
            # Test specific services
            services_to_test = {
                "virustotal": "https://www.virustotal.com",
                "openai": "https://api.openai.com",
                "anthropic": "https://api.anthropic.com"
            }
            
            if status["internet"]:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                    for service, url in services_to_test.items():
                        try:
                            async with session.get(url) as response:
                                status["services"][service] = {
                                    "reachable": True,
                                    "status_code": response.status
                                }
                        except Exception as e:
                            status["services"][service] = {
                                "reachable": False,
                                "error": str(e)
                            }
            
        except ImportError:
            status["issues"].append("aiohttp not available for network testing")
            if status["status"] == "PASS":
                status["status"] = "WARN"
        except Exception as e:
            status["status"] = "ERROR"
            status["error"] = str(e)
        
        return status
    
    def _generate_recommendations(self):
        """Generate system recommendations based on check results"""
        self.recommendations = []
        
        # Check for missing required dependencies
        if hasattr(self, 'dependency_status'):
            missing_required = self.dependency_status.get("missing_required", [])
            if missing_required:
                self.recommendations.append(
                    f"Install required Python packages: pip install {' '.join(missing_required)}"
                )
        
        # Memory recommendations
        memory_gb = psutil.virtual_memory().total / (1024**3)
        if memory_gb < 8:
            self.recommendations.append(
                "Consider upgrading to 8GB+ RAM for better performance with large files"
            )
        
        # GPU recommendations
        if not self._has_gpu():
            self.recommendations.append(
                "Install NVIDIA GPU drivers and CUDA for accelerated ML analysis"
            )
        
        # Virtual environment recommendation
        in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
        if not in_venv:
            self.recommendations.append(
                "Use a virtual environment: python -m venv steganalyzer_env && source steganalyzer_env/bin/activate"
            )
        
        # Tool installation recommendations
        self.recommendations.append(
            "Install steganography tools: apt-get install steghide outguess binwalk foremost (Ubuntu/Debian)"
        )
        
        # Performance recommendations
        cpu_cores = psutil.cpu_count()
        if cpu_cores < 4:
            self.recommendations.append(
                "Multi-core CPU (4+ cores) recommended for parallel analysis"
            )
    
    def _calculate_overall_status(self, check_results: List[Dict[str, Any]]) -> str:
        """Calculate overall system status"""
        statuses = [check.get("status", "UNKNOWN") for check in check_results]
        
        if "ERROR" in statuses:
            return "ERROR"
        elif "FAIL" in statuses:
            return "FAIL"
        elif "WARN" in statuses:
            return "WARN"
        else:
            return "PASS"
    
    def _assess_capabilities(self) -> Dict[str, Any]:
        """Assess system capabilities for different analysis types"""
        capabilities = {
            "basic_analysis": True,
            "image_analysis": False,
            "audio_analysis": False,
            "ml_analysis": False,
            "gpu_acceleration": False,
            "cloud_integration": False,
            "parallel_processing": True
        }
        
        # Image analysis capability
        try:
            import PIL
            import cv2
            capabilities["image_analysis"] = True
        except ImportError:
            pass
        
        # Audio analysis capability
        try:
            import librosa
            capabilities["audio_analysis"] = True
        except ImportError:
            try:
                import wave
                import scipy
                capabilities["audio_analysis"] = True
            except ImportError:
                pass
        
        # ML analysis capability
        try:
            import torch
            capabilities["ml_analysis"] = True
            if torch.cuda.is_available():
                capabilities["gpu_acceleration"] = True
        except ImportError:
            try:
                import tensorflow as tf
                capabilities["ml_analysis"] = True
                if tf.config.list_physical_devices('GPU'):
                    capabilities["gpu_acceleration"] = True
            except ImportError:
                pass
        
        # Cloud integration capability
        try:
            import requests
            capabilities["cloud_integration"] = True
        except ImportError:
            pass
        
        return capabilities
    
    def _has_gpu(self) -> bool:
        """Check if system has GPU"""
        try:
            # Try NVIDIA
            try:
                import pynvml
                pynvml.nvmlInit()
                return pynvml.nvmlDeviceGetCount() > 0
            except:
                pass
            
            # Try PyTorch
            try:
                import torch
                return torch.cuda.is_available()
            except:
                pass
            
            # Try TensorFlow
            try:
                import tensorflow as tf
                return len(tf.config.list_physical_devices('GPU')) > 0
            except:
                pass
            
            return False
            
        except Exception:
            return False
    
    def generate_install_script(self) -> str:
        """Generate installation script based on system check"""
        script_lines = [
            "#!/bin/bash",
            "# StegAnalyzer Installation Script",
            "# Generated based on system requirements check",
            "",
            "set -e",
            "",
            "echo 'Installing StegAnalyzer dependencies...'",
            ""
        ]
        
        # System package installation (Ubuntu/Debian)
        system_packages = [
            "python3-pip", "python3-venv", "python3-dev",
            "build-essential", "libmagic1", "libmagic-dev",
            "steghide", "outguess", "binwalk", "foremost",
            "exiftool", "yara", "clamav"
        ]
        
        script_lines.extend([
            "# System packages",
            "if command -v apt-get &> /dev/null; then",
            "    sudo apt-get update",
            f"    sudo apt-get install -y {' '.join(system_packages)}",
            "elif command -v yum &> /dev/null; then",
            "    sudo yum install -y python3-pip python3-devel gcc",
            "elif command -v brew &> /dev/null; then",
            "    brew install python magic steghide exiftool",
            "fi",
            ""
        ])
        
        # Python packages
        required_packages = [
            "numpy", "Pillow", "opencv-python", "matplotlib",
            "scipy", "scikit-learn", "requests", "aiohttp",
            "jinja2", "networkx", "psutil", "python-magic"
        ]
        
        optional_packages = [
            "torch", "torchvision", "tensorflow", "transformers",
            "anthropic", "openai", "weasyprint", "librosa"
        ]
        
        script_lines.extend([
            "# Python virtual environment",
            "python3 -m venv steganalyzer_env",
            "source steganalyzer_env/bin/activate",
            "",
            "# Required Python packages",
            f"pip install {' '.join(required_packages)}",
            "",
            "# Optional packages (install if needed)",
            f"# pip install {' '.join(optional_packages)}",
            "",
            "echo 'Installation completed!'",
            "echo 'Activate environment with: source steganalyzer_env/bin/activate'"
        ])
        
        return '\n'.join(script_lines)
    
    def save_system_report(self, check_result: Dict[str, Any], output_path: str = "system_check_report.json"):
        """Save detailed system check report"""
        try:
            with open(output_path, 'w') as f:
                json.dump(check_result, f, indent=2, default=str)
            
            self.logger.info(f"System check report saved to {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Failed to save system report: {e}")
            return None
