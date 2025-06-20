#!/usr/bin/env python3
"""
Configuration updates for Cascade Analyzer integration
Add these to your existing config/steg_config.py
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class CascadeConfig:
    """Configuration for cascade analyzer"""
    
    # Core cascade settings
    max_depth: int = 10
    enable_zsteg: bool = True
    enable_binwalk: bool = True
    save_extracts: bool = True
    
    # Analysis parameters
    zsteg_timeout: float = 30.0
    binwalk_timeout: float = 120.0
    max_file_size: int = 100 * 1024 * 1024  # 100MB limit
    
    # File type filters
    image_extensions: List[str] = None
    binwalk_extensions: List[str] = None
    
    # Extraction settings
    extraction_dir: str = "cascade_extracts"
    keep_extraction_tree: bool = True
    compress_results: bool = False
    
    # Performance settings
    max_concurrent_extractions: int = 3
    memory_limit_mb: int = 2048
    
    # Confidence thresholds
    min_zsteg_confidence: float = 0.3
    min_extract_size: int = 10  # Minimum bytes to consider valid extraction
    
    # Exotic zsteg parameters
    enable_exotic_params: bool = True
    custom_zsteg_params: List[str] = None
    
    # Safety limits
    max_extractions_per_file: int = 1000
    max_total_extractions: int = 10000
    
    def __post_init__(self):
        # Set default file extensions if not provided
        if self.image_extensions is None:
            self.image_extensions = ['.png', '.bmp', '.gif', '.tiff', '.tif', '.webp']
        
        if self.binwalk_extensions is None:
            self.binwalk_extensions = [
                '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.tif', '.webp',
                '.pdf', '.zip', '.rar', '.7z', '.tar', '.gz', '.exe', '.bin',
                '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx'
            ]
        
        if self.custom_zsteg_params is None:
            self.custom_zsteg_params = []


# Add this to your main Config class in steg_config.py
class Config:
    """Enhanced StegAnalyzer configuration with cascade support"""
    
    def __init__(self):
        # ... existing config initialization ...
        
        # Add cascade configuration
        self.cascade = CascadeConfig()
        
        # Cascade mode settings
        self.cascade_mode = False  # Enable to run cascade by default
        self.force_cascade = False  # Force cascade even on non-image files
        
        # Integration settings
        self.cascade_integration = {
            'auto_cascade_on_findings': True,  # Auto-run cascade if initial analysis finds something
            'cascade_priority': 1,  # Priority level for cascade tasks
            'cascade_after_tools': ['classic_stego', 'image_forensics'],  # Run cascade after these tools
            'cascade_timeout': 300,  # Max time for cascade analysis (seconds)
        }
    
    def enable_cascade_mode(self, max_depth: int = 10, enable_exotic: bool = True):
        """Enable cascade analysis mode"""
        self.cascade_mode = True
        self.cascade.max_depth = max_depth
        self.cascade.enable_exotic_params = enable_exotic
    
    def set_cascade_performance(self, max_concurrent: int = 3, memory_limit: int = 2048):
        """Configure cascade performance settings"""
        self.cascade.max_concurrent_extractions = max_concurrent
        self.cascade.memory_limit_mb = memory_limit
    
    def add_custom_zsteg_params(self, params: List[str]):
        """Add custom zsteg parameter combinations"""
        if not self.cascade.custom_zsteg_params:
            self.cascade.custom_zsteg_params = []
        self.cascade.custom_zsteg_params.extend(params)


# Update the JSON config schema to include cascade settings
CASCADE_CONFIG_SCHEMA = {
    "cascade": {
        "max_depth": 10,
        "enable_zsteg": True,
        "enable_binwalk": True,
        "save_extracts": True,
        "zsteg_timeout": 30.0,
        "binwalk_timeout": 120.0,
        "max_file_size": 104857600,
        "extraction_dir": "cascade_extracts",
        "keep_extraction_tree": True,
        "compress_results": False,
        "max_concurrent_extractions": 3,
        "memory_limit_mb": 2048,
        "min_zsteg_confidence": 0.3,
        "min_extract_size": 10,
        "enable_exotic_params": True,
        "max_extractions_per_file": 1000,
        "max_total_extractions": 10000,
        "image_extensions": [".png", ".bmp", ".gif", ".tiff", ".tif", ".webp"],
        "binwalk_extensions": [
            ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".tif", ".webp",
            ".pdf", ".zip", ".rar", ".7z", ".tar", ".gz", ".exe", ".bin",
            ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx"
        ],
        "custom_zsteg_params": []
    },
    "cascade_mode": False,
    "force_cascade": False,
    "cascade_integration": {
        "auto_cascade_on_findings": True,
        "cascade_priority": 1,
        "cascade_after_tools": ["classic_stego", "image_forensics"],
        "cascade_timeout": 300
    }
}

# Add this to your default config JSON file (config/default.json)
DEFAULT_CASCADE_CONFIG = """
{
    "cascade": {
        "max_depth": 10,
        "enable_zsteg": true,
        "enable_binwalk": true,
        "save_extracts": true,
        "zsteg_timeout": 30.0,
        "binwalk_timeout": 120.0,
        "max_file_size": 104857600,
        "extraction_dir": "cascade_extracts",
        "keep_extraction_tree": true,
        "compress_results": false,
        "max_concurrent_extractions": 3,
        "memory_limit_mb": 2048,
        "min_zsteg_confidence": 0.3,
        "min_extract_size": 10,
        "enable_exotic_params": true,
        "max_extractions_per_file": 1000,
        "max_total_extractions": 10000,
        "custom_zsteg_params": []
    },
    "cascade_mode": false,
    "force_cascade": false,
    "cascade_integration": {
        "auto_cascade_on_findings": true,
        "cascade_priority": 1,
        "cascade_after_tools": ["classic_stego", "image_forensics"],
        "cascade_timeout": 300
    }
}
"""

# Configuration validation for cascade settings
def validate_cascade_config(config: CascadeConfig) -> List[str]:
    """Validate cascade configuration and return any errors"""
    errors = []
    
    if config.max_depth < 1 or config.max_depth > 50:
        errors.append("max_depth must be between 1 and 50")
    
    if config.zsteg_timeout < 1 or config.zsteg_timeout > 300:
        errors.append("zsteg_timeout must be between 1 and 300 seconds")
    
    if config.binwalk_timeout < 1 or config.binwalk_timeout > 600:
        errors.append("binwalk_timeout must be between 1 and 600 seconds")
    
    if config.max_file_size < 1024 or config.max_file_size > 1024**3:  # 1GB limit
        errors.append("max_file_size must be between 1KB and 1GB")
    
    if config.max_concurrent_extractions < 1 or config.max_concurrent_extractions > 10:
        errors.append("max_concurrent_extractions must be between 1 and 10")
    
    if config.memory_limit_mb < 256 or config.memory_limit_mb > 16384:
        errors.append("memory_limit_mb must be between 256 and 16384 MB")
    
    if config.min_zsteg_confidence < 0 or config.min_zsteg_confidence > 1:
        errors.append("min_zsteg_confidence must be between 0 and 1")
    
    return errors

# Example configuration presets
CASCADE_PRESETS = {
    "fast": {
        "max_depth": 5,
        "enable_exotic_params": False,
        "zsteg_timeout": 15.0,
        "binwalk_timeout": 60.0,
        "max_concurrent_extractions": 5
    },
    "balanced": {
        "max_depth": 10,
        "enable_exotic_params": True,
        "zsteg_timeout": 30.0,
        "binwalk_timeout": 120.0,
        "max_concurrent_extractions": 3
    },
    "thorough": {
        "max_depth": 20,
        "enable_exotic_params": True,
        "zsteg_timeout": 60.0,
        "binwalk_timeout": 300.0,
        "max_concurrent_extractions": 2
    },
    "extreme": {
        "max_depth": 50,
        "enable_exotic_params": True,
        "zsteg_timeout": 120.0,
        "binwalk_timeout": 600.0,
        "max_concurrent_extractions": 1
    }
}

def apply_cascade_preset(config: Config, preset_name: str):
    """Apply a cascade configuration preset"""
    if preset_name not in CASCADE_PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}")
    
    preset = CASCADE_PRESETS[preset_name]
    
    for key, value in preset.items():
        if hasattr(config.cascade, key):
            setattr(config.cascade, key, value)
