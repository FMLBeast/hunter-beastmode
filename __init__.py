"""
StegAnalyzer - Advanced Steganography Detection Framework
"""

__version__ = "1.0.0"
__author__ = "StegAnalyzer Team"

# Core functionality should always be available
try:
    from config import Config
except ImportError:
    Config = None

__all__ = []
if Config:
    __all__.append('Config')
