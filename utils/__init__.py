"""
Utility functions
"""
from checkpoint import CheckpointManager  # This is checkpoint.py, not checkpoint_manager.py
from system_check import SystemChecker

try:
    from gpu_manager import GPUManager
except ImportError:
    GPUManager = None

try:
    from logger import setup_logging
except ImportError:
    def setup_logging(level="INFO"):
        import logging
        logging.basicConfig(level=getattr(logging, level.upper()))

__all__ = ['CheckpointManager', 'SystemChecker', 'setup_logging']
if GPUManager:
    __all__.append('GPUManager')
