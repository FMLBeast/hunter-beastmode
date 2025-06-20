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
    