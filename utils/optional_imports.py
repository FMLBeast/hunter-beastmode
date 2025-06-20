"""
Optional import wrapper - handles missing dependencies gracefully
"""

import logging
import warnings

def optional_import(module_name, package=None, min_version=None):
    """
    Import a module optionally, returning None if not available
    
    Args:
        module_name: Name of module to import
        package: Package name if different from module
        min_version: Minimum version required
    
    Returns:
        Module if available, None otherwise
    """
    try:
        module = __import__(module_name, fromlist=[''])
        
        if min_version and hasattr(module, '__version__'):
            from packaging import version
            if version.parse(module.__version__) < version.parse(min_version):
                warnings.warn(f"{module_name} version {module.__version__} < {min_version}")
                
        return module
        
    except ImportError as e:
        logging.warning(f"Optional dependency {module_name} not available: {e}")
        return None

# Pre-import common optional dependencies
anthropic = optional_import('anthropic')
easyocr = optional_import('easyocr')
aubio = optional_import('aubio')
deepspeech = optional_import('deepspeech')
PyPDF2 = optional_import('PyPDF2')
pdfplumber = optional_import('pdfplumber')

# Cryptography imports
try:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.backends import default_backend
    cryptography_available = True
except ImportError:
    cryptography_available = False
    Cipher = algorithms = modes = default_backend = None

# OLE tools
try:
    from oletools import olevba, rtfobj
    oletools_available = True
except ImportError:
    oletools_available = False
    olevba = rtfobj = None
