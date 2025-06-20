"""
Analysis tools
"""
# Add cascade analyzer import
try:
    from .cascade_analyzer import CascadeAnalyzer
except ImportError as e:
    print(f"Warning: CascadeAnalyzer not available: {e}")
    CascadeAnalyzer = None

# Add to __all__
__all__ = [
    'ClassicStegoTools', 'ImageForensicsTools', 'AudioAnalysisTools',
    'FileForensicsTools', 'CryptoAnalysisTools', 'MetadataCarving',
    'CascadeAnalyzer'  # Add this line
]
