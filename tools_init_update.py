"""
Analysis tools and integrations - Updated with Cascade Analyzer
"""

# Import core tools with error handling
try:
    from .classic_stego import ClassicStegoTools
except ImportError as e:
    print(f"Warning: ClassicStegoTools not available: {e}")
    ClassicStegoTools = None

try:
    from .file_forensics import FileForensicsTools
except ImportError as e:
    print(f"Warning: FileForensicsTools not available: {e}")
    FileForensicsTools = None

try:
    from .crypto_analysis import CryptoAnalysisTools
except ImportError as e:
    print(f"Warning: CryptoAnalysisTools not available: {e}")
    CryptoAnalysisTools = None

try:
    from .metadata_carving import MetadataCarving
except ImportError as e:
    print(f"Warning: MetadataCarving not available: {e}")
    MetadataCarving = None

# Tools with heavy dependencies
try:
    from .image_forensics import ImageForensicsTools
except ImportError as e:
    print(f"Warning: ImageForensicsTools not available (missing image dependencies): {e}")
    ImageForensicsTools = None

try:
    from .audio_analysis import AudioAnalysisTools
except ImportError as e:
    print(f"Warning: AudioAnalysisTools not available (missing audio dependencies): {e}")
    AudioAnalysisTools = None

# NEW: Cascade Analyzer
try:
    from .cascade_analyzer import CascadeAnalyzer
    print("✅ CascadeAnalyzer loaded successfully")
except ImportError as e:
    print(f"Warning: CascadeAnalyzer not available (missing zsteg/binwalk): {e}")
    CascadeAnalyzer = None

# Export available tools
__all__ = []

# Add available core tools
if ClassicStegoTools:
    __all__.append('ClassicStegoTools')
if ImageForensicsTools:
    __all__.append('ImageForensicsTools')
if AudioAnalysisTools:
    __all__.append('AudioAnalysisTools')
if FileForensicsTools:
    __all__.append('FileForensicsTools')
if CryptoAnalysisTools:
    __all__.append('CryptoAnalysisTools')
if MetadataCarving:
    __all__.append('MetadataCarving')

# Add cascade analyzer
if CascadeAnalyzer:
    __all__.append('CascadeAnalyzer')

# Tool availability checker
def get_available_tools():
    """Return dictionary of available tools"""
    return {
        'ClassicStegoTools': ClassicStegoTools is not None,
        'ImageForensicsTools': ImageForensicsTools is not None,
        'AudioAnalysisTools': AudioAnalysisTools is not None,
        'FileForensicsTools': FileForensicsTools is not None,
        'CryptoAnalysisTools': CryptoAnalysisTools is not None,
        'MetadataCarving': MetadataCarving is not None,
        'CascadeAnalyzer': CascadeAnalyzer is not None,
    }

# Tool initialization helper
def initialize_available_tools(config):
    """Initialize all available tools with config"""
    tools = {}
    
    if ClassicStegoTools:
        try:
            tools['classic_stego'] = ClassicStegoTools(config)
        except Exception as e:
            print(f"Failed to initialize ClassicStegoTools: {e}")
    
    if ImageForensicsTools:
        try:
            tools['image_forensics'] = ImageForensicsTools(config)
        except Exception as e:
            print(f"Failed to initialize ImageForensicsTools: {e}")
    
    if AudioAnalysisTools:
        try:
            tools['audio_analysis'] = AudioAnalysisTools(config)
        except Exception as e:
            print(f"Failed to initialize AudioAnalysisTools: {e}")
    
    if FileForensicsTools:
        try:
            tools['file_forensics'] = FileForensicsTools(config)
        except Exception as e:
            print(f"Failed to initialize FileForensicsTools: {e}")
    
    if CryptoAnalysisTools:
        try:
            tools['crypto_analysis'] = CryptoAnalysisTools(config)
        except Exception as e:
            print(f"Failed to initialize CryptoAnalysisTools: {e}")
    
    if MetadataCarving:
        try:
            tools['metadata_carving'] = MetadataCarving(config)
        except Exception as e:
            print(f"Failed to initialize MetadataCarving: {e}")
    
    # Initialize cascade analyzer
    if CascadeAnalyzer:
        try:
            tools['cascade_analyzer'] = CascadeAnalyzer(config)
            print("🎯 CascadeAnalyzer initialized successfully")
        except Exception as e:
            print(f"Failed to initialize CascadeAnalyzer: {e}")
    
    return tools

# Print tool status on import
if __name__ != "__main__":
    available = get_available_tools()
    available_count = sum(available.values())
    total_count = len(available)
    print(f"📊 Tools available: {available_count}/{total_count}")
    
    if CascadeAnalyzer:
        print("🚀 Cascade analysis ready!")
