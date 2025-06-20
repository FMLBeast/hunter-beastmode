"""
Configuration Management System for StegAnalyzer
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import logging

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

        
@dataclass
class DatabaseConfig:
    type: str = "sqlite"  # sqlite, neo4j, postgresql
    path: str = "data/steganalyzer.db"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "steganalyzer"
    postgres_user: str = "steguser"
    postgres_password: str = "password"
    connection_pool_size: int = 10
    query_timeout: int = 300

@dataclass
class OrchestratorConfig:
    max_cpu_workers: int = 16
    max_gpu_workers: int = 2
    max_concurrent_files: int = 4
    task_timeout: int = 3600
    checkpoint_interval: int = 300  # seconds
    retry_attempts: int = 3
    memory_limit_gb: int = 32
    temp_directory: str = "/tmp/steganalyzer"

@dataclass
class AnalysisConfig:
    file_patterns: List[str] = None
    max_file_size_mb: int = 1024
    skip_extensions: List[str] = None
    deep_scan: bool = True
    quick_mode: bool = False
    extraction_depth: int = 10
    
    def __post_init__(self):
        if self.file_patterns is None:
            self.file_patterns = [
                "*.jpg", "*.jpeg", "*.png", "*.gif", "*.bmp", "*.tiff", "*.webp",
                "*.mp3", "*.wav", "*.flac", "*.ogg", "*.m4a", "*.aac",
                "*.mp4", "*.avi", "*.mkv", "*.mov", "*.wmv", "*.flv",
                "*.pdf", "*.doc", "*.docx", "*.xls", "*.xlsx", "*.ppt", "*.pptx",
                "*.zip", "*.rar", "*.7z", "*.tar", "*.gz", "*.bz2",
                "*.exe", "*.dll", "*.so", "*.bin", "*.iso"
            ]
        if self.skip_extensions is None:
            self.skip_extensions = [".log", ".tmp", ".cache", ".lock"]

@dataclass
class ClassicStegoConfig:
    steghide_enabled: bool = True
    steghide_wordlist: str = "wordlists/common.txt"
    outguess_enabled: bool = True
    zsteg_enabled: bool = True
    stegseek_enabled: bool = True
    stegseek_wordlist: str = "wordlists/rockyou.txt"
    stegoveritas_enabled: bool = True
    stegcracker_enabled: bool = True
    stegcracker_timeout: int = 1800
    exiftool_enabled: bool = True
    binwalk_enabled: bool = True
    binwalk_extract: bool = True
    foremost_enabled: bool = True
    strings_enabled: bool = True
    strings_min_length: int = 4

@dataclass
class ImageForensicsConfig:
    stegdetect_enabled: bool = True
    lsb_analysis_enabled: bool = True
    noise_analysis_enabled: bool = True
    error_level_analysis: bool = True
    jpeg_analysis: bool = True
    metadata_analysis: bool = True
    histogram_analysis: bool = True
    frequency_analysis: bool = True
    pixelknot_enabled: bool = True

@dataclass
class AudioAnalysisConfig:
    spectral_analysis: bool = True
    lsb_audio_enabled: bool = True
    echo_hiding_detection: bool = True
    phase_coding_detection: bool = True
    spread_spectrum_detection: bool = True
    deep_speech_enabled: bool = True
    deep_speech_model: str = "deepspeech-0.9.3-models.pbmm"
    sox_enabled: bool = True
    ffmpeg_enabled: bool = True

@dataclass
class MLConfig:
    gpu_enabled: bool = True
    gpu_memory_limit: int = 8192  # MB
    model_cache_dir: str = "models/"
    cnn_steg_model: str = "models/cnn_steg_detector.pth"
    noiseprint_model: str = "models/noiseprint.pth"
    deep_stego_model: str = "models/deep_stego.pth"
    batch_size: int = 32
    confidence_threshold: float = 0.7
    use_ensemble: bool = True

@dataclass
class LLMConfig:
    provider: str = "anthropic"  # anthropic, openai, huggingface, local
    model_name: str = "claude-3-sonnet"
    api_key: str = ""
    base_url: str = ""
    max_tokens: int = 4096
    temperature: float = 0.1
    local_model_path: str = ""
    gpu_layers: int = 35
    context_window: int = 8192
    system_prompt: str = ""

@dataclass
class MultimodalConfig:
    enabled: bool = True
    vision_model: str = "gpt-4-vision-preview"
    ocr_model: str = "easyocr"
    trocr_model: str = "microsoft/trocr-base-printed"
    clip_model: str = "openai/clip-vit-base-patch32"
    gpu_acceleration: bool = True

@dataclass
class CryptoConfig:
    entropy_analysis: bool = True
    pattern_detection: bool = True
    frequency_analysis: bool = True
    key_search_enabled: bool = True
    hashcat_enabled: bool = True
    john_enabled: bool = True
    wordlists_dir: str = "wordlists/"
    custom_patterns: List[str] = None
    
    def __post_init__(self):
        if self.custom_patterns is None:
            self.custom_patterns = [
                r"[A-Za-z0-9+/]{20,}={0,2}",  # Base64
                r"[0-9a-fA-F]{32,}",          # Hex
                r"-----BEGIN.*-----",          # PEM
            ]

@dataclass
class FileForensicsConfig:
    magic_analysis: bool = True
    signature_verification: bool = True
    header_analysis: bool = True
    footer_analysis: bool = True
    polyglot_detection: bool = True
    file_carving: bool = True
    bulk_extractor: bool = True
    photorec_enabled: bool = True
    pdf_analysis: bool = True
    office_analysis: bool = True

@dataclass
class CloudConfig:
    enabled: bool = False
    virustotal_api_key: str = ""
    hashlookup_enabled: bool = True
    nsrl_enabled: bool = True
    hybrid_analysis_api_key: str = ""
    malware_bazaar_enabled: bool = True
    timeout: int = 30
    rate_limit: float = 1.0  # requests per second

@dataclass
class DashboardConfig:
    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 8080
    debug: bool = False
    auto_open: bool = True
    refresh_interval: int = 2
    max_log_entries: int = 1000

@dataclass
class ReportConfig:
    output_dir: str = "reports/"
    format: str = "html"  # html, json, pdf, markdown
    include_metadata: bool = True
    include_raw_data: bool = False
    include_screenshots: bool = True
    template_dir: str = "templates/"
    auto_open: bool = True

@dataclass
class LoggingConfig:
    level: str = "INFO"
    log_dir: str = "logs/"
    max_size_mb: int = 100
    backup_count: int = 5
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_logging: bool = True
    console_logging: bool = True

class Config:
    def __init__(self, config_path: str = None):
        self.config_path = config_path or "config/default.json"
        
        # Initialize with defaults
        self.database = DatabaseConfig()
        self.orchestrator = OrchestratorConfig()
        self.analysis = AnalysisConfig()
        self.classic_stego = ClassicStegoConfig()
        self.image_forensics = ImageForensicsConfig()
        self.audio_analysis = AudioAnalysisConfig()
        self.ml = MLConfig()
        self.llm = LLMConfig()
        self.multimodal = MultimodalConfig()
        self.crypto = CryptoConfig()
        self.file_forensics = FileForensicsConfig()
        self.cloud = CloudConfig()
        self.dashboard = DashboardConfig()
        self.report = ReportConfig()
        self.logging = LoggingConfig()
        
        # Load configuration file if it exists
        self.load_config()
        
        # Apply environment variable overrides
        self._apply_env_overrides()
        
        # Validate configuration
        self._validate_config()
    
    def load_config(self):
        """Load configuration from JSON file"""
        if not os.path.exists(self.config_path):
            # Create default config file
            self.save_config()
            return
        
        try:
            with open(self.config_path, 'r') as f:
                config_data = json.load(f)
            
            # Update configurations
            for section, data in config_data.items():
                if hasattr(self, section):
                    config_obj = getattr(self, section)
                    for key, value in data.items():
                        if hasattr(config_obj, key):
                            setattr(config_obj, key, value)
            
            logging.info(f"Configuration loaded from {self.config_path}")
            
        except Exception as e:
            logging.error(f"Failed to load config from {self.config_path}: {e}")
            logging.info("Using default configuration")
    
    def save_config(self):
        """Save current configuration to JSON file"""
        config_dir = Path(self.config_path).parent
        config_dir.mkdir(parents=True, exist_ok=True)
        
        config_data = {}
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if hasattr(attr, '__dataclass_fields__'):
                config_data[attr_name] = asdict(attr)
        
        try:
            with open(self.config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            logging.info(f"Configuration saved to {self.config_path}")
        except Exception as e:
            logging.error(f"Failed to save config to {self.config_path}: {e}")
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides"""
        env_mappings = {
            'STEG_DB_TYPE': ('database', 'type'),
            'STEG_DB_PATH': ('database', 'path'),
            'STEG_NEO4J_URI': ('database', 'neo4j_uri'),
            'STEG_NEO4J_USER': ('database', 'neo4j_user'),
            'STEG_NEO4J_PASSWORD': ('database', 'neo4j_password'),
            'STEG_MAX_WORKERS': ('orchestrator', 'max_cpu_workers'),
            'STEG_GPU_WORKERS': ('orchestrator', 'max_gpu_workers'),
            'STEG_TEMP_DIR': ('orchestrator', 'temp_directory'),
            'STEG_LLM_PROVIDER': ('llm', 'provider'),
            'STEG_LLM_MODEL': ('llm', 'model_name'),
            'STEG_LLM_API_KEY': ('llm', 'api_key'),
            'STEG_LLM_BASE_URL': ('llm', 'base_url'),
            'STEG_GPU_ENABLED': ('ml', 'gpu_enabled'),
            'STEG_GPU_MEMORY': ('ml', 'gpu_memory_limit'),
            'STEG_CLOUD_ENABLED': ('cloud', 'enabled'),
            'STEG_VT_API_KEY': ('cloud', 'virustotal_api_key'),
            'STEG_DASHBOARD_PORT': ('dashboard', 'port'),
            'STEG_DASHBOARD_HOST': ('dashboard', 'host'),
            'STEG_OUTPUT_DIR': ('report', 'output_dir'),
            'STEG_LOG_LEVEL': ('logging', 'level'),
        }
        
        for env_var, (section, key) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                config_obj = getattr(self, section)
                # Type conversion
                current_value = getattr(config_obj, key)
                if isinstance(current_value, bool):
                    value = value.lower() in ('true', '1', 'yes', 'on')
                elif isinstance(current_value, int):
                    value = int(value)
                elif isinstance(current_value, float):
                    value = float(value)
                
                setattr(config_obj, key, value)
                logging.debug(f"Applied env override: {env_var} = {value}")
    
    def _validate_config(self):
        """Validate configuration settings"""
        errors = []
        
        # Validate directories
        directories = [
            self.orchestrator.temp_directory,
            self.ml.model_cache_dir,
            self.crypto.wordlists_dir,
            self.report.output_dir,
            self.logging.log_dir
        ]
        
        for directory in directories:
            try:
                Path(directory).mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create directory {directory}: {e}")
        
        # Validate file paths
        if self.classic_stego.steghide_enabled:
            if not os.path.exists(self.classic_stego.steghide_wordlist):
                logging.warning(f"Steghide wordlist not found: {self.classic_stego.steghide_wordlist}")
        
        # Validate GPU settings
        if self.ml.gpu_enabled:
            try:
                import torch
                if not torch.cuda.is_available():
                    logging.warning("GPU enabled but CUDA not available")
                    self.ml.gpu_enabled = False
            except ImportError:
                logging.warning("GPU enabled but PyTorch not available")
                self.ml.gpu_enabled = False
        
        # Validate API keys
        if self.cloud.enabled:
            if not self.cloud.virustotal_api_key and "virustotal" in self.get_enabled_cloud_services():
                logging.warning("VirusTotal enabled but no API key provided")
        
        if self.llm.provider in ["openai", "anthropic"] and not self.llm.api_key:
            logging.warning(f"LLM provider {self.llm.provider} requires API key")
        
        # Validate port availability
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex((self.dashboard.host, self.dashboard.port))
        if result == 0:
            logging.warning(f"Dashboard port {self.dashboard.port} already in use")
        sock.close()
        
        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(errors))
    
    def get_enabled_tools(self) -> Dict[str, List[str]]:
        """Get list of enabled tools by category"""
        enabled = {
            "classic_stego": [],
            "image_forensics": [],
            "audio_analysis": [],
            "ml": [],
            "crypto": [],
            "file_forensics": [],
            "cloud": []
        }
        
        # Classic stego tools
        if self.classic_stego.steghide_enabled:
            enabled["classic_stego"].append("steghide")
        if self.classic_stego.outguess_enabled:
            enabled["classic_stego"].append("outguess")
        if self.classic_stego.zsteg_enabled:
            enabled["classic_stego"].append("zsteg")
        if self.classic_stego.binwalk_enabled:
            enabled["classic_stego"].append("binwalk")
        if self.classic_stego.foremost_enabled:
            enabled["classic_stego"].append("foremost")
        
        # Image forensics
        if self.image_forensics.stegdetect_enabled:
            enabled["image_forensics"].append("stegdetect")
        if self.image_forensics.lsb_analysis_enabled:
            enabled["image_forensics"].append("lsb_analysis")
        if self.image_forensics.noise_analysis_enabled:
            enabled["image_forensics"].append("noise_analysis")
        
        # Audio analysis
        if self.audio_analysis.spectral_analysis:
            enabled["audio_analysis"].append("spectral_analysis")
        if self.audio_analysis.lsb_audio_enabled:
            enabled["audio_analysis"].append("lsb_audio")
        if self.audio_analysis.deep_speech_enabled:
            enabled["audio_analysis"].append("deep_speech")
        
        # ML tools
        if self.ml.gpu_enabled:
            enabled["ml"].append("cnn_steg_detection")
            enabled["ml"].append("noiseprint")
            enabled["ml"].append("deep_stego")
        
        # Crypto tools
        if self.crypto.entropy_analysis:
            enabled["crypto"].append("entropy_analysis")
        if self.crypto.pattern_detection:
            enabled["crypto"].append("pattern_detection")
        if self.crypto.hashcat_enabled:
            enabled["crypto"].append("hashcat")
        
        # File forensics
        if self.file_forensics.magic_analysis:
            enabled["file_forensics"].append("magic_analysis")
        if self.file_forensics.bulk_extractor:
            enabled["file_forensics"].append("bulk_extractor")
        if self.file_forensics.photorec_enabled:
            enabled["file_forensics"].append("photorec")
        
        # Cloud services
        if self.cloud.enabled:
            enabled["cloud"] = self.get_enabled_cloud_services()
        
        return enabled
    
    def get_enabled_cloud_services(self) -> List[str]:
        """Get list of enabled cloud services"""
        services = []
        if self.cloud.enabled:
            if self.cloud.virustotal_api_key:
                services.append("virustotal")
            if self.cloud.hashlookup_enabled:
                services.append("hashlookup")
            if self.cloud.nsrl_enabled:
                services.append("nsrl")
            if self.cloud.hybrid_analysis_api_key:
                services.append("hybrid_analysis")
            if self.cloud.malware_bazaar_enabled:
                services.append("malware_bazaar")
        return services
    
    def update(self, config_dict: Dict[str, Any]):
        """Update configuration from dictionary"""
        for section, data in config_dict.items():
            if hasattr(self, section):
                config_obj = getattr(self, section)
                for key, value in data.items():
                    if hasattr(config_obj, key):
                        setattr(config_obj, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        config_data = {}
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if hasattr(attr, '__dataclass_fields__'):
                config_data[attr_name] = asdict(attr)
        return config_data
    
    def get_resource_limits(self) -> Dict[str, Any]:
        """Get resource usage limits"""
        return {
            "max_cpu_workers": self.orchestrator.max_cpu_workers,
            "max_gpu_workers": self.orchestrator.max_gpu_workers,
            "max_concurrent_files": self.orchestrator.max_concurrent_files,
            "memory_limit_gb": self.orchestrator.memory_limit_gb,
            "gpu_memory_limit_mb": self.ml.gpu_memory_limit,
            "task_timeout": self.orchestrator.task_timeout
        }


def create_default_config(output_path: str = "config/default.json"):
    """Create a default configuration file"""
    config = Config()
    config.config_path = output_path
    config.save_config()
    return config