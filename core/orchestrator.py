#!/usr/bin/env python3
"""
Robust Orchestrator for StegAnalyzer
Handles async/await correctly, integrates checkpointing, and tolerates extra init args.
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

# Core imports
from .file_analyzer import FileAnalyzer
from .database import DatabaseManager
from utils.checkpoint import CheckpointManager

# Optional tool imports
try:
    from tools.file_forensics import FileForensicsTools
except ImportError:
    FileForensicsTools = None

try:
    from tools.classic_stego import ClassicStegoTools
except ImportError:
    ClassicStegoTools = None

try:
    from tools.image_forensics import ImageForensicsTools
except ImportError:
    ImageForensicsTools = None

try:
    from tools.audio_analysis import AudioAnalysisTools
except ImportError:
    AudioAnalysisTools = None

try:
    from tools.crypto_analysis import CryptoAnalysisTools
except ImportError:
    CryptoAnalysisTools = None

try:
    from ai.ml_detector import MLStegDetector
except ImportError:
    MLStegDetector = None

try:
    from ai.llm_analyzer import LLMAnalyzer
except ImportError:
    LLMAnalyzer = None

try:
    from tools.cascade_analyzer import CascadeAnalyzer
except ImportError:
    CascadeAnalyzer = None

@dataclass
class AnalysisTask:
    file_path: Path
    method: str
    tool_name: str
    priority: int = 1
    dependencies: List[str] = None
    gpu_required: bool = False
    estimated_time: float = 1.0

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []

class StegOrchestrator:
    """
    Orchestrates analysis tools over files/directories.
    Accepts a Config and a DatabaseManager instance (extra args ignored).
    """

    def __init__(self, config, database: DatabaseManager, *args, **kwargs):
        # Accept extra args from CLI layer without error
        self.config = config
        self.db = database
        self.logger = logging.getLogger(__name__)

        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(config.orchestrator)

        # Thread pool for blocking tasks
        self.thread_pool = ThreadPoolExecutor(max_workers=config.orchestrator.max_cpu_workers)

        # Analyzer and tools
        # Pass file_forensics config into FileAnalyzer to satisfy its signature
        self.file_analyzer = FileAnalyzer(self.config.file_forensics)
        self._initialize_tools()
        self.logger.info("StegOrchestrator initialized")

    def _initialize_tools(self):
        # Each tool block safely initializes or logs failure
        try:
            self.file_tools = FileForensicsTools(self.config.file_forensics) if FileForensicsTools else None
            self.logger.info("File forensics tools initialized")
        except Exception as e:
            self.file_tools = None
            self.logger.error(f"File forensics init failed: {e}")

        try:
            self.classic_tools = ClassicStegoTools(self.config.classic_stego) if ClassicStegoTools else None
            self.logger.info("Classic stego tools initialized")
        except Exception as e:
            self.classic_tools = None
            self.logger.warning(f"Classic stego init failed: {e}")

        try:
            self.image_tools = ImageForensicsTools(self.config.image_forensics) if ImageForensicsTools else None
            self.logger.info("Image forensics tools initialized")
        except Exception as e:
            self.image_tools = None
            self.logger.warning(f"Image forensics init failed: {e}")

        try:
            self.audio_tools = AudioAnalysisTools(self.config.audio_analysis) if AudioAnalysisTools else None
            self.logger.info("Audio analysis tools initialized")
        except Exception as e:
            self.audio_tools = None
            self.logger.warning(f"Audio analysis init failed: {e}")

        try:
            self.crypto_tools = CryptoAnalysisTools(self.config.crypto) if CryptoAnalysisTools else None
            self.logger.info("Crypto analysis tools initialized")
        except Exception as e:
            self.crypto_tools = None
            self.logger.warning(f"Crypto analysis init failed: {e}")

        try:
            self.ml_detector = MLStegDetector(self.config.ml) if MLStegDetector and getattr(self.config.ml, 'enabled', False) else None
            self.logger.info("ML detector initialized")
        except Exception as e:
            self.ml_detector = None
            self.logger.error(f"ML detector init failed: {e}")

        try:
            self.llm_analyzer = LLMAnalyzer(self.config.llm) if LLMAnalyzer and getattr(self.config.llm, 'enabled', False) else None
            self.logger.info("LLM analyzer initialized")
        except Exception as e:
            self.llm_analyzer = None
            self.logger.warning(f"LLM analyzer init failed: {e}")

        try:
            self.cascade_analyzer = CascadeAnalyzer(self.config) if CascadeAnalyzer else None
            self.logger.info("Cascade analyzer initialized")
        except Exception as e:
            self.cascade_analyzer = None
            self.logger.warning(f"Cascade analyzer init failed: {e}")

    # rest of file unchanged...
    async def analyze_file(self, file_path: Path, method: str, tool_name: str, priority: int = 1,
                           dependencies: Optional[List[str]] = None, gpu_required: bool = False,
                           estimated_time: float = 1.0) -> Dict[str, Any]:
        """
        Analyze a single file with the specified method and tool.
        """
        task = AnalysisTask(file_path=file_path, method=method, tool_name=tool_name, priority=priority,
                            dependencies=dependencies or [], gpu_required=gpu_required,
                            estimated_time=estimated_time)

        # Checkpointing logic
        if self.checkpoint_manager.is_checkpointed(task):
            self.logger.info(f"Skipping already completed task for {file_path}")
            return self.checkpoint_manager.get_checkpoint(task)

        # Run analysis in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(self.thread_pool, self._run_analysis_task, task)

        # Save checkpoint
        self.checkpoint_manager.save_checkpoint(task, result)
        return result