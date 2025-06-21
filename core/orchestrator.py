#!/usr/bin/env python3
"""
Robust Orchestrator for StegAnalyzer
Handles async/await correctly, integrates checkpointing, and tolerates extra init args.
Provides unified `analyze` entrypoint.
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
    Provides an `analyze` method to process a directory or single file.
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

    async def _run_task(self, task: AnalysisTask):
        """Executes a single AnalysisTask, logs results."""
        # Dispatch based on method/tool
        try:
            method = getattr(self, f"_{task.method}")
            result = await method(task)
            self.db.save_finding(task.file_path, task.tool_name, result)
        except Exception as e:
            self.logger.error(f"Task {task.tool_name}.{task.method} failed: {e}")

    async def analyze(self, target: Path):
        """Entry point: analyze a file or directory."""
        paths = [target] if target.is_file() else list(target.rglob('*'))
        tasks: List[AnalysisTask] = []
        for p in paths:
            if p.is_file():
                tasks.extend(self._schedule_file_tasks(p))

        # Run all tasks with concurrency control
        await asyncio.gather(*(self._run_task(t) for t in tasks))

    def _schedule_file_tasks(self, file_path: Path) -> List[AnalysisTask]:
        """Generate AnalysisTask instances for each enabled tool on a file."""
        tasks = []
        # File forensics
        if self.file_tools:
            tasks.append(AnalysisTask(file_path, 'file_forensics', 'file_tools'))
        # Classic stego
        if self.classic_tools:
            tasks.append(AnalysisTask(file_path, 'classic_stego', 'classic_tools'))
        # Image forensic
        if self.image_tools:
            tasks.append(AnalysisTask(file_path, 'image_forensics', 'image_tools'))
        # Audio analysis
        if self.audio_tools:
            tasks.append(AnalysisTask(file_path, 'audio_analysis', 'audio_tools'))
        # Crypto analysis
        if self.crypto_tools:
            tasks.append(AnalysisTask(file_path, 'crypto_analysis', 'crypto_tools'))
        # ML detection
        if self.ml_detector:
            tasks.append(AnalysisTask(file_path, 'ml_detection', 'ml_detector', gpu_required=True))
        # LLM analysis
        if self.llm_analyzer:
            tasks.append(AnalysisTask(file_path, 'llm_analysis', 'llm_analyzer'))
        # Cascade
        if self.cascade_analyzer:
            tasks.append(AnalysisTask(file_path, 'cascade_analysis', 'cascade_analyzer'))
        return tasks

    # Task implementations
    async def _file_forensics(self, task: AnalysisTask) -> Any:
        return await asyncio.get_event_loop().run_in_executor(
            self.thread_pool,
            self.file_tools.run, str(task.file_path)
        )

    async def _classic_stego(self, task: AnalysisTask) -> Any:
        return await asyncio.get_event_loop().run_in_executor(
            self.thread_pool,
            self.classic_tools.run, str(task.file_path)
        )

    async def _image_forensics(self, task: AnalysisTask) -> Any:
        return await asyncio.get_event_loop().run_in_executor(
            self.thread_pool,
            self.image_tools.run, str(task.file_path)
        )

    async def _audio_analysis(self, task: AnalysisTask) -> Any:
        return await asyncio.get_event_loop().run_in_executor(
            self.thread_pool,
            self.audio_tools.run, str(task.file_path)
        )

    async def _crypto_analysis(self, task: AnalysisTask) -> Any:
        return await asyncio.get_event_loop().run_in_executor(
            self.thread_pool,
            self.crypto_tools.run, str(task.file_path)
        )

    async def _ml_detection(self, task: AnalysisTask) -> Any:
        return await self.ml_detector.detect(str(task.file_path))

    async def _llm_analysis(self, task: AnalysisTask) -> Any:
        return await self.llm_analyzer.analyze(str(task.file_path))

    async def _cascade_analysis(self, task: AnalysisTask) -> Any:
        return await self.cascade_analyzer.analyze(str(task.file_path))

# End of orchestrator.py
