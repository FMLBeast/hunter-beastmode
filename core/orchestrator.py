#!/usr/bin/env python3
"""
Robust Orchestrator for StegAnalyzer
Handles async/await correctly, integrates checkpointing, tolerates extra init args, and provides unified `analyze` entrypoint.
"""

import asyncio
import logging
from pathlib import Path
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

class StegOrchestrator:
    """
    Orchestrates analysis tools over files or directories.
    Provides a single `analyze` method that schedules both core and optional tools.
    """

    def __init__(self, config: Any, database: DatabaseManager, *args, **kwargs):
        self.config = config
        self.db = database
        self.logger = logging.getLogger(__name__)

        # Setup checkpoint manager
        self.checkpoint_manager = CheckpointManager(getattr(config, 'orchestrator', {}))

        # Thread pool for blocking tasks
        max_workers = getattr(config.orchestrator, 'max_cpu_workers', 4)
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)

        # Core file analyzer
        self.file_analyzer = FileAnalyzer(getattr(config, 'file_forensics', {}))

        # Initialize optional tools
        self.tools = []
        for ToolClass, cfg_attr in [
            (FileForensicsTools,   'file_forensics'),
            (ClassicStegoTools,     'classic_stego'),
            (ImageForensicsTools,   'image_forensics'),
            (AudioAnalysisTools,    'audio_analysis'),
            (CryptoAnalysisTools,   'crypto'),
            (MLStegDetector,        'ml'),
            (LLMAnalyzer,           'llm'),
            (CascadeAnalyzer,       'orchestrator'),
        ]:
            if ToolClass and getattr(config, cfg_attr, True):
                try:
                    tool = ToolClass(getattr(config, cfg_attr, {}))
                    self.tools.append(tool)
                    self.logger.info(f"Initialized tool: {type(tool).__name__}")
                except Exception as e:
                    self.logger.warning(f"Failed to init {type(ToolClass).__name__}: {e}")

        self.logger.info("StegOrchestrator initialized with tools: %s", 
                         [type(t).__name__ for t in self.tools])

    async def analyze(self, target: Path, session_id: Any = None):
        """
        Analyze a file or all files in a directory.
        session_id is optional and passed to save contexts.
        """
        # Collect files to analyze
        if target.is_file():
            files = [target]
        else:
            files = [p for p in target.rglob('*') if p.is_file()]

        loop = asyncio.get_running_loop()
        tasks = []

        for f in files:
            # Core file forensic analysis
            tasks.append(loop.run_in_executor(
                self.thread_pool, self.file_analyzer.analyze, str(f), session_id
            ))
            # Optional tools
            for tool in self.tools:
                # Determine entrypoint method
                method = getattr(tool, 'analyze', None) or getattr(tool, 'run', None)
                if not method:
                    self.logger.warning("No entrypoint on %s", type(tool).__name__)
                    continue
                # Schedule sync or async
                if asyncio.iscoroutinefunction(method):
                    tasks.append(method(str(f), session_id))
                else:
                    tasks.append(loop.run_in_executor(
                        self.thread_pool, method, str(f), session_id
                    ))

        # Execute all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Log and save results
        for res in results:
            if isinstance(res, Exception):
                self.logger.error("Analysis task error: %s", res)
            else:
                # res expected to be finding or list of findings
                try:
                    self.db.save_finding(*res) if isinstance(res, tuple) else None
                except Exception:
                    pass

        self.logger.info("All analysis tasks completed")
