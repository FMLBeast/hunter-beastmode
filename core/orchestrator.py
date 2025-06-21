#!/usr/bin/env python3
"""
Robust Orchestrator for StegAnalyzer
Handles async/await correctly, integrates checkpointing, tolerates extra init args, and accepts optional session_id.
Provides unified `analyze` entrypoint.
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Any
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

    def __init__(self, config: Any, database: DatabaseManager, *args, **kwargs):
        self.config = config
        self.db = database
        self.logger = logging.getLogger(__name__)

        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(getattr(config, 'orchestrator', {}))

        # Thread pool for blocking tasks
        max_workers = getattr(config.orchestrator, 'max_cpu_workers', 4)
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)

        # Analyzer and tools
        self.file_analyzer = FileAnalyzer(getattr(self.config, 'file_forensics', {}))
        self._initialize_tools()
        self.logger.info("StegOrchestrator initialized")

    def _initialize_tools(self):
        def init_tool(ToolClass, cfg_attr, name):
            try:
                cfg = getattr(self.config, cfg_attr, {})
                tool = ToolClass(cfg) if ToolClass else None
                if tool:
                    self.logger.info(f"{name} initialized")
                return tool
            except Exception as e:
                self.logger.warning(f"{name} init failed: {e}")
                return None

        self.file_tools    = init_tool(FileForensicsTools,   'file_forensics',   'File forensics tools')
        self.classic_tools = init_tool(ClassicStegoTools,     'classic_stego',    'Classic stego tools')
        self.image_tools   = init_tool(ImageForensicsTools,   'image_forensics',  'Image forensics tools')
        self.audio_tools   = init_tool(AudioAnalysisTools,    'audio_analysis',   'Audio analysis tools')
        self.crypto_tools  = init_tool(CryptoAnalysisTools,   'crypto',           'Crypto analysis tools')

        if MLStegDetector and getattr(self.config.ml, 'enabled', False):
            try:
                self.ml_detector = MLStegDetector(self.config.ml)
                self.logger.info("ML detector initialized")
            except Exception as e:
                self.ml_detector = None
                self.logger.error(f"ML detector init failed: {e}")
        else:
            self.ml_detector = None

        if LLMAnalyzer and getattr(self.config.llm, 'enabled', False):
            try:
                self.llm_analyzer = LLMAnalyzer(self.config.llm)
                self.logger.info("LLM analyzer initialized")
            except Exception as e:
                self.llm_analyzer = None
                self.logger.warning(f"LLM analyzer init failed: {e}")
        else:
            self.llm_analyzer = None

        self.cascade_analyzer = init_tool(CascadeAnalyzer, 'orchestrator', 'Cascade analyzer')

    async def _execute_task(self, task: AnalysisTask, session_id: Any = None):
        tool = getattr(self, task.tool_name)
        if not tool:
            self.logger.error(f"Tool '{task.tool_name}' not available")
            return

        # pick the method on the tool
        method = getattr(tool, task.method, None)
        if not method:
            self.logger.error(f"Method '{task.method}' not found on tool '{task.tool_name}'")
            return

        try:
            if asyncio.iscoroutinefunction(method):
                result = await method(str(task.file_path))
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(self.thread_pool, method, str(task.file_path))

            # save findings
            try:
                self.db.save_finding(task.file_path, task.tool_name, result)
            except TypeError:
                self.db.save_finding(session_id, task.file_path, task.tool_name, result)

        except Exception as e:
            self.logger.error(f"Task {task.tool_name}.{task.method} failed: {e}")

    async def analyze(self, target: Path, *args):
        session_id = args[0] if args else None
        files = [target] if target.is_file() else [p for p in target.rglob('*') if p.is_file()]
        tasks = []
        for f in files:
            tasks.extend(self._schedule_file_tasks(f))

        await asyncio.gather(*(self._execute_task(t, session_id) for t in tasks))

    def _schedule_file_tasks(self, file_path: Path) -> List[AnalysisTask]:
        tasks = []
        mapping = [
            ('file_forensics',   'file_tools'),
            ('classic_stego',    'classic_tools'),
            ('image_forensics',  'image_tools'),
            ('audio_analysis',   'audio_tools'),
            ('crypto_analysis',  'crypto_tools'),
            ('ml_detection',     'ml_detector'),
            ('llm_analysis',     'llm_analyzer'),
            ('cascade_analysis', 'cascade_analyzer'),
        ]
        for method, tool_name in mapping:
            tool = getattr(self, tool_name)
            if tool:
                tasks.append(AnalysisTask(file_path, method, tool_name,
                                          gpu_required=(tool_name=='ml_detector')))
        return tasks
