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
    dependencies: List[str] = None  # methods to wait for
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
        # Accept extra args from CLI layer without error
        self.config = config
        self.db = database
        self.logger = logging.getLogger(__name__)

        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(getattr(config, 'orchestrator', {}))

        # Thread pool for blocking tasks
        self.thread_pool = ThreadPoolExecutor(max_workers=getattr(config.orchestrator, 'max_cpu_workers', 4))

        # Analyzer and tools
        self.file_analyzer = FileAnalyzer(getattr(self.config, 'file_forensics', {}))
        self._initialize_tools()
        self.logger.info("StegOrchestrator initialized")

    def _initialize_tools(self):
        def init_tool(ToolClass, cfg_attr, name, log_success=True):
            try:
                cfg = getattr(self.config, cfg_attr, {})
                tool = ToolClass(cfg) if ToolClass else None
                if tool and log_success:
                    self.logger.info(f"{name} initialized")
                return tool
            except Exception as e:
                self.logger.warning(f"{name} init failed: {e}")
                return None

        self.file_tools = init_tool(FileForensicsTools, 'file_forensics', 'File forensics tools')
        self.classic_tools = init_tool(ClassicStegoTools, 'classic_stego', 'Classic stego tools')
        self.image_tools = init_tool(ImageForensicsTools, 'image_forensics', 'Image forensics tools')
        self.audio_tools = init_tool(AudioAnalysisTools, 'audio_analysis', 'Audio analysis tools')
        self.crypto_tools = init_tool(CryptoAnalysisTools, 'crypto', 'Crypto analysis tools')

        try:
            if MLStegDetector and getattr(self.config.ml, 'enabled', False):
                self.ml_detector = MLStegDetector(self.config.ml)
                self.logger.info("ML detector initialized")
            else:
                self.ml_detector = None
        except Exception as e:
            self.ml_detector = None
            self.logger.error(f"ML detector init failed: {e}")

        try:
            if LLMAnalyzer and getattr(self.config.llm, 'enabled', False):
                self.llm_analyzer = LLMAnalyzer(self.config.llm)
                self.logger.info("LLM analyzer initialized")
            else:
                self.llm_analyzer = None
        except Exception as e:
            self.llm_analyzer = None
            self.logger.warning(f"LLM analyzer init failed: {e}")

        self.cascade_analyzer = init_tool(CascadeAnalyzer, 'orchestrator', 'Cascade analyzer', log_success=False)

    def _get_sync_method(self, tool, names=('run','analyze','execute')):
        for nm in names:
            if hasattr(tool, nm):
                return getattr(tool, nm)
        raise AttributeError(f"No sync entrypoint found on {tool}")

    async def _invoke_sync(self, tool, file_path: Path):
        method = self._get_sync_method(tool)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.thread_pool, method, str(file_path))

    async def _invoke_async(self, tool, file_path: Path):
        # detect coroutine or async method
        for nm in ('analyze','detect','execute'):
            if hasattr(tool, nm):
                fn = getattr(tool, nm)
                if asyncio.iscoroutinefunction(fn):
                    return await fn(str(file_path))
                else:
                    # wrap sync as thread
                    return await self._invoke_sync(tool, file_path)
        raise AttributeError(f"No async entrypoint found on {tool}")

    async def _run_task(self, task: AnalysisTask, session_id: Any = None):
        try:
            tool = getattr(self, task.tool_name)
            if task.tool_name in ('ml_detector','llm_analyzer','cascade_analyzer'):
                result = await self._invoke_async(tool, task.file_path)
            else:
                result = await self._invoke_sync(tool, task.file_path)
            try:
                self.db.save_finding(task.file_path, task.tool_name, result)
            except TypeError:
                self.db.save_finding(session_id, task.file_path, task.tool_name, result)
        except Exception as e:
            self.logger.error(f"Task {task.tool_name}.{task.method} failed: {e}")

    async def analyze(self, target: Path, *args):
        session_id = args[0] if args else None
        targets = [target] if target.is_file() else list(target.rglob('*'))
        tasks = [t for p in targets if p.is_file() for t in self._schedule_file_tasks(p)]
        await asyncio.gather(*(self._run_task(t, session_id) for t in tasks))

    def _schedule_file_tasks(self, file_path: Path) -> List[AnalysisTask]:
        tasks=[]
        if self.file_tools:    tasks.append(AnalysisTask(file_path,'file_forensics','file_tools'))
        if self.classic_tools: tasks.append(AnalysisTask(file_path,'classic_stego','classic_tools'))
        if self.image_tools:   tasks.append(AnalysisTask(file_path,'image_forensics','image_tools'))
        if self.audio_tools:   tasks.append(AnalysisTask(file_path,'audio_analysis','audio_tools'))
        if self.crypto_tools:  tasks.append(AnalysisTask(file_path,'crypto_analysis','crypto_tools'))
        if self.ml_detector:   tasks.append(AnalysisTask(file_path,'ml_detection','ml_detector',gpu_required=True))
        if self.llm_analyzer:  tasks.append(AnalysisTask(file_path,'llm_analysis','llm_analyzer'))
        if self.cascade_analyzer: tasks.append(AnalysisTask(file_path,'cascade_analysis','cascade_analyzer'))
        return tasks
