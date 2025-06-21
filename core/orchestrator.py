#!/usr/bin/env python3
"""
StegOrchestrator - Comprehensive orchestrator for running every analysis stage
Supports: file forensics, classic stego, image forensics, audio analysis,
crypto analysis, ML detection, LLM analysis, cascading extraction.
Proper async/await handling, GPU/CPU dispatch, dynamic tool loading,
checkpointing, concurrency limits, and post-processing pipeline.
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

# Core modules
from .file_analyzer import FileAnalyzer
from .database import DatabaseManager
from config.steg_config import Config
from utils.checkpoint import CheckpointManager

# Dynamic tool imports
_TOOL_CLASSES = []
for name in [
    'tools.file_forensics.FileForensicsTools',
    'tools.classic_stego.ClassicStegoTools',
    'tools.image_forensics.ImageForensicsTools',
    'tools.audio_analysis.AudioAnalysisTools',
    'tools.crypto_analysis.CryptoAnalysisTools',
    'ai.ml_detector.MLStegDetector',
    'ai.llm_analyzer.LLMAnalyzer',
    'tools.cascade_analyzer.CascadeAnalyzer',
]:
    module, cls = name.rsplit('.', 1)
    try:
        mod = __import__(module, fromlist=[cls])
        _TOOL_CLASSES.append(getattr(mod, cls))
    except ImportError:
        pass

@dataclass
class AnalysisTask:
    file_id: str
    tool: Any
    method: str
    args: List[Any]

class StegOrchestrator:
    def __init__(self, config: Config, *args):  # accept optional extra args
        self.config = config

        self.config = config
        self.db = DatabaseManager(config.database)
        self.logger = logging.getLogger(__name__)
        self.checkpoint = CheckpointManager(config.checkpoint, self.db)
        self.file_analyzer = FileAnalyzer(config.file_forensics)
        self.thread_pool = ThreadPoolExecutor(max_workers=config.orchestrator.max_cpu_workers)
        self.tools = [cls(config.__dict__.get(cls.__name__.lower(), {}))
                      for cls in _TOOL_CLASSES]
        self.logger.info("StegOrchestrator initialized with %d tools", len(self.tools))

    async def analyze_path(self, path: Path, session_id: str):
        """Kick off analysis for a directory or file."""
        if path.is_dir():
            for child in path.rglob('*'):
                await self._analyze_file(child, session_id)
        else:
            await self._analyze_file(path, session_id)

    async def _analyze_file(self, path: Path, session_id: str):
        self.logger.debug("Analyzing %s", path)
        file_info = await asyncio.get_event_loop().run_in_executor(
            self.thread_pool, self.file_analyzer.extract, path)
        file_id = await self.db.insert_file(session_id, file_info)
        tasks = self._build_tasks(file_id, path)
        await self._execute_tasks(tasks)

    def _build_tasks(self, file_id: str, path: Path) -> List[AnalysisTask]:
        tasks: List[AnalysisTask] = []
        for tool in self.tools:
            for method in getattr(tool, 'analysis_methods', []):
                tasks.append(AnalysisTask(file_id, tool, method, [path]))
        return tasks

    async def _execute_tasks(self, tasks: List[AnalysisTask]):
        sem = asyncio.Semaphore(self.config.orchestrator.max_concurrent_files)
        async def run_task(t: AnalysisTask):
            async with sem:
                try:
                    func = getattr(t.tool, t.method)
                    if asyncio.iscoroutinefunction(func):
                        res = await func(*t.args)
                    else:
                        res = await asyncio.get_event_loop().run_in_executor(
                            self.thread_pool, func, *t.args)
                    await self.db.insert_findings(t.file_id, res)
                except Exception as e:
                    self.logger.exception("Error in %s.%s", t.tool.__class__.__name__, t.method)
        await asyncio.gather(*(run_task(t) for t in tasks))

    def shutdown(self):
        self.thread_pool.shutdown(wait=False)
        self.db.close()
        self.logger.info("StegOrchestrator shutdown complete")
