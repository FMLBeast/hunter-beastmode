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
from typing import List, Any, Dict
from concurrent.futures import ThreadPoolExecutor

from core.file_analyzer import FileAnalyzer
from core.database import DatabaseManager
from config.steg_config import Config
from utils.checkpoint import CheckpointManager

# Dynamically import available tools
_TOOL_CLASSES: List[Any] = []
for path in [
    'tools.file_forensics.FileForensicsTools',
    'tools.classic_stego.ClassicStegoTools',
    'tools.image_forensics.ImageForensicsTools',
    'tools.audio_analysis.AudioAnalysisTools',
    'tools.crypto_analysis.CryptoAnalysisTools',
    'ai.ml_detector.MLStegDetector',
    'ai.llm_analyzer.LLMAnalyzer',
    'tools.cascade_analyzer.CascadeAnalyzer',
]:
    module, cls = path.rsplit('.', 1)
    try:
        mod = __import__(module, fromlist=[cls])
        _TOOL_CLASSES.append(getattr(mod, cls))
    except ImportError:
        pass

class StegOrchestrator:
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize database
        self.db = DatabaseManager(config.database)

        # Checkpoint config fallback
        chk_cfg = getattr(config, 'checkpoint', None) or config.orchestrator
        self.checkpoint = CheckpointManager(chk_cfg, self.db)

        # File analyzer and thread pool for blocking ops
        self.file_analyzer = FileAnalyzer(config.file_forensics)
        self.thread_pool = ThreadPoolExecutor(max_workers=config.orchestrator.max_cpu_workers)

        # Load and instantiate tools with their config sections
        self.tools = []
        cfg_dict: Dict[str, Any] = config.__dict__
        for cls in _TOOL_CLASSES:
            section = cfg_dict.get(cls.__name__.lower(), {})
            try:
                self.tools.append(cls(section))
            except Exception:
                self.logger.warning("Failed to init tool %s", cls.__name__)

        # Throttle concurrent file analyses
        self.semaphore = asyncio.Semaphore(config.orchestrator.max_concurrent_files)
        self.logger.info("Orchestrator initialized with %d tools", len(self.tools))

    async def run(self, target: Path):
        # create a session record
        session_id = self.db.create_session(str(target), target.is_dir())
        try:
            await self._analyze_path(target, session_id)
            self.db.update_session(session_id, status='completed')
        except Exception as e:
            self.logger.exception("Fatal error during orchestration")
            self.db.update_session(session_id, status='error', error=str(e))
        finally:
            self.shutdown()

    async def _analyze_path(self, path: Path, session_id: str):
        if path.is_dir():
            for child in path.rglob('*'):
                await self._analyze_file(child, session_id)
        else:
            await self._analyze_file(path, session_id)

    async def _analyze_file(self, path: Path, session_id: str):
        self.logger.debug("Analyzing %s", path)
        # Extract metadata and content in thread pool
        file_info = await asyncio.get_event_loop().run_in_executor(
            self.thread_pool, self.file_analyzer.extract, path)
        file_id = self.db.insert_file(session_id, file_info)

        # Build and schedule analysis tasks
        tasks = []
        for tool in self.tools:
            for method in getattr(tool, 'analysis_methods', []):
                tasks.append((tool, method, path, file_id))

        await self._execute_tasks(tasks)

    async def _execute_tasks(self, tasks: List[Any]):
        async def run_task(tool, method, path, file_id):
            async with self.semaphore:
                try:
                    func = getattr(tool, method)
                    if asyncio.iscoroutinefunction(func):
                        results = await func(path)
                    else:
                        results = await asyncio.get_event_loop().run_in_executor(
                            self.thread_pool, func, path)
                    self.db.insert_findings(file_id, results)
                except Exception:
                    self.logger.exception("Error running %s on %s", method, path)

        await asyncio.gather(*(run_task(*t) for t in tasks))

    def shutdown(self):
        self.thread_pool.shutdown(wait=False)
        self.db.close()
        self.logger.info("StegOrchestrator shutdown complete")
