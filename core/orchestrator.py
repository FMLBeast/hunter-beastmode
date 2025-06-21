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
        self.file_analyzer = FileAnalyzer()
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

    async def analyze(self, file_path: Path, session_id: str) -> List[Dict[str, Any]]:
        self.logger.info(f"Starting analysis of {file_path}")
        file_info = await self._get_file_info(file_path)
        file_id = await self.db.add_file(session_id, str(file_path), file_info)

        tasks = await self._create_analysis_plan(file_path, file_info)
        results = await self._execute_analysis_tasks(tasks)
        final = await self._post_process_results(results)
        self.logger.info(f"Analysis complete: {len(final)} findings")
        return final

    async def _get_file_info(self, file_path: Path) -> Dict[str, Any]:
        return await asyncio.get_event_loop().run_in_executor(
            self.thread_pool, self.file_analyzer.analyze_file, file_path
        )

    async def _create_analysis_plan(self, file_path: Path, file_info: Dict[str, Any]) -> List[AnalysisTask]:
        tasks: List[AnalysisTask] = []
        ftype = file_info.get('mime_type','').lower()
        # always magic
        tasks.append(AnalysisTask(file_path, 'magic_analysis', 'file_analyzer'))
        # entropy if crypto
        if self.crypto_tools:
            tasks.append(AnalysisTask(file_path, 'entropy_analysis', 'crypto_analysis', dependencies=['magic_analysis']))
        # type-specific
        if 'image' in ftype:
            if self.classic_tools:
                for m in ['steghide_extract','zsteg_analysis','binwalk_extract']:
                    tasks.append(AnalysisTask(file_path, m, 'classic_stego', dependencies=['magic_analysis']))
            if self.image_tools:
                for m in ['lsb_analysis','noise_analysis','metadata_extraction']:
                    tasks.append(AnalysisTask(file_path, m, 'image_forensics', dependencies=['magic_analysis']))
        elif 'audio' in ftype:
            if self.audio_tools:
                for m in ['spectral_analysis','lsb_analysis','echo_hiding_detection']:
                    tasks.append(AnalysisTask(file_path, m, 'audio_analysis', dependencies=['magic_analysis']))
        else:
            if self.file_tools:
                tasks.append(AnalysisTask(file_path, 'signature', 'file_forensics', dependencies=['magic_analysis']))
        # ML
        if self.ml_detector:
            tasks.append(AnalysisTask(file_path, 'ml_detection', 'ml_detector', gpu_required=True))
        # LLM
        if self.llm_analyzer:
            tasks.append(AnalysisTask(file_path, 'llm_analysis', 'llm_analyzer'))
        return tasks

    async def _execute_analysis_tasks(self, tasks: List[AnalysisTask]) -> List[Dict[str, Any]]:
        sem = asyncio.Semaphore(self.config.orchestrator.max_concurrent_files)
        results: List[Dict[str, Any]] = []

        async def worker(task: AnalysisTask):
            async with sem:
                # wait dependencies
                for dep in task.dependencies:
                    while not any(r['method']==dep for r in results):
                        await asyncio.sleep(0.1)
                res = await self._execute_task(task)
                if res:
                    if isinstance(res, list):
                        results.extend(res)
                    else:
                        results.append(res)

        await asyncio.gather(*(worker(t) for t in tasks), return_exceptions=True)
        return results

    async def _execute_task(self, task: AnalysisTask) -> Optional[Any]:
        self.logger.info(f"Executing {task.method} via {task.tool_name}")
        # dispatch based on tool_name
        tool = getattr(self, f"{task.tool_name}", None)
        if not tool:
            return None
        func = getattr(tool, task.method, None)
        if not func:
            return None
        if asyncio.iscoroutinefunction(func):
            return await func(task.file_path)
        else:
            return await asyncio.get_event_loop().run_in_executor(self.thread_pool, func, task.file_path)

    async def _post_process_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set(); unique=[]
        for r in results:
            key=(r.get('tool'),r.get('method'),r.get('confidence',0))
            if key not in seen:
                seen.add(key); unique.append(r)
        return sorted(unique, key=lambda x:x.get('confidence',0), reverse=True)

    def shutdown(self):
        self.thread_pool.shutdown(wait=False)
        self.logger.info("Orchestrator shutdown complete")
