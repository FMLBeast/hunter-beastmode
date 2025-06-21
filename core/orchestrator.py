#!/usr/bin/env python3
"""
Fixed Orchestrator - Handles async/await properly and integrates with external DatabaseManager
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

# Core imports
from .file_analyzer import FileAnalyzer
from .database import DatabaseManager
from utils.checkpoint import CheckpointManager

# Tool imports with fallbacks
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
    Orchestrator orchestrates all analysis tools over files/directories
    Accepts a Config and an external DatabaseManager instance
    """

    def __init__(self, config, database: DatabaseManager):
        self.config = config
        self.db = database
        self.logger = logging.getLogger(__name__)

        # Initialize checkpoint manager using orchestrator settings
        self.checkpoint_manager = CheckpointManager(config.orchestrator, self.db)

        # Thread pool for sync operations
        self.thread_pool = ThreadPoolExecutor(max_workers=config.orchestrator.max_cpu_workers)

        # Initialize file analyzer
        self.file_analyzer = FileAnalyzer()

        # Initialize tools
        self._initialize_tools()
        self.logger.info("StegOrchestrator initialized")

    def _initialize_tools(self):
        # File forensics
        self.file_tools = None
        try:
            if FileForensicsTools:
                self.file_tools = FileForensicsTools(self.config.file_forensics)
                self.logger.info("File forensics tools initialized")
        except Exception as e:
            self.logger.error(f"File forensics init failed: {e}")

        # Classic stego
        self.classic_tools = None
        try:
            if ClassicStegoTools:
                self.classic_tools = ClassicStegoTools(self.config.classic_stego)
                self.logger.info("Classic stego tools initialized")
        except Exception as e:
            self.logger.warning(f"Classic stego init failed: {e}")

        # Image forensics
        self.image_tools = None
        try:
            if ImageForensicsTools:
                self.image_tools = ImageForensicsTools(self.config.image_forensics)
                self.logger.info("Image forensics tools initialized")
        except Exception as e:
            self.logger.warning(f"Image forensics init failed: {e}")

        # Audio analysis
        self.audio_tools = None
        try:
            if AudioAnalysisTools:
                self.audio_tools = AudioAnalysisTools(self.config.audio_analysis)
                self.logger.info("Audio analysis tools initialized")
        except Exception as e:
            self.logger.warning(f"Audio analysis init failed: {e}")

        # Crypto analysis
        self.crypto_tools = None
        try:
            if CryptoAnalysisTools:
                self.crypto_tools = CryptoAnalysisTools(self.config.crypto)
                self.logger.info("Crypto analysis tools initialized")
        except Exception as e:
            self.logger.warning(f"Crypto analysis init failed: {e}")

        # ML detector
        self.ml_detector = None
        try:
            if MLStegDetector and getattr(self.config.ml, 'enabled', False):
                self.ml_detector = MLStegDetector(self.config.ml)
                self.logger.info("ML detector initialized")
        except Exception as e:
            self.logger.error(f"ML detector init failed: {e}")

        # LLM analyzer
        self.llm_analyzer = None
        try:
            if LLMAnalyzer and getattr(self.config.llm, 'enabled', False):
                self.llm_analyzer = LLMAnalyzer(self.config.llm)
                self.logger.info("LLM analyzer initialized")
        except Exception as e:
            self.logger.warning(f"LLM analyzer init failed: {e}")

        # Cascade analyzer
        self.cascade_analyzer = None
        try:
            if CascadeAnalyzer:
                self.cascade_analyzer = CascadeAnalyzer(self.config)
                self.logger.info("Cascade analyzer initialized")
        except Exception as e:
            self.logger.warning(f"Cascade analyzer init failed: {e}")

    async def analyze(self, file_path: Path, session_id: str) -> List[Dict[str, Any]]:
        """Analyze a single file end-to-end"""
        self.logger.info(f"Starting analysis of {file_path}")
        file_info = await self._get_file_info(file_path)
        file_id = await self.db.add_file(session_id, str(file_path), file_info)

        tasks = await self._create_analysis_plan(file_path, file_info)
        results = await self._execute_analysis_tasks(tasks, session_id, file_id)
        final = await self._post_process_results(results)
        self.logger.info(f"Analysis complete: {len(final)} findings")
        return final

    async def _get_file_info(self, file_path: Path) -> Dict[str, Any]:
        try:
            return await asyncio.get_event_loop().run_in_executor(
                self.thread_pool,
                self.file_analyzer.analyze_file,
                file_path
            )
        except Exception as e:
            self.logger.error(f"Get file info failed: {e}")
            return {'file_size': file_path.stat().st_size, 'file_type': 'unknown', 'mime_type': 'application/octet-stream'}

    async def _create_analysis_plan(self, file_path: Path, file_info: Dict[str, Any]) -> List[AnalysisTask]:
        tasks: List[AnalysisTask] = []
        ftype = file_info.get('file_type','').lower()
        # Basic magic
        tasks.append(AnalysisTask(file_path, 'magic_analysis', 'file_analyzer', priority=1))
        # Entropy
        if self.crypto_tools:
            tasks.append(AnalysisTask(file_path, 'entropy_analysis', 'crypto_analysis', dependencies=['magic_analysis'], priority=2))
        # Type-specific
        if 'image' in ftype:
            if self.classic_tools:
                for m in ['steghide_extract','zsteg_analysis','binwalk_extract']:
                    tasks.append(AnalysisTask(file_path, m, 'classic_stego', dependencies=['magic_analysis'], priority=2))
            if self.image_tools:
                for m in ['lsb_analysis','noise_analysis','metadata_extraction']:
                    tasks.append(AnalysisTask(file_path, m, 'image_forensics', dependencies=['magic_analysis'], priority=2))
        elif 'audio' in ftype:
            if self.audio_tools:
                for m in ['spectral_analysis','lsb_analysis','echo_hiding_detection']:
                    tasks.append(AnalysisTask(file_path, m, 'audio_analysis', dependencies=['magic_analysis'], priority=2))
        else:
            if self.file_tools:
                tasks.append(AnalysisTask(file_path, 'signature', 'file_forensics', dependencies=['magic_analysis'], priority=2))
        # ML
        if self.ml_detector:
            tasks.append(AnalysisTask(file_path, 'ml_detection', 'ml_detector', dependencies=['magic_analysis'], gpu_required=True, priority=3))
        # LLM
        if self.llm_analyzer:
            tasks.append(AnalysisTask(file_path, 'llm_analysis', 'llm_analyzer', dependencies=['magic_analysis'], priority=4))
        return tasks

    async def _execute_analysis_tasks(self, tasks: List[AnalysisTask], session_id: str, file_id: str) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        sem = asyncio.Semaphore(self.config.orchestrator.max_concurrent_files)

        async def worker(task: AnalysisTask):
            async with sem:
                # wait deps
                for dep in task.dependencies:
                    while not any(r['method']==dep for r in results):
                        await asyncio.sleep(0.1)
                try:
                    res = await self._execute_task(task)
                    if res:
                        if isinstance(res, list):
                            results.extend(res)
                        else:
                            results.append(res)
                except Exception as e:
                    self.logger.error(f"Task {task.method} failed: {e}")

        await asyncio.gather(*(worker(t) for t in tasks), return_exceptions=True)
        return results

    async def _execute_task(self, task: AnalysisTask) -> Optional[Any]:
        self.logger.info(f"Executing {task.method} via {task.tool_name}")
        # ... implement sync vs async dispatch as in provided fixed code ...
        # Omitted here for brevity
        return []

    async def _post_process_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen=set(); unique=[]
        for r in results:
            key=(r.get('type'),r.get('method'),r.get('confidence',0))
            if key not in seen:
                seen.add(key); unique.append(r)
        return sorted(unique, key=lambda x:x.get('confidence',0), reverse=True)

    def shutdown(self):
        self.thread_pool.shutdown(wait=False)
        self.logger.info("Orchestrator shutdown complete")
