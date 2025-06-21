#!/usr/bin/env python3
"""
Robust Orchestrator for StegAnalyzer
Handles sync/async tools, unified analyze entrypoint, and result storage.
"""

import asyncio
import logging
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Callable, Optional

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


def _get_tool_method(tool: Any) -> Optional[Callable]:
    """
    Find a supported entrypoint on a tool instance.
    """
    for name in ('analyze', 'run', 'execute', 'execute_method', 'cascade_analyze'):
        fn = getattr(tool, name, None)
        if callable(fn):
            return fn
    return None


def _invoke_tool(method: Callable, file_path: Path, session_id: Any) -> Any:
    """
    Invoke a synchronous tool method, passing session if supported.
    """
    try:
        return method(file_path, session_id)
    except TypeError:
        return method(file_path)


class StegOrchestrator:
    """
    Orchestrates analysis across files, using configured tools and database.
    Provides a single `analyze(target)` method.
    """

    def __init__(self, config: Any, database: DatabaseManager):
        self.config = config
        self.db = database
        self.logger = logging.getLogger(__name__)

        # Optional checkpoint manager
        try:
            self.checkpoint = CheckpointManager(getattr(config, 'orchestrator', {}), database)
        except Exception:
            self.checkpoint = None

        # Thread pool for blocking/sync tools
        max_workers = getattr(config.orchestrator, 'max_cpu_workers', 4)
        self.pool = ThreadPoolExecutor(max_workers=max_workers)

        # Core file analyzer
        fa_cfg = getattr(config, 'file_forensics', getattr(config, 'file_analyzer', {}))
        self.file_analyzer = FileAnalyzer(fa_cfg)

        # Initialize optional tools
        self.tools: List[Any] = []
        def _init_tool(cls, cfg_attr: str, name: str):
            cfg = getattr(config, cfg_attr, None)
            if cls and cfg is not None:
                try:
                    inst = cls(cfg)
                    self.tools.append(inst)
                    self.logger.info(f"Initialized tool: {name}")
                except Exception as e:
                    self.logger.warning(f"Failed to init {name}: {e}")

        _init_tool(FileForensicsTools, 'file_forensics', 'FileForensicsTools')
        _init_tool(ClassicStegoTools, 'classic_stego', 'ClassicStegoTools')
        _init_tool(ImageForensicsTools, 'image_forensics', 'ImageForensicsTools')
        _init_tool(AudioAnalysisTools, 'audio_analysis', 'AudioAnalysisTools')
        _init_tool(CryptoAnalysisTools, 'crypto_analysis', 'CryptoAnalysisTools')
        _init_tool(MLStegDetector, 'ml', 'MLStegDetector')
        _init_tool(LLMAnalyzer, 'llm', 'LLMAnalyzer')
        if CascadeAnalyzer:
            cascade_cfg = getattr(config, 'cascade', None)
            if cascade_cfg is None:
                cascade_cfg = getattr(config, 'orchestrator', {})
            try:
                inst = CascadeAnalyzer(cascade_cfg)
                self.tools.append(inst)
                self.logger.info("Initialized tool: CascadeAnalyzer")
            except Exception as e:
                self.logger.warning(f"Failed to init CascadeAnalyzer: {e}")

        self.logger.info("StegOrchestrator ready with tools: %s", 
                         [type(t).__name__ for t in self.tools])

    def analyze(self, target: str) -> None:
        """
        Sync entrypoint: setup session and invoke async pipeline.
        """
        path = Path(target)
        try:
            session_id = asyncio.run(self.db.create_session(str(path)))
        except Exception:
            session_id = None

        try:
            asyncio.run(self._analyze_async(path, session_id))
        except Exception as e:
            self.logger.error(f"Orchestration failed: {e}\n{traceback.format_exc()}")

    async def _analyze_async(self, target: Path, session_id: Any) -> None:
        # Collect all files under target
        if target.is_file():
            files = [target]
        else:
            files = [p for p in target.rglob('*') if p.is_file()]

        loop = asyncio.get_running_loop()
        tasks: List[Any] = []

        for f in files:
            # Core analysis
            tasks.append(self._run_file_analysis(f, session_id))

            # Optional tools
            for tool in self.tools:
                method = _get_tool_method(tool)
                if not method:
                    self.logger.debug("No entrypoint for tool %s", type(tool).__name__)
                    continue

                if asyncio.iscoroutinefunction(method):
                    tasks.append(self._invoke_async_tool(method, f, session_id))
                else:
                    tasks.append(loop.run_in_executor(
                        self.pool,
                        _invoke_tool,
                        method,
                        f,
                        session_id
                    ))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Store all findings
        for res in results:
            if isinstance(res, Exception):
                self.logger.error(f"Task error: {res}")
            else:
                await self._store_result(session_id, res)

        self.logger.info("Analysis complete for session %s", session_id)

    async def _run_file_analysis(self, file_path: Path, session_id: Any) -> Any:
        try:
            info = await self.file_analyzer.analyze_file(file_path)
            file_id = await self.db.add_file(session_id, str(file_path), info)
            return {'file_id': file_id, 'type': 'file', 'data': info}
        except Exception as e:
            self.logger.error(f"File analysis error for %s: %s", file_path, e)
            return None

    async def _invoke_async_tool(self, method: Callable, f: Path, session_id: Any) -> Any:
        try:
            return await method(f, session_id)
        except TypeError:
            return await method(f)
        except Exception as e:
            self.logger.error(f"Async tool error: {e}")
            return None

    async def _store_result(self, session_id: Any, result: Any) -> None:
        if not result or not isinstance(result, dict):
            return
        # Skip file-type entries
        if result.get('type') == 'file':
            return
        # Batch of findings
        if isinstance(result, list):
            for finding in result:
                await self.db.store_finding(session_id, finding.get('file_id'), finding)
        else:
            await self.db.store_finding(session_id, result.get('file_id'), result)
