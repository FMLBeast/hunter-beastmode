#!/usr/bin/env python3
"""
Robust Orchestrator for StegAnalyzer
Handles file analysis and optional tools invocation.
"""

import asyncio
import logging
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Callable, Union

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
    Orchestrates analysis across files, using configured tools and database.
    Provides a single `analyze(target, session_id)` method returning findings.
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
        # Thread pool for blocking tasks
        max_workers = getattr(config.orchestrator, 'max_cpu_workers', 4)
        self.pool = ThreadPoolExecutor(max_workers=max_workers)
        # Core file analyzer
        fa_cfg = getattr(config, 'file_forensics', {})
        self.file_analyzer = FileAnalyzer(fa_cfg)
        # Initialize optional tools
        self.file_forensics_tool = None
        self.tools: List[Any] = []
        def init_optional(cls, cfg_attr, name, assign_main=False):
            cfg = getattr(config, cfg_attr, None)
            if cls and cfg is not None:
                try:
                    inst = cls(cfg)
                    if assign_main:
                        self.file_forensics_tool = inst
                        self.logger.info(f"Initialized {name}")
                    else:
                        self.tools.append(inst)
                        self.logger.info(f"Initialized tool: {name}")
                except Exception as e:
                    self.logger.warning(f"Failed to init {name}: {e}")
        # File forensics uses its own config
        init_optional(FileForensicsTools, 'file_forensics', 'FileForensicsTools', assign_main=True)
        init_optional(ClassicStegoTools, 'classic_stego', 'ClassicStegoTools')
        init_optional(ImageForensicsTools, 'image_forensics', 'ImageForensicsTools')
        init_optional(AudioAnalysisTools, 'audio_analysis', 'AudioAnalysisTools')
        init_optional(CryptoAnalysisTools, 'crypto', 'CryptoAnalysisTools')
        init_optional(MLStegDetector, 'ml', 'MLStegDetector')
        init_optional(LLMAnalyzer, 'llm', 'LLMAnalyzer')
        # Cascade analyzer fallback to orchestrator config
        if CascadeAnalyzer:
            cascade_cfg = getattr(config, 'cascade', getattr(config, 'orchestrator', {}))
            try:
                inst = CascadeAnalyzer(cascade_cfg)
                self.tools.append(inst)
                self.logger.info("Initialized tool: CascadeAnalyzer")
            except Exception as e:
                self.logger.warning(f"Failed to init CascadeAnalyzer: {e}")
        self.logger.info(
            "StegOrchestrator ready. FileForensics: %s, Other tools: %s",
            bool(self.file_forensics_tool),
            [type(t).__name__ for t in self.tools]
        )

    def analyze(self, target: Union[str, Path], session_id: Any) -> List[Any]:
        """
        Entry point for analysis.
        :param target: file or directory to analyze
        :param session_id: existing session identifier
        :return: list of findings
        """
        path = Path(target)
        # Use a fresh event loop to avoid conflicts
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self._analyze_async(path, session_id))
        except Exception as e:
            self.logger.error(f"Orchestration failed: {e}\n{traceback.format_exc()}")
            return []
        finally:
            loop.close()

    async def _analyze_async(self, target: Path, session_id: Any) -> List[Any]:
        # Gather files
        files = [target] if target.is_file() else [p for p in target.rglob('*') if p.is_file()]
        loop = asyncio.get_running_loop()
        tasks: List[Any] = []
        for f in files:
            # File analysis
            tasks.append(self._run_file_analysis(f, session_id))
            # File forensics
            if self.file_forensics_tool:
                for method in self.file_forensics_tool.get_supported_methods():
                    tasks.append(loop.run_in_executor(
                        self.pool,
                        self.file_forensics_tool.execute_method,
                        method,
                        f
                    ))
            # Other tools
            for tool in self.tools:
                if tool is self.file_forensics_tool:
                    continue
                method = _get_tool_method(tool)
                if not method:
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
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)
        findings: List[Any] = []
        for res in raw_results:
            if isinstance(res, Exception):
                self.logger.error(f"Task error: {res}")
                continue
            # store and collect
            if isinstance(res, list):
                for item in res:
                    await self._store_result(session_id, item)
                    if item.get('type') != 'file':
                        findings.append(item)
            elif isinstance(res, dict):
                if res.get('type') != 'file':
                    await self._store_result(session_id, res)
                    findings.append(res)
        self.logger.info("Analysis complete for session %s", session_id)
        return findings

    async def _run_file_analysis(self, file_path: Path, session_id: Any) -> dict:
        try:
            info = await self.file_analyzer.analyze_file(file_path)
            file_id = await self.db.add_file(session_id, str(file_path), info)
            return {'file_id': file_id, 'type': 'file', 'data': info}
        except Exception as e:
            self.logger.error(f"File analysis error for {file_path}: {e}")
            return {'type': 'file', 'error': str(e)}

    async def _invoke_async_tool(self, method: Callable, f: Path, session_id: Any) -> Any:
        try:
            return await method(f, session_id)
        except TypeError:
            return await method(f)
        except Exception as e:
            self.logger.error(f"Async tool error: {e}")
            return {}

    async def _store_result(self, session_id: Any, result: dict) -> None:
        if result.get('type') == 'file':
            return
        file_id = result.get('file_id')
        await self.db.store_finding(session_id, file_id, result)

# Helpers

def _get_tool_method(tool: Any) -> Callable:
    for name in ('analyze', f'{type(tool).__name__.lower()}_analysis'):
        if hasattr(tool, name):
            return getattr(tool, name)
    return None

def _invoke_tool(method: Callable, f: Path, session_id: Any) -> Any:
    try:
        return method(f, session_id)
    except TypeError:
        return method(f)
    except Exception as e:
        logging.getLogger(__name__).error(f"Tool error: {e}")
        return {}
