#!/usr/bin/env python3
"""
Robust Orchestrator for StegAnalyzer
Handles file analysis and optional tools invocation with proper async integration and per-file context.
"""

import asyncio
import nest_asyncio
nest_asyncio.apply()
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
        # Checkpoint manager
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
        def init_optional(cls, cfg_attr, name, main=False):
            cfg = getattr(config, cfg_attr, None)
            if cls and cfg is not None:
                try:
                    inst = cls(cfg)
                    if main:
                        self.file_forensics_tool = inst
                        self.logger.info(f"Initialized {name}")
                    else:
                        self.tools.append(inst)
                        self.logger.info(f"Initialized tool: {name}")
                except Exception as e:
                    self.logger.warning(f"Failed to init {name}: {e}")
        init_optional(FileForensicsTools, 'file_forensics', 'FileForensicsTools', main=True)
        init_optional(ClassicStegoTools, 'classic_stego', 'ClassicStegoTools')
        init_optional(ImageForensicsTools, 'image_forensics', 'ImageForensicsTools')
        init_optional(AudioAnalysisTools, 'audio_analysis', 'AudioAnalysisTools')
        init_optional(CryptoAnalysisTools, 'crypto', 'CryptoAnalysisTools')
        init_optional(MLStegDetector, 'ml', 'MLStegDetector')
        init_optional(LLMAnalyzer, 'llm', 'LLMAnalyzer')
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
            bool(self.file_forensics_tool), [type(t).__name__ for t in self.tools]
        )

    def analyze(self, target: Union[str, Path], session_id: Any) -> List[Any]:
        """
        Entry point for analysis.
        :param target: file or directory to analyze
        :param session_id: existing session identifier
        :return: list of findings
        """
        path = Path(target)
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self._analyze_async(path, session_id))
        except Exception as e:
            self.logger.error(f"Orchestration failed: {e}\n{traceback.format_exc()}")
            return []

    async def _analyze_async(self, target: Path, session_id: Any) -> List[Any]:
        files = [target] if target.is_file() else list(target.rglob('*'))
        findings: List[Any] = []
        for f in files:
            if f.is_dir():
                continue
            try:
                info = await self.file_analyzer.analyze_file(f)
                file_id = await self.db.add_file(session_id, str(f), info)
            except Exception as e:
                self.logger.error(f"File analysis error for {f}: {e}")
                continue
            tool_tasks = []
            loop = asyncio.get_running_loop()
            if self.file_forensics_tool:
                for m in self.file_forensics_tool.get_supported_methods():
                    tool_tasks.append(
                        loop.run_in_executor(self.pool,
                            lambda method=m: _wrap_sync(method, self.file_forensics_tool, f, session_id)
                        )
                    )
            for tool in self.tools:
                method = _get_tool_method(tool)
                if not method:
                    continue
                if asyncio.iscoroutinefunction(method):
                    task = _wrap_async(method, tool, f, session_id)
                else:
                    task = loop.run_in_executor(self.pool, _wrap_sync, method, tool, f, session_id)
                tool_tasks.append(task)
            if tool_tasks:
                raw = await asyncio.gather(*tool_tasks, return_exceptions=True)
                for res in raw:
                    if isinstance(res, Exception):
                        self.logger.error(f"Tool task error: {res}")
                        continue
                    items = res if isinstance(res, list) else [res]
                    for item in items:
                        if not item or not isinstance(item, dict):
                            continue
                        item.setdefault('file_id', file_id)
                        item.setdefault('tool_name', type(item.get('tool')).__name__ if item.get('tool') else item.get('tool_name', None) or 'UnknownTool')
                        await self._store_result(session_id, item)
                        if item.get('type') != 'file':
                            findings.append(item)
        self.logger.info("Analysis complete for session %s", session_id)
        return findings

    async def _store_result(self, session_id: Any, result: dict) -> None:
        if result.get('type') == 'file':
            return
        fid = result.get('file_id')
        if fid is None:
            self.logger.warning("Skipping result without file_id: %s", result)
            return
        # Ensure tool_name present
        if 'tool_name' not in result or not result['tool_name']:
            result['tool_name'] = 'UnknownTool'
        await self.db.store_finding(session_id, fid, result)

# Helpers

def _get_tool_method(tool: Any) -> Callable:
    for name in ('analyze', f'{type(tool).__name__.lower()}_analysis'):
        if hasattr(tool, name):
            return getattr(tool, name)
    return None

async def _wrap_async(method: Callable, tool: Any, f: Path, session_id: Any) -> Any:
    try:
        res = await method(f, session_id)
    except TypeError:
        res = await method(f)
    except Exception as e:
        logging.getLogger(__name__).error(f"Async tool error: {e}")
        return []
    # annotate result(s)
    if isinstance(res, list):
        for item in res:
            if isinstance(item, dict):
                item['tool_name'] = type(tool).__name__
        return res
    if isinstance(res, dict):
        res['tool_name'] = type(tool).__name__
    return res

def _wrap_sync(method: Callable, tool: Any, f: Path, session_id: Any) -> Any:
    try:
        res = method(f, session_id)
    except TypeError:
        res = method(f)
    except Exception as e:
        logging.getLogger(__name__).error(f"Sync tool error: {e}")
        return []
    if isinstance(res, list):
        for item in res:
            if isinstance(item, dict):
                item['tool_name'] = type(tool).__name__
        return res
    if isinstance(res, dict):
        res['tool_name'] = type(tool).__name__
    return res
