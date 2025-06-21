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

# Optional tool imports (if available)
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
        # optional checkpoint manager
        try:
            self.checkpoint = CheckpointManager(getattr(config, 'orchestrator', {}), database)
        except Exception:
            self.checkpoint = None
        # thread pool
        max_workers = getattr(config.orchestrator, 'max_cpu_workers', 4)
        self.pool = ThreadPoolExecutor(max_workers=max_workers)
        # core file analyzer (async)
        fa_cfg = getattr(config, 'file_forensics', {})
        self.file_analyzer = FileAnalyzer(fa_cfg)
        # initialize tools
        self.file_forensics_tool = None
        self.tools: List[Any] = []
        self._init_tool(FileForensicsTools, 'file_forensics', main=True)
        self._init_tool(ClassicStegoTools, 'classic_stego')
        self._init_tool(ImageForensicsTools, 'image_forensics')
        self._init_tool(AudioAnalysisTools, 'audio_analysis')
        self._init_tool(CryptoAnalysisTools, 'crypto')
        self._init_tool(MLStegDetector, 'ml')
        self._init_tool(LLMAnalyzer, 'llm')
        # cascade analyzer from orchestrator config
        if CascadeAnalyzer:
            cascade_cfg = getattr(config, 'cascade', getattr(config, 'orchestrator', {}))
            try:
                inst = CascadeAnalyzer(cascade_cfg)
                self.tools.append(inst)
                self.logger.info("Initialized tool: CascadeAnalyzer")
            except Exception as e:
                self.logger.warning(f"Failed to init CascadeAnalyzer: {e}")
        self.logger.info(
            "StegOrchestrator ready: FileForensics=%s, OtherTools=%s",
            bool(self.file_forensics_tool), [type(t).__name__ for t in self.tools]
        )

    def _init_tool(self, cls, cfg_attr: str, main: bool=False):
        cfg = getattr(self.config, cfg_attr, None)
        if cls and cfg is not None:
            try:
                inst = cls(cfg)
                if main:
                    self.file_forensics_tool = inst
                    self.logger.info(f"Initialized FileForensicsTools")
                else:
                    self.tools.append(inst)
                    self.logger.info(f"Initialized tool: {type(inst).__name__}")
            except Exception as e:
                self.logger.warning(f"Failed to init {type(inst).__name__ if 'inst' in locals() else cls.__name__}: {e}")

    def analyze(self, target: Union[str, Path], session_id: Any) -> List[Any]:
        """
        Entry point for analysis. Synchronous wrapper.
        """
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self._analyze_async(Path(target), session_id))
        except Exception as e:
            self.logger.error(f"Analysis orchestration error: {e}\n{traceback.format_exc()}")
            return []

    async def _analyze_async(self, path: Path, session_id: Any) -> List[Any]:
        # gather files
        files = [path] if path.is_file() else [p for p in path.rglob('*') if p.is_file()]
        findings: List[Any] = []
        for f in files:
            # file analysis
            try:
                info = await self.file_analyzer.analyze_file(f)
                file_id = await self.db.add_file(session_id, str(f), info)
            except Exception as e:
                self.logger.error(f"Failed file analysis {f}: {e}")
                continue
            # prepare tool tasks
            tasks = []
            if self.file_forensics_tool:
                for method in self.file_forensics_tool.get_supported_methods():
                    tasks.append(
                        asyncio.get_running_loop().run_in_executor(
                            self.pool, _wrap_sync, method, self.file_forensics_tool, f, session_id
                        )
                    )
            for tool in self.tools:
                method = _get_tool_method(tool)
                if not method:
                    continue
                if asyncio.iscoroutinefunction(method):
                    tasks.append(_wrap_async(method, tool, f, session_id))
                else:
                    tasks.append(
                        asyncio.get_running_loop().run_in_executor(
                            self.pool, _wrap_sync, method, tool, f, session_id
                        )
                    )
            # run all
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for res in results:
                    if isinstance(res, Exception):
                        self.logger.error(f"Tool task error: {res}")
                        continue
                    items = res if isinstance(res, list) else [res]
                    for item in items:
                        if not isinstance(item, dict):
                            continue
                        item.setdefault('file_id', file_id)
                        item.setdefault('tool_name', item.get('tool_name', type(tool).__name__ if 'tool' in locals() else 'Unknown'))
                        await self._store_finding(session_id, item)
                        if item.get('type') != 'file':
                            findings.append(item)
        self.logger.info("Session %s analysis complete", session_id)
        return findings

    async def _store_finding(self, session_id: Any, result: dict):
        if result.get('type') == 'file':
            return
        file_id = result.get('file_id')
        if not file_id:
            self.logger.warning(f"Skip result without file_id: {result}")
            return
        tool_name = result.get('tool_name') or 'UnknownTool'
        result['tool_name'] = tool_name
        await self.db.store_finding(session_id, file_id, result)

# Helpers

def _get_tool_method(tool: Any) -> Callable:
    for name in ('analyze', f'{type(tool).__name__.lower()}_analysis'):
        if hasattr(tool, name):
            return getattr(tool, name)
    return None

async def _wrap_async(method: Callable, tool: Any, f: Path, session_id: Any):
    try:
        return await method(f, session_id)
    except TypeError:
        return await method(f)
    except Exception as e:
        logging.getLogger(__name__).error(f"Async error {tool}: {e}")
        return []

def _wrap_sync(method: Callable, tool: Any, f: Path, session_id: Any):
    try:
        return method(f, session_id)
    except TypeError:
        return method(f)
    except Exception as e:
        logging.getLogger(__name__).error(f"Sync error {tool}: {e}")
        return []
