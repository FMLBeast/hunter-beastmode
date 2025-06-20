#!/usr/bin/env python3
"""
StegAnalyzer Orchestrator - Working Version
"""

import os
import sys
import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Union
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import json
import hashlib

@dataclass
class AnalysisTask:
    """Analysis task definition"""
    file_path: Path
    method: str
    tool_name: str
    priority: int = 1
    dependencies: List[str] = None
    estimated_time: float = 1.0
    created_at: float = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.created_at is None:
            self.created_at = time.time()

class StegOrchestrator:
    """Main orchestrator for steganography analysis"""
    
    def __init__(self, config, database=None):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize all tools to None first
        self.file_forensics_tools = None
        self.classic_tools = None
        self.image_tools = None
        self.audio_tools = None
        self.crypto_tools = None
        self.metadata_tools = None
        self.ml_detector = None
        self.llm_analyzer = None
        self.cascade_analyzer = None
        self.cloud = None
        self.gpu_manager = None
        
        # Initialize tools
        self._initialize_tools()
        
        # Thread pools for parallel execution
        self.cpu_executor = None
        self.gpu_executor = None
        
        # Analysis state
        self.active_sessions = {}
        self.completed_tasks = set()
        
    def _initialize_tools(self):
        """Initialize all available tools"""
        
        # File forensics tools
        try:
            from tools.file_forensics import FileForensicsTools
            self.file_forensics_tools = FileForensicsTools(self.config)
            self.logger.info("File forensics tools initialized")
        except ImportError as e:
            self.logger.warning(f"File forensics tools not available: {e}")
        
        # Classic steganography tools
        try:
            from tools.classic_stego import ClassicStegoTools
            self.classic_tools = ClassicStegoTools(self.config)
            self.logger.info("Classic stego tools initialized")
        except ImportError as e:
            self.logger.warning(f"Classic stego tools not available: {e}")
        
        # Image forensics tools
        try:
            from tools.image_forensics import ImageForensicsTools
            self.image_tools = ImageForensicsTools(self.config)
            self.logger.info("Image forensics tools initialized")
        except ImportError as e:
            self.logger.warning(f"Image forensics tools not available: {e}")
        
        # Audio analysis tools
        try:
            from tools.audio_analysis import AudioAnalysisTools
            self.audio_tools = AudioAnalysisTools(self.config)
            self.logger.info("Audio analysis tools initialized")
        except ImportError as e:
            self.logger.warning(f"Audio analysis tools not available: {e}")
        
        # Crypto analysis tools
        try:
            from tools.crypto_analysis import CryptoAnalysisTools
            self.crypto_tools = CryptoAnalysisTools(self.config)
            self.logger.info("Crypto analysis tools initialized")
        except ImportError as e:
            self.logger.warning(f"Crypto analysis tools not available: {e}")
        
        # Metadata carving tools
        try:
            from tools.metadata_carving import MetadataCarving
            self.metadata_tools = MetadataCarving(self.config)
            self.logger.info("Metadata carving tools initialized")
        except ImportError as e:
            self.logger.warning(f"Metadata carving tools not available: {e}")
        
        # ML detector
        try:
            from ai.ml_detector import MLStegDetector
            self.ml_detector = MLStegDetector(self.config)
            self.logger.info("ML detector initialized")
        except ImportError as e:
            self.logger.warning(f"ML detector not available: {e}")
        
        # LLM analyzer
        try:
            from ai.llm_analyzer import LLMAnalyzer
            self.llm_analyzer = LLMAnalyzer(self.config)
            self.logger.info("LLM analyzer initialized")
        except ImportError as e:
            self.logger.warning(f"LLM analyzer not available: {e}")
        
        # Cascade analyzer
        try:
            from tools.cascade_analyzer import CascadeAnalyzer
            self.cascade_analyzer = CascadeAnalyzer(self.config)
            self.logger.info("Cascade analyzer initialized")
        except ImportError as e:
            self.logger.warning(f"Cascade analyzer not available: {e}")
    
    def get_tool(self, tool_name: str):
        """Get tool by name"""
        tool_map = {
            'file_forensics': self.file_forensics_tools,
            'classic_stego': self.classic_tools,
            'image_forensics': self.image_tools,
            'audio_analysis': self.audio_tools,
            'crypto_analysis': self.crypto_tools,
            'metadata_carving': self.metadata_tools,
            'ml_detector': self.ml_detector,
            'llm_analyzer': self.llm_analyzer,
            'cascade_analyzer': self.cascade_analyzer
        }
        return tool_map.get(tool_name)
    
    def _get_tool_for_method(self, method: str) -> str:
        """Get the appropriate tool for a given method"""
        method_tool_map = {
            # File analysis methods
            "magic_analysis": "file_forensics",
            "basic_analysis": "file_forensics",
            "entropy_analysis": "file_forensics",
            "hex_analysis": "file_forensics",
            
            # Image analysis methods
            "lsb_analysis": "classic_stego",
            "metadata_analysis": "metadata_carving",
            "image_forensics": "image_forensics",
            "steghide_extract": "classic_stego",
            "outguess_extract": "classic_stego",
            "zsteg_analysis": "classic_stego",
            
            # Audio analysis methods
            "audio_spectral": "audio_analysis",
            "audio_lsb": "audio_analysis",
            "audio_metadata": "metadata_carving",
            
            # Crypto analysis methods
            "crypto_analysis": "crypto_analysis",
            "hash_analysis": "crypto_analysis",
            "cipher_detection": "crypto_analysis",
            
            # ML analysis methods
            "ml_detection": "ml_detector",
            "anomaly_detection": "ml_detector",
            "statistical_analysis": "ml_detector",
            
            # LLM analysis methods
            "llm_analysis": "llm_analyzer",
            "pattern_recognition": "llm_analyzer",
            
            # Cascade analysis methods
            "cascade_analysis": "cascade_analyzer",
            "recursive_extract": "cascade_analyzer",
        }
        
        return method_tool_map.get(method, "file_forensics")  # Default to file_forensics
    
    def execute_method(self, method: str, file_path: Path) -> List[Dict[str, Any]]:
        """Execute a specific analysis method"""
        try:
            tool_name = self._get_tool_for_method(method)
            tool = self.get_tool(tool_name)
            
            if not tool:
                self.logger.warning(f"Tool {tool_name} not available for {method}")
                return []
            
            if not hasattr(tool, 'execute_method'):
                self.logger.warning(f"Tool {tool_name} missing execute_method for {method}")
                return []
            
            return tool.execute_method(method, file_path)
            
        except Exception as e:
            self.logger.error(f"Error executing {method}: {e}")
            return []
    
    async def analyze(self, file_path: Path, session_id: str = None) -> List[Dict[str, Any]]:
        """Main analysis method"""
        if session_id is None:
            session_id = self._generate_session_id()
        
        self.logger.info(f"Starting analysis of {file_path}")
        
        try:
            # Create analysis tasks
            tasks = self._create_analysis_tasks(file_path, set())
            
            # Execute tasks
            results = []
            for task in tasks:
                try:
                    self.logger.info(f"Executing {task.method} on {file_path}")
                    task_results = self.execute_method(task.method, task.file_path)
                    results.extend(task_results)
                except Exception as e:
                    self.logger.error(f"Task {task.method} failed: {e}")
                    continue
            
            self.logger.info(f"Post-processing {len(results)} results")
            
            # Post-process results
            processed_results = self._post_process_results(results, file_path)
            
            return processed_results
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            return [{
                "type": "error",
                "error": str(e),
                "file_path": str(file_path),
                "session_id": session_id
            }]
    
    def _create_analysis_tasks(self, file_path: Path, completed_set: Set[str]) -> List[AnalysisTask]:
        """Create analysis tasks based on file type and available tools"""
        tasks = []
        file_ext = file_path.suffix.lower()
        
        # Basic file analysis tasks
        if "magic_analysis" not in completed_set:
            tasks.append(AnalysisTask(
                file_path=file_path,
                method="magic_analysis",
                tool_name="file_forensics",
                priority=10
            ))
        
        if "entropy_analysis" not in completed_set:
            tasks.append(AnalysisTask(
                file_path=file_path,
                method="entropy_analysis",
                tool_name="file_forensics",
                priority=8
            ))
        
        # Image-specific tasks
        if file_ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']:
            if self.classic_tools and "zsteg_analysis" not in completed_set:
                tasks.append(AnalysisTask(
                    file_path=file_path,
                    method="zsteg_analysis",
                    tool_name="classic_stego",
                    priority=7
                ))
            
            if self.classic_tools and "steghide_extract" not in completed_set:
                tasks.append(AnalysisTask(
                    file_path=file_path,
                    method="steghide_extract",
                    tool_name="classic_stego",
                    priority=6
                ))
        
        # ML detection
        if self.ml_detector and "ml_detection" not in completed_set:
            tasks.append(AnalysisTask(
                file_path=file_path,
                method="ml_detection",
                tool_name="ml_detector",
                priority=5
            ))
        
        # LLM analysis
        if self.llm_analyzer and "llm_analysis" not in completed_set:
            tasks.append(AnalysisTask(
                file_path=file_path,
                method="llm_analysis",
                tool_name="llm_analyzer",
                priority=4
            ))
        
        return sorted(tasks, key=lambda t: t.priority, reverse=True)
    
    def _post_process_results(self, results: List[Dict[str, Any]], file_path: Path) -> List[Dict[str, Any]]:
        """Post-process analysis results"""
        processed_results = []
        
        for result in results:
            if isinstance(result, dict):
                # Add metadata
                result['file_path'] = str(file_path)
                result['timestamp'] = time.time()
                result['file_size'] = file_path.stat().st_size if file_path.exists() else 0
                
                # Add confidence if missing
                if 'confidence' not in result:
                    result['confidence'] = 0.5
                
                processed_results.append(result)
        
        return processed_results
    
    def _generate_session_id(self) -> str:
        """Generate a unique session ID"""
        import uuid
        return str(uuid.uuid4())
    
    async def analyze_cascade(self, file_path: Path, session_id: str = None) -> Dict[str, Any]:
        """Run cascade analysis"""
        if not self.cascade_analyzer:
            return {
                "type": "cascade_error",
                "error": "Cascade analyzer not available",
                "file_path": str(file_path)
            }
        
        try:
            self.logger.info(f"Running cascade analysis on {file_path}")
            results = await self.cascade_analyzer.cascade_analyze(file_path, session_id)
            
            return {
                "type": "cascade_complete",
                "total_results": len(results),
                "results": results,
                "file_path": str(file_path)
            }
            
        except Exception as e:
            self.logger.error(f"Cascade analysis failed: {e}")
            return {
                "type": "cascade_error",
                "error": str(e),
                "file_path": str(file_path)
            }
    
    async def get_available_tools(self) -> Dict[str, bool]:
        """Get status of all available tools"""
        return {
            'file_forensics': self.file_forensics_tools is not None,
            'classic_stego': self.classic_tools is not None,
            'image_forensics': self.image_tools is not None,
            'audio_analysis': self.audio_tools is not None,
            'crypto_analysis': self.crypto_tools is not None,
            'metadata_carving': self.metadata_tools is not None,
            'ml_detector': self.ml_detector is not None,
            'llm_analyzer': self.llm_analyzer is not None,
            'cascade_analyzer': self.cascade_analyzer is not None
        }
    
    def cleanup(self):
        """Cleanup resources"""
        if self.cpu_executor:
            self.cpu_executor.shutdown(wait=True)
        if self.gpu_executor:
            self.gpu_executor.shutdown(wait=True)
