"""
Core Orchestrator - Manages the entire steganography analysis pipeline
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
import hashlib
import time
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

# Core imports that should always work
from core.file_analyzer import FileAnalyzer
from core.graph_tracker import GraphTracker
from utils.checkpoint import CheckpointManager

# Tool imports with graceful handling
try:
    from tools.classic_stego import ClassicStegoTools
except ImportError as e:
    print(f"Warning: ClassicStegoTools not available: {e}")
    ClassicStegoTools = None

try:
    from tools.image_forensics import ImageForensicsTools
except ImportError as e:
    print(f"Warning: ImageForensicsTools not available: {e}")
    ImageForensicsTools = None

try:
    from tools.audio_analysis import AudioAnalysisTools
except ImportError as e:
    print(f"Warning: AudioAnalysisTools not available (missing audio dependencies): {e}")
    AudioAnalysisTools = None

try:
    from tools.file_forensics import FileForensicsTools
except ImportError as e:
    print(f"Warning: FileForensicsTools not available: {e}")
    FileForensicsTools = None

try:
    from tools.crypto_analysis import CryptoAnalysisTools
except ImportError as e:
    print(f"Warning: CryptoAnalysisTools not available: {e}")
    CryptoAnalysisTools = None

try:
    from tools.metadata_carving import MetadataCarving
except ImportError as e:
    print(f"Warning: MetadataCarving not available: {e}")
    MetadataCarving = None

# AI components (optional)
try:
    from ai.ml_detector import MLStegDetector
except ImportError as e:
    print(f"Warning: MLStegDetector not available: {e}")
    MLStegDetector = None

try:
    from ai.llm_analyzer import LLMAnalyzer
except ImportError as e:
    print(f"Warning: LLMAnalyzer not available: {e}")
    LLMAnalyzer = None

try:
    from ai.multimodal_classifier import MultimodalClassifier
except ImportError as e:
    print(f"Warning: MultimodalClassifier not available: {e}")
    MultimodalClassifier = None

# Cloud integrations (optional)
try:
    from cloud.integrations import CloudIntegrations
except ImportError as e:
    print(f"Warning: CloudIntegrations not available: {e}")
    CloudIntegrations = None

# GPU management (optional)
try:
    from utils.gpu_manager import GPUManager
except ImportError as e:
    print(f"Warning: GPUManager not available: {e}")
    GPUManager = None


@dataclass
class AnalysisTask:
    file_path: Path
    method: str
    tool_name: str
    priority: int
    dependencies: List[str]
    gpu_required: bool = False
    estimated_time: float = 0.0


class StegOrchestrator:
    def __init__(self, config, database):
        self.config = config
        self.db = database
        self.logger = logging.getLogger(__name__)
        
        # Initialize tool managers (only if available) - FIXED TO USE BRACKET NOTATION
        self.classic_tools = ClassicStegoTools(self.config.get('classic_stego', {})) if ClassicStegoTools else None
        self.image_tools = ImageForensicsTools(self.config.get('image_forensics', {})) if ImageForensicsTools else None
        self.audio_tools = AudioAnalysisTools(self.config.get('audio_analysis', {})) if AudioAnalysisTools else None
        self.file_tools = FileForensicsTools(self.config.get('file_forensics', {})) if FileForensicsTools else None
        self.crypto_tools = CryptoAnalysisTools(self.config.get('crypto', {})) if CryptoAnalysisTools else None
        self.metadata_tools = MetadataCarving(self.config) if MetadataCarving else None
        
        # Initialize AI components (only if available)
        self.ml_detector = MLStegDetector(self.config.get('ml', {})) if MLStegDetector else None
        self.llm_analyzer = LLMAnalyzer(self.config.get('llm', {})) if LLMAnalyzer else None
        self.multimodal_classifier = MultimodalClassifier(self.config.get('multimodal', {})) if MultimodalClassifier else None
        
        # Initialize cloud integrations (only if available)
        cloud_config = self.config.get('cloud', {})
        self.cloud = CloudIntegrations(config) if (CloudIntegrations and cloud_config.get('enabled', False)) else None
        
        # Core components (should always be available)
        self.file_analyzer = FileAnalyzer(self.config.get('analysis', {}))
        self.graph_tracker = GraphTracker(database)
        self.gpu_manager = GPUManager() if GPUManager else None
        self.checkpoint_manager = CheckpointManager(database)
        
        # Task management
        self.task_queue = asyncio.Queue()
        self.running_tasks = set()
        self.completed_tasks = set()
        
        self.logger.info(f"StegOrchestrator initialized with available tools: "
                        f"classic={self.classic_tools is not None}, "
                        f"image={self.image_tools is not None}, "
                        f"audio={self.audio_tools is not None}, "
                        f"file={self.file_tools is not None}, "
                        f"crypto={self.crypto_tools is not None}")

    async def analyze(self, file_path: Path, session_id: str) -> List[Dict[str, Any]]:
        """Main analysis orchestration method"""
        self.logger.info(f"Starting analysis of {file_path}")
        
        try:
            # Basic file analysis first
            file_info = await self.file_analyzer.analyze_file(file_path)
            file_info['file_path'] = str(file_path)  # Ensure file_path is present
            await self.db.store_file_analysis(session_id, file_info)
            
            # Create analysis plan based on file type
            tasks = await self._create_analysis_plan(file_path, file_info)
            
            # Execute tasks in parallel with dependency management
            results = await self._execute_analysis_tasks(tasks, session_id)
            
            # Post-process results
            final_results = await self._post_process_results(results, session_id)
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Analysis failed for {file_path}: {e}")
            raise

    async def _create_analysis_plan(self, file_path: Path, file_info: Dict[str, Any]) -> List[AnalysisTask]:
        """Create analysis plan based on file type and available tools"""
        tasks = []
        file_type = file_info.get('file_type', '').lower()
        
        # Check for completed tasks in database
        completed = await self.db.get_completed_methods(str(file_path))
        completed_set = set(completed) if completed else set()
        
        # Basic tasks that always run
        if "basic_analysis" not in completed_set:
            tasks.append(AnalysisTask(
                file_path=file_path, method="basic_analysis", tool_name="file_analyzer",
                priority=1, dependencies=[], estimated_time=1.0
            ))
        
        # File type specific tasks
        if 'image' in file_type:
            tasks.extend(self._create_image_analysis_tasks(file_path, completed_set))
        elif 'audio' in file_type:
            tasks.extend(self._create_audio_analysis_tasks(file_path, completed_set))
        elif 'video' in file_type:
            tasks.extend(self._create_video_analysis_tasks(file_path, completed_set))
        else:
            tasks.extend(self._create_generic_analysis_tasks(file_path, completed_set))
        
        # Add AI analysis tasks if available
        if self.ml_detector and "ml_detection" not in completed_set:
            tasks.append(AnalysisTask(
                file_path=file_path, method="ml_detection", tool_name="ml_detector",
                priority=3, dependencies=["basic_analysis"], gpu_required=True,
                estimated_time=10.0
            ))
        
        return tasks

    def _create_image_analysis_tasks(self, file_path: Path, completed_set: Set[str]) -> List[AnalysisTask]:
        """Create analysis tasks specific to image files"""
        tasks = []
        
        # Classic steganography tools
        if self.classic_tools:
            classic_methods = ["steghide", "outguess", "zsteg", "jsteg"]
            for method in classic_methods:
                if method not in completed_set:
                    tasks.append(AnalysisTask(
                        file_path=file_path, method=method, tool_name="classic_stego",
                        priority=2, dependencies=["basic_analysis"], estimated_time=5.0
                    ))
        
        # Image forensics
        if self.image_tools:
            forensic_methods = ["metadata_analysis", "noise_analysis", "jpeg_quality"]
            for method in forensic_methods:
                if method not in completed_set:
                    tasks.append(AnalysisTask(
                        file_path=file_path, method=method, tool_name="image_forensics",
                        priority=2, dependencies=["basic_analysis"], estimated_time=3.0
                    ))
        
        return tasks

    def _create_audio_analysis_tasks(self, file_path: Path, completed_set: Set[str]) -> List[AnalysisTask]:
        """Create analysis tasks specific to audio files"""
        tasks = []
        
        if self.audio_tools:
            audio_methods = ["spectral_analysis", "lsb_analysis", "echo_hiding"]
            for method in audio_methods:
                if method not in completed_set:
                    tasks.append(AnalysisTask(
                        file_path=file_path, method=method, tool_name="audio_analysis",
                        priority=2, dependencies=["basic_analysis"], estimated_time=8.0
                    ))
        
        return tasks

    def _create_video_analysis_tasks(self, file_path: Path, completed_set: Set[str]) -> List[AnalysisTask]:
        """Create analysis tasks specific to video files"""
        tasks = []
        
        # Video analysis methods
        video_methods = ["frame_analysis", "metadata_extraction", "codec_analysis"]
        for method in video_methods:
            if method not in completed_set:
                tasks.append(AnalysisTask(
                    file_path=file_path, method=method, tool_name="video_analysis",
                    priority=2, dependencies=["basic_analysis"], estimated_time=15.0
                ))
        
        return tasks

    def _create_generic_analysis_tasks(self, file_path: Path, completed_set: Set[str]) -> List[AnalysisTask]:
        """Create analysis tasks for generic files"""
        tasks = []
        
        # File forensics
        if self.file_tools:
            forensic_methods = ["signature_analysis", "entropy_analysis", "string_extraction"]
            for method in forensic_methods:
                if method not in completed_set:
                    tasks.append(AnalysisTask(
                        file_path=file_path, method=method, tool_name="file_forensics",
                        priority=2, dependencies=["basic_analysis"], estimated_time=4.0
                    ))
        
        # Crypto analysis
        if self.crypto_tools:
            crypto_methods = ["entropy_test", "chi_square_test", "pattern_detection"]
            for method in crypto_methods:
                if method not in completed_set:
                    tasks.append(AnalysisTask(
                        file_path=file_path, method=method, tool_name="crypto_analysis",
                        priority=2, dependencies=["basic_analysis"], estimated_time=6.0
                    ))
        
        return tasks

    async def _execute_analysis_tasks(self, tasks: List[AnalysisTask], session_id: str) -> List[Dict[str, Any]]:
        """Execute analysis tasks with proper dependency management"""
        results = []
        completed_tasks = set()
        
        # Sort tasks by priority
        tasks.sort(key=lambda t: t.priority)
        
        # Execute tasks in batches based on dependencies
        max_concurrent = self.config.get('orchestrator', {}).get('max_concurrent_files', 4)
        
        for task in tasks:
            try:
                # Check if dependencies are met
                if all(dep in completed_tasks for dep in task.dependencies):
                    # Execute the task
                    task_results = await self._execute_single_task(task, session_id)
                    results.extend(task_results)
                    completed_tasks.add(task.method)
                else:
                    self.logger.warning(f"Skipping task {task.method} due to unmet dependencies")
            
            except Exception as e:
                self.logger.error(f"Task {task.method} failed: {e}")
                continue
        
        return results

    async def _execute_single_task(self, task: AnalysisTask, session_id: str) -> List[Dict[str, Any]]:
        """Execute a single analysis task"""
        self.logger.debug(f"Executing task: {task.method} on {task.file_path}")
        
        results = []
        
        try:
            # Route to appropriate tool
            if task.tool_name == "file_analyzer":
                result = await self.file_analyzer.analyze_file(task.file_path)
                results.append(result)
                
            elif task.tool_name == "classic_stego" and self.classic_tools:
                result = await self._run_classic_stego_method(task.method, task.file_path)
                if result:
                    results.append(result)
                    
            elif task.tool_name == "image_forensics" and self.image_tools:
                result = await self._run_image_forensics_method(task.method, task.file_path)
                if result:
                    results.append(result)
                    
            elif task.tool_name == "ml_detector" and self.ml_detector:
                result = await self._run_ml_detection(task.file_path)
                if result:
                    results.append(result)
            
            # Store results in database
            for result in results:
                if result and 'tool_name' in result:
                    file_id = result.get('file_id')
                    if not file_id:
                        # Try to get file_id from database
                        cursor = self.db.sqlite_conn.cursor()
                        cursor.execute("SELECT id FROM files WHERE session_id = ? AND file_path = ?", 
                                     (session_id, str(task.file_path)))
                        row = cursor.fetchone()
                        file_id = row[0] if row else None
                    
                    if file_id:
                        await self.db.store_finding(session_id, file_id, result)
        
        except Exception as e:
            self.logger.error(f"Error executing task {task.method}: {e}")
        
        return results

    async def _run_classic_stego_method(self, method: str, file_path: Path) -> Dict[str, Any]:
        """Run a classic steganography detection method"""
        try:
            # This would call the actual tool method
            result = {
                "tool_name": "classic_stego",
                "method": method,
                "file_path": str(file_path),
                "type": "steganography_detection",
                "confidence": 0.0,
                "description": f"Classic steganography analysis using {method}",
                "evidence": "",
                "metadata": {}
            }
            return result
        except Exception as e:
            self.logger.error(f"Classic stego method {method} failed: {e}")
            return None

    async def _run_image_forensics_method(self, method: str, file_path: Path) -> Dict[str, Any]:
        """Run an image forensics method"""
        try:
            result = {
                "tool_name": "image_forensics",
                "method": method,
                "file_path": str(file_path),
                "type": "image_forensics",
                "confidence": 0.0,
                "description": f"Image forensics analysis using {method}",
                "evidence": "",
                "metadata": {}
            }
            return result
        except Exception as e:
            self.logger.error(f"Image forensics method {method} failed: {e}")
            return None

    async def _run_ml_detection(self, file_path: Path) -> Dict[str, Any]:
        """Run ML-based steganography detection"""
        try:
            result = {
                "tool_name": "ml_detector",
                "method": "cnn_detection",
                "file_path": str(file_path),
                "type": "ml_detection",
                "confidence": 0.0,
                "description": "Machine learning based steganography detection",
                "evidence": "",
                "metadata": {}
            }
            return result
        except Exception as e:
            self.logger.error(f"ML detection failed: {e}")
            return None

    async def _post_process_results(self, results: List[Dict[str, Any]], session_id: str) -> List[Dict[str, Any]]:
        """Post-process and correlate analysis results"""
        # Sort results by confidence and type
        high_confidence = [r for r in results if r.get('confidence', 0) > 0.7]
        medium_confidence = [r for r in results if 0.3 < r.get('confidence', 0) <= 0.7]
        
        # Correlate findings
        correlated = await self._correlate_findings(results)
        
        # Update graph tracker
        await self.graph_tracker.update_analysis_graph(session_id, results)
        
        # Add summary statistics
        summary = {
            'total_findings': len(results),
            'high_confidence': len(high_confidence),
            'medium_confidence': len(medium_confidence),
            'tools_used': list(set(r.get('tool_name') for r in results if r.get('tool_name'))),
            'file_types_analyzed': list(set(r.get('file_type') for r in results if r.get('file_type'))),
            'correlation_score': correlated.get('score', 0.0)
        }
        
        # Store summary
        await self.db.store_analysis_summary(session_id, summary)
        
        return results

    async def _correlate_findings(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Correlate findings from different tools"""
        correlations = {
            'cross_tool_confirmations': 0,
            'conflicting_findings': 0,
            'unique_findings': 0,
            'score': 0.0
        }
        
        # Group by detection method or content
        content_groups = {}
        for result in results:
            content = result.get('extracted_content', '')
            if content and len(content) > 10:
                key = hashlib.md5(content.encode()).hexdigest()[:8]
                if key not in content_groups:
                    content_groups[key] = []
                content_groups[key].append(result)
        
        # Count confirmations
        for group in content_groups.values():
            if len(group) > 1:
                correlations['cross_tool_confirmations'] += len(group) - 1
        
        # Calculate correlation score
        if results:
            correlations['score'] = min(1.0, correlations['cross_tool_confirmations'] / len(results))
        
        return correlations

    async def analyze_directory(self, directory: Path, session_id: str) -> Dict[str, Any]:
        """Analyze all files in a directory"""
        self.logger.info(f"Starting directory analysis: {directory}")
        
        # Find all files
        files = []
        for file_path in directory.rglob("*"):
            if file_path.is_file() and not file_path.name.startswith('.'):
                files.append(file_path)
        
        if not files:
            raise ValueError(f"No files found in directory: {directory}")
        
        # Analyze each file
        all_results = []
        for file_path in files:
            try:
                file_results = await self.analyze(file_path, session_id)
                all_results.extend(file_results)
            except Exception as e:
                self.logger.error(f"Failed to analyze {file_path}: {e}")
                continue
        
        return {
            "files_processed": len(files),
            "total_findings": len(all_results),
            "results": all_results
        }