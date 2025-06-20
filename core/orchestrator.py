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
        
        # Initialize tool managers (only if available)
        self.classic_tools = ClassicStegoTools(config) if ClassicStegoTools else None
        self.image_tools = ImageForensicsTools(config) if ImageForensicsTools else None
        self.audio_tools = AudioAnalysisTools(config) if AudioAnalysisTools else None
        self.file_tools = FileForensicsTools(config) if FileForensicsTools else None
        self.crypto_tools = CryptoAnalysisTools(config) if CryptoAnalysisTools else None
        self.metadata_tools = MetadataCarving(config) if MetadataCarving else None
        
        # Initialize AI components (only if available)
        self.ml_detector = MLStegDetector(config) if MLStegDetector else None
        self.llm_analyzer = LLMAnalyzer(config) if LLMAnalyzer else None
        self.multimodal_classifier = MultimodalClassifier(config) if MultimodalClassifier else None
        
        # Initialize cloud integrations (only if available)
        self.cloud = CloudIntegrations(config) if (CloudIntegrations and config.cloud.enabled) else None
        
        # Core components (should always be available)
        self.file_analyzer = FileAnalyzer(config)
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
            file_info = await self.file_analyzer.analyze(file_path)
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
                priority=3, dependencies=["basic_analysis"], gpu_required=True, estimated_time=10.0
            ))
        
        if self.llm_analyzer and "llm_analysis" not in completed_set:
            tasks.append(AnalysisTask(
                file_path=file_path, method="llm_analysis", tool_name="llm_analyzer",
                priority=4, dependencies=["basic_analysis"], estimated_time=15.0
            ))
        
        return sorted(tasks, key=lambda t: t.priority)

    def _create_image_analysis_tasks(self, file_path: Path, completed: Set[str]) -> List[AnalysisTask]:
        """Create image-specific analysis tasks"""
        tasks = []
        
        if self.classic_tools:
            if "steghide" not in completed:
                tasks.append(AnalysisTask(
                    file_path=file_path, method="steghide", tool_name="classic_stego",
                    priority=2, dependencies=["basic_analysis"], estimated_time=5.0
                ))
            
            if "outguess" not in completed:
                tasks.append(AnalysisTask(
                    file_path=file_path, method="outguess", tool_name="classic_stego",
                    priority=2, dependencies=["basic_analysis"], estimated_time=5.0
                ))
            
            if "zsteg" not in completed:
                tasks.append(AnalysisTask(
                    file_path=file_path, method="zsteg", tool_name="classic_stego",
                    priority=2, dependencies=["basic_analysis"], estimated_time=8.0
                ))
        
        if self.image_tools:
            if "ela_analysis" not in completed:
                tasks.append(AnalysisTask(
                    file_path=file_path, method="ela_analysis", tool_name="image_forensics",
                    priority=2, dependencies=["basic_analysis"], estimated_time=3.0
                ))
            
            if "noise_analysis" not in completed:
                tasks.append(AnalysisTask(
                    file_path=file_path, method="noise_analysis", tool_name="image_forensics",
                    priority=3, dependencies=["basic_analysis"], estimated_time=7.0
                ))
        
        return tasks

    def _create_audio_analysis_tasks(self, file_path: Path, completed: Set[str]) -> List[AnalysisTask]:
        """Create audio-specific analysis tasks"""
        tasks = []
        
        if self.audio_tools:
            if "spectral_analysis" not in completed:
                tasks.append(AnalysisTask(
                    file_path=file_path, method="spectral_analysis", tool_name="audio_analysis",
                    priority=2, dependencies=["basic_analysis"], estimated_time=10.0
                ))
            
            if "lsb_audio" not in completed:
                tasks.append(AnalysisTask(
                    file_path=file_path, method="lsb_audio", tool_name="audio_analysis",
                    priority=2, dependencies=["basic_analysis"], estimated_time=8.0
                ))
        
        return tasks

    def _create_video_analysis_tasks(self, file_path: Path, completed: Set[str]) -> List[AnalysisTask]:
        """Create video-specific analysis tasks"""
        tasks = []
        
        # Video analysis would go here when implemented
        if "video_metadata" not in completed:
            tasks.append(AnalysisTask(
                file_path=file_path, method="video_metadata", tool_name="file_forensics",
                priority=2, dependencies=["basic_analysis"], estimated_time=5.0
            ))
        
        return tasks

    def _create_generic_analysis_tasks(self, file_path: Path, completed: Set[str]) -> List[AnalysisTask]:
        """Create generic file analysis tasks"""
        tasks = []
        
        if self.file_tools:
            if "file_signature" not in completed:
                tasks.append(AnalysisTask(
                    file_path=file_path, method="file_signature", tool_name="file_forensics",
                    priority=2, dependencies=["basic_analysis"], estimated_time=2.0
                ))
        
        if self.crypto_tools:
            if "entropy_analysis" not in completed:
                tasks.append(AnalysisTask(
                    file_path=file_path, method="entropy_analysis", tool_name="crypto_analysis",
                    priority=2, dependencies=["basic_analysis"], estimated_time=4.0
                ))
        
        return tasks

    async def _execute_analysis_tasks(self, tasks: List[AnalysisTask], session_id: str) -> List[Dict[str, Any]]:
        """Execute analysis tasks with dependency management"""
        results = []
        completed_methods = set()
        
        # Create task dependency graph
        task_map = {task.method: task for task in tasks}
        
        # Execute tasks in dependency order
        max_concurrent = self.config.performance.max_concurrent_tasks
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def execute_task(task: AnalysisTask):
            async with semaphore:
                # Wait for dependencies
                for dep in task.dependencies:
                    while dep not in completed_methods:
                        await asyncio.sleep(0.1)
                
                try:
                    result = await self._execute_single_task(task, session_id)
                    if result:
                        results.extend(result if isinstance(result, list) else [result])
                    completed_methods.add(task.method)
                    
                except Exception as e:
                    self.logger.error(f"Task {task.method} failed: {e}")
        
        # Start all tasks
        task_futures = [asyncio.create_task(execute_task(task)) for task in tasks]
        await asyncio.gather(*task_futures, return_exceptions=True)
        
        return results

    async def _execute_single_task(self, task: AnalysisTask, session_id: str) -> Optional[List[Dict[str, Any]]]:
        """Execute a single analysis task"""
        self.logger.info(f"Executing {task.method} on {task.file_path}")
        
        try:
            # Record task start
            await self.checkpoint_manager.save_checkpoint(session_id, {
                'current_task': task.method,
                'file_path': str(task.file_path),
                'timestamp': time.time()
            })
            
            # Execute based on tool
            result = None
            
            if task.tool_name == "classic_stego" and self.classic_tools:
                result = await self._run_classic_tool(task)
            elif task.tool_name == "image_forensics" and self.image_tools:
                result = await self._run_image_tool(task)
            elif task.tool_name == "audio_analysis" and self.audio_tools:
                result = await self._run_audio_tool(task)
            elif task.tool_name == "file_forensics" and self.file_tools:
                result = await self._run_file_tool(task)
            elif task.tool_name == "crypto_analysis" and self.crypto_tools:
                result = await self._run_crypto_tool(task)
            elif task.tool_name == "ml_detector" and self.ml_detector:
                result = await self._run_ml_tool(task)
            elif task.tool_name == "llm_analyzer" and self.llm_analyzer:
                result = await self._run_llm_tool(task)
            else:
                self.logger.warning(f"Tool {task.tool_name} not available for {task.method}")
            
            # Store result
            if result:
                await self.db.store_analysis_result(session_id, task.method, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to execute {task.method}: {e}")
            return None

    async def _run_classic_tool(self, task: AnalysisTask) -> List[Dict[str, Any]]:
        """Run classic steganography tool"""
        if task.method == "steghide":
            return self.classic_tools.run_steghide(task.file_path)
        elif task.method == "outguess":
            return self.classic_tools.run_outguess(task.file_path)
        elif task.method == "zsteg":
            return self.classic_tools.run_zsteg(task.file_path)
        return []

    async def _run_image_tool(self, task: AnalysisTask) -> List[Dict[str, Any]]:
        """Run image forensics tool"""
        if task.method == "ela_analysis":
            return self.image_tools.ela_analysis(task.file_path)
        elif task.method == "noise_analysis":
            return self.image_tools.noise_analysis(task.file_path)
        return []

    async def _run_audio_tool(self, task: AnalysisTask) -> List[Dict[str, Any]]:
        """Run audio analysis tool"""
        if task.method == "spectral_analysis":
            return self.audio_tools.spectral_analysis(task.file_path)
        elif task.method == "lsb_audio":
            return self.audio_tools.lsb_analysis(task.file_path)
        return []

    async def _run_file_tool(self, task: AnalysisTask) -> List[Dict[str, Any]]:
        """Run file forensics tool"""
        if task.method == "file_signature":
            return self.file_tools.signature_analysis(task.file_path)
        elif task.method == "video_metadata":
            return self.file_tools.metadata_analysis(task.file_path)
        return []

    async def _run_crypto_tool(self, task: AnalysisTask) -> List[Dict[str, Any]]:
        """Run cryptographic analysis tool"""
        if task.method == "entropy_analysis":
            return self.crypto_tools.entropy_analysis(task.file_path)
        return []

    async def _run_ml_tool(self, task: AnalysisTask) -> List[Dict[str, Any]]:
        """Run ML detection tool"""
        return await self.ml_detector.detect(task.file_path)

    async def _run_llm_tool(self, task: AnalysisTask) -> List[Dict[str, Any]]:
        """Run LLM analysis tool"""
        return await self.llm_analyzer.analyze(task.file_path)

    async def _post_process_results(self, results: List[Dict[str, Any]], session_id: str) -> List[Dict[str, Any]]:
        """Post-process and correlate analysis results"""
        self.logger.info(f"Post-processing {len(results)} results")
        
        # Group results by confidence and type
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
        
        self.logger.info(f"Found {len(files)} files to analyze")
        
        # Analyze files in batches
        batch_size = self.config.performance.max_concurrent_files
        all_results = []
        
        for i in range(0, len(files), batch_size):
            batch = files[i:i + batch_size]
            
            # Create tasks for this batch
            batch_tasks = [self.analyze(file_path, session_id) for file_path in batch]
            
            # Execute batch
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Collect successful results
            for result in batch_results:
                if isinstance(result, list):
                    all_results.extend(result)
                elif not isinstance(result, Exception):
                    all_results.append(result)
        
        return {
            'total_files': len(files),
            'results': all_results,
            'session_id': session_id
        }

    def can_run(self, task):
        """Check if a task can be run based on available tools"""
        if task.tool_name == "classic_stego":
            return self.classic_tools is not None
        elif task.tool_name == "image_forensics":
            return self.image_tools is not None
        elif task.tool_name == "audio_analysis":
            return self.audio_tools is not None
        elif task.tool_name == "file_forensics":
            return self.file_tools is not None
        elif task.tool_name == "crypto_analysis":
            return self.crypto_tools is not None
        elif task.tool_name == "ml_detector":
            return self.ml_detector is not None
        elif task.tool_name == "llm_analyzer":
            return self.llm_analyzer is not None
        else:
            return False

    async def get_available_tools(self) -> Dict[str, bool]:
        """Get status of all available tools"""
        return {
            'classic_stego': self.classic_tools is not None,
            'image_forensics': self.image_tools is not None,
            'audio_analysis': self.audio_tools is not None,
            'file_forensics': self.file_tools is not None,
            'crypto_analysis': self.crypto_tools is not None,
            'metadata_carving': self.metadata_tools is not None,
            'ml_detector': self.ml_detector is not None,
            'llm_analyzer': self.llm_analyzer is not None,
            'multimodal_classifier': self.multimodal_classifier is not None,
            'cloud_integrations': self.cloud is not None,
            'gpu_manager': self.gpu_manager is not None
        }
