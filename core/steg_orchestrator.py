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

from tools.classic_stego_tools import ClassicStegoTools
from tools.image_forensics_tools import ImageForensicsTools
from tools.audio_analysis_tools import AudioAnalysisTools
from tools.file_forensics_tools import FileForensicsTools
from tools.crypto_analysis_tools import CryptoAnalysisTools
from tools.metadata_carving import MetadataCarving
from ai.ml_detector import MLStegDetector
from ai.llm_analyzer import LLMAnalyzer
from ai.multimodal_classifier import MultimodalClassifier
from cloud.cloud_integrations import CloudIntegrations
from core.file_analyzer import FileAnalyzer
from core.graph_tracker import GraphTracker
from utils.gpu_manager import GPUManager
from utils.checkpoint_manager import CheckpointManager

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
        
        # Initialize tool managers
        self.classic_tools = ClassicStegoTools(config)
        self.image_tools = ImageForensicsTools(config)
        self.audio_tools = AudioAnalysisTools(config)
        self.file_tools = FileForensicsTools(config)
        self.crypto_tools = CryptoAnalysisTools(config)
        self.metadata_tools = MetadataCarving(config)
        
        # Initialize AI components
        self.ml_detector = MLStegDetector(config)
        self.llm_analyzer = LLMAnalyzer(config)
        self.multimodal_classifier = MultimodalClassifier(config)
        
        # Initialize cloud integrations
        self.cloud = CloudIntegrations(config) if config.cloud.enabled else None
        
        # Core components
        self.file_analyzer = FileAnalyzer(config)
        self.graph_tracker = GraphTracker(database)
        self.gpu_manager = GPUManager()
        self.checkpoint_manager = CheckpointManager(database)
        
        # Execution pools
        self.cpu_executor = ThreadPoolExecutor(max_workers=config.orchestrator.max_cpu_workers)
        self.gpu_executor = ThreadPoolExecutor(max_workers=config.orchestrator.max_gpu_workers)
        self.process_executor = ProcessPoolExecutor(max_workers=mp.cpu_count())
        
        # State tracking
        self.active_sessions = {}
        self.task_queue = asyncio.Queue()
        self.results_queue = asyncio.Queue()
        self.running_tasks = set()
        
    async def analyze(self, file_path: Path, session_id: str, resume: bool = False) -> List[Dict[str, Any]]:
        """Main analysis entry point"""
        start_time = time.time()
        
        try:
            # Initialize session tracking
            if not resume:
                await self._initialize_session(file_path, session_id)
            else:
                await self._restore_session(session_id)
            
            # Get file characteristics
            file_info = await self.file_analyzer.analyze_file(file_path)
            await self.db.store_file_info(session_id, str(file_path), file_info)
            
            # Create analysis plan
            analysis_plan = await self._create_analysis_plan(file_path, file_info, session_id, resume)
            
            self.logger.info(f"Created analysis plan with {len(analysis_plan)} tasks for {file_path}")
            
            # Execute analysis pipeline
            results = await self._execute_analysis_pipeline(analysis_plan, session_id)
            
            # Post-process and correlate findings
            correlated_results = await self._correlate_findings(results, session_id)
            
            # AI-enhanced analysis of findings
            enhanced_results = await self._ai_enhance_findings(correlated_results, file_path, session_id)
            
            # Update session completion
            await self.db.update_session_status(session_id, "completed")
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"Analysis completed in {elapsed_time:.2f}s with {len(enhanced_results)} findings")
            
            return enhanced_results
            
        except Exception as e:
            self.logger.error(f"Analysis failed for {file_path}: {e}")
            await self.db.update_session_status(session_id, "failed", str(e))
            raise
    
    async def _initialize_session(self, file_path: Path, session_id: str):
        """Initialize a new analysis session"""
        self.active_sessions[session_id] = {
            "start_time": time.time(),
            "file_path": file_path,
            "completed_tasks": set(),
            "failed_tasks": set(),
            "findings": [],
            "graph_nodes": set(),
            "checkpoints": []
        }
        
        # Initialize graph tracking
        await self.graph_tracker.create_file_node(session_id, str(file_path))
    
    async def _restore_session(self, session_id: str):
        """Restore session state from checkpoint"""
        checkpoint = await self.checkpoint_manager.load_checkpoint(session_id)
        if checkpoint:
            self.active_sessions[session_id] = checkpoint
            self.logger.info(f"Restored session {session_id} from checkpoint")
        else:
            raise ValueError(f"No checkpoint found for session {session_id}")
    
    async def _create_analysis_plan(self, file_path: Path, file_info: Dict, session_id: str, resume: bool) -> List[AnalysisTask]:
        """Create comprehensive analysis plan based on file characteristics"""
        tasks = []
        completed_tasks = set()
        
        if resume:
            completed_tasks = self.active_sessions[session_id].get("completed_tasks", set())
        
        file_type = file_info["type"]
        file_size = file_info["size"]
        
        # Priority levels: 1=highest, 5=lowest
        
        # Essential file analysis (always run first)
        if "basic_analysis" not in completed_tasks:
            tasks.append(AnalysisTask(
                file_path=file_path,
                method="basic_analysis",
                tool_name="file_analyzer",
                priority=1,
                dependencies=[],
                estimated_time=1.0
            ))
        
        # Cloud hash lookups (fast, high value)
        if self.cloud and "hash_lookup" not in completed_tasks:
            tasks.append(AnalysisTask(
                file_path=file_path,
                method="hash_lookup",
                tool_name="cloud",
                priority=1,
                dependencies=[],
                estimated_time=2.0
            ))
        
        # File format specific analysis
        if file_type.startswith("image/"):
            tasks.extend(self._create_image_analysis_tasks(file_path, completed_tasks))
        elif file_type.startswith("audio/"):
            tasks.extend(self._create_audio_analysis_tasks(file_path, completed_tasks))
        elif file_type == "application/pdf":
            tasks.extend(self._create_pdf_analysis_tasks(file_path, completed_tasks))
        
        # Classic steganography tools (universal)
        tasks.extend(self._create_classic_stego_tasks(file_path, completed_tasks))
        
        # Metadata extraction
        if "metadata_extraction" not in completed_tasks:
            tasks.append(AnalysisTask(
                file_path=file_path,
                method="metadata_extraction",
                tool_name="metadata_carving",
                priority=2,
                dependencies=["basic_analysis"],
                estimated_time=3.0
            ))
        
        # Deep learning analysis (GPU intensive)
        if self.gpu_manager.gpu_available():
            tasks.extend(self._create_ml_analysis_tasks(file_path, file_type, completed_tasks))
        
        # Cryptographic analysis
        tasks.extend(self._create_crypto_analysis_tasks(file_path, completed_tasks))
        
        # Advanced forensics
        tasks.extend(self._create_forensics_tasks(file_path, completed_tasks))
        
        # AI analysis (depends on other findings)
        tasks.extend(self._create_ai_analysis_tasks(file_path, completed_tasks))
        
        # Sort tasks by priority and dependencies
        tasks = self._sort_tasks_by_dependencies(tasks)
        
        return tasks
    
    def _create_image_analysis_tasks(self, file_path: Path, completed: Set[str]) -> List[AnalysisTask]:
        """Create image-specific analysis tasks"""
        tasks = []
        
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
                priority=2, dependencies=["basic_analysis"], estimated_time=10.0
            ))
        
        if "stegdetect" not in completed:
            tasks.append(AnalysisTask(
                file_path=file_path, method="stegdetect", tool_name="image_forensics",
                priority=2, dependencies=["basic_analysis"], estimated_time=8.0
            ))
        
        if "lsb_analysis" not in completed:
            tasks.append(AnalysisTask(
                file_path=file_path, method="lsb_analysis", tool_name="image_forensics",
                priority=2, dependencies=["basic_analysis"], estimated_time=15.0
            ))
        
        if "exif_analysis" not in completed:
            tasks.append(AnalysisTask(
                file_path=file_path, method="exif_analysis", tool_name="image_forensics",
                priority=1, dependencies=["basic_analysis"], estimated_time=2.0
            ))
        
        if "noise_analysis" not in completed:
            tasks.append(AnalysisTask(
                file_path=file_path, method="noise_analysis", tool_name="image_forensics",
                priority=3, dependencies=["basic_analysis"], gpu_required=True, estimated_time=30.0
            ))
        
        return tasks
    
    def _create_audio_analysis_tasks(self, file_path: Path, completed: Set[str]) -> List[AnalysisTask]:
        """Create audio-specific analysis tasks"""
        tasks = []
        
        if "spectral_analysis" not in completed:
            tasks.append(AnalysisTask(
                file_path=file_path, method="spectral_analysis", tool_name="audio_analysis",
                priority=2, dependencies=["basic_analysis"], estimated_time=20.0
            ))
        
        if "lsb_audio" not in completed:
            tasks.append(AnalysisTask(
                file_path=file_path, method="lsb_audio", tool_name="audio_analysis",
                priority=2, dependencies=["basic_analysis"], estimated_time=25.0
            ))
        
        if "deep_speech" not in completed:
            tasks.append(AnalysisTask(
                file_path=file_path, method="deep_speech", tool_name="audio_analysis",
                priority=3, dependencies=["basic_analysis"], gpu_required=True, estimated_time=60.0
            ))
        
        if "echo_hiding" not in completed:
            tasks.append(AnalysisTask(
                file_path=file_path, method="echo_hiding", tool_name="audio_analysis",
                priority=3, dependencies=["spectral_analysis"], estimated_time=30.0
            ))
        
        return tasks
    
    def _create_pdf_analysis_tasks(self, file_path: Path, completed: Set[str]) -> List[AnalysisTask]:
        """Create PDF-specific analysis tasks"""
        tasks = []
        
        if "pdf_structure" not in completed:
            tasks.append(AnalysisTask(
                file_path=file_path, method="pdf_structure", tool_name="file_forensics",
                priority=2, dependencies=["basic_analysis"], estimated_time=10.0
            ))
        
        if "pdf_streams" not in completed:
            tasks.append(AnalysisTask(
                file_path=file_path, method="pdf_streams", tool_name="file_forensics",
                priority=2, dependencies=["pdf_structure"], estimated_time=15.0
            ))
        
        if "pdf_javascript" not in completed:
            tasks.append(AnalysisTask(
                file_path=file_path, method="pdf_javascript", tool_name="file_forensics",
                priority=3, dependencies=["pdf_structure"], estimated_time=5.0
            ))
        
        return tasks
    
    def _create_classic_stego_tasks(self, file_path: Path, completed: Set[str]) -> List[AnalysisTask]:
        """Create classic steganography tool tasks"""
        tasks = []
        
        classic_tools = ["binwalk", "foremost", "strings", "hexdump_analysis"]
        
        for tool in classic_tools:
            if tool not in completed:
                tasks.append(AnalysisTask(
                    file_path=file_path, method=tool, tool_name="classic_stego",
                    priority=2, dependencies=["basic_analysis"], 
                    estimated_time=8.0 if tool == "binwalk" else 3.0
                ))
        
        if "stegcracker" not in completed:
            tasks.append(AnalysisTask(
                file_path=file_path, method="stegcracker", tool_name="classic_stego",
                priority=4, dependencies=["steghide"], estimated_time=300.0  # Can be very slow
            ))
        
        return tasks
    
    def _create_ml_analysis_tasks(self, file_path: Path, file_type: str, completed: Set[str]) -> List[AnalysisTask]:
        """Create machine learning analysis tasks"""
        tasks = []
        
        if "cnn_steg_detection" not in completed:
            tasks.append(AnalysisTask(
                file_path=file_path, method="cnn_steg_detection", tool_name="ml_detector",
                priority=3, dependencies=["basic_analysis"], gpu_required=True, estimated_time=45.0
            ))
        
        if "multimodal_classification" not in completed:
            tasks.append(AnalysisTask(
                file_path=file_path, method="multimodal_classification", tool_name="multimodal_classifier",
                priority=3, dependencies=["basic_analysis"], gpu_required=True, estimated_time=30.0
            ))
        
        if file_type.startswith("image/") and "noiseprint" not in completed:
            tasks.append(AnalysisTask(
                file_path=file_path, method="noiseprint", tool_name="ml_detector",
                priority=3, dependencies=["basic_analysis"], gpu_required=True, estimated_time=60.0
            ))
        
        return tasks
    
    def _create_crypto_analysis_tasks(self, file_path: Path, completed: Set[str]) -> List[AnalysisTask]:
        """Create cryptographic analysis tasks"""
        tasks = []
        
        if "entropy_analysis" not in completed:
            tasks.append(AnalysisTask(
                file_path=file_path, method="entropy_analysis", tool_name="crypto_analysis",
                priority=2, dependencies=["basic_analysis"], estimated_time=5.0
            ))
        
        if "pattern_detection" not in completed:
            tasks.append(AnalysisTask(
                file_path=file_path, method="pattern_detection", tool_name="crypto_analysis",
                priority=3, dependencies=["entropy_analysis"], estimated_time=20.0
            ))
        
        if "key_search" not in completed:
            tasks.append(AnalysisTask(
                file_path=file_path, method="key_search", tool_name="crypto_analysis",
                priority=4, dependencies=["pattern_detection"], estimated_time=120.0
            ))
        
        return tasks
    
    def _create_forensics_tasks(self, file_path: Path, completed: Set[str]) -> List[AnalysisTask]:
        """Create advanced forensics tasks"""
        tasks = []
        
        if "bulk_extractor" not in completed:
            tasks.append(AnalysisTask(
                file_path=file_path, method="bulk_extractor", tool_name="metadata_carving",
                priority=3, dependencies=["basic_analysis"], estimated_time=30.0
            ))
        
        if "file_carving" not in completed:
            tasks.append(AnalysisTask(
                file_path=file_path, method="file_carving", tool_name="file_forensics",
                priority=3, dependencies=["basic_analysis"], estimated_time=40.0
            ))
        
        if "signature_analysis" not in completed:
            tasks.append(AnalysisTask(
                file_path=file_path, method="signature_analysis", tool_name="file_forensics",
                priority=2, dependencies=["basic_analysis"], estimated_time=10.0
            ))
        
        return tasks
    
    def _create_ai_analysis_tasks(self, file_path: Path, completed: Set[str]) -> List[AnalysisTask]:
        """Create AI-powered analysis tasks"""
        tasks = []
        
        if "llm_content_analysis" not in completed:
            tasks.append(AnalysisTask(
                file_path=file_path, method="llm_content_analysis", tool_name="llm_analyzer",
                priority=4, dependencies=["metadata_extraction"], gpu_required=True, estimated_time=90.0
            ))
        
        if "anomaly_detection" not in completed:
            tasks.append(AnalysisTask(
                file_path=file_path, method="anomaly_detection", tool_name="ml_detector",
                priority=4, dependencies=["entropy_analysis", "pattern_detection"], 
                gpu_required=True, estimated_time=60.0
            ))
        
        return tasks
    
    def _sort_tasks_by_dependencies(self, tasks: List[AnalysisTask]) -> List[AnalysisTask]:
        """Sort tasks respecting dependencies and priorities"""
        # Create dependency graph
        task_map = {task.method: task for task in tasks}
        sorted_tasks = []
        completed = set()
        
        def can_run(task):
            return all(dep in completed for dep in task.dependencies)
        
        # Process tasks in waves
        remaining = set(task.method for task in tasks)
        
        while remaining:
            wave = []
            for task_name in list(remaining):
                task = task_map[task_name]
                if can_run(task):
                    wave.append(task)
                    remaining.remove(task_name)
                    completed.add(task_name)
            
            if not wave:
                # Break circular dependencies by picking lowest priority
                task_name = min(remaining, key=lambda x: task_map[x].priority)
                task = task_map[task_name]
                wave.append(task)
                remaining.remove(task_name)
                completed.add(task_name)
            
            # Sort wave by priority
            wave.sort(key=lambda x: (x.priority, x.estimated_time))
            sorted_tasks.extend(wave)
        
        return sorted_tasks
    
    async def _execute_analysis_pipeline(self, analysis_plan: List[AnalysisTask], session_id: str) -> List[Dict[str, Any]]:
        """Execute the analysis pipeline with parallel processing"""
        all_results = []
        
        # Create task semaphores
        cpu_semaphore = asyncio.Semaphore(self.config.orchestrator.max_cpu_workers)
        gpu_semaphore = asyncio.Semaphore(self.config.orchestrator.max_gpu_workers)
        
        # Group tasks by dependency level
        task_levels = self._group_tasks_by_level(analysis_plan)
        
        for level, tasks in enumerate(task_levels):
            self.logger.info(f"Executing level {level} with {len(tasks)} tasks")
            
            # Execute tasks in parallel within each level
            level_tasks = []
            for task in tasks:
                if task.gpu_required:
                    level_tasks.append(self._execute_gpu_task(task, session_id, gpu_semaphore))
                else:
                    level_tasks.append(self._execute_cpu_task(task, session_id, cpu_semaphore))
            
            # Wait for level completion
            level_results = await asyncio.gather(*level_tasks, return_exceptions=True)
            
            # Process results
            for task, result in zip(tasks, level_results):
                if isinstance(result, Exception):
                    self.logger.error(f"Task {task.method} failed: {result}")
                    await self.db.record_task_failure(session_id, task.method, str(result))
                    self.active_sessions[session_id]["failed_tasks"].add(task.method)
                else:
                    if result:  # Only add non-empty results
                        all_results.extend(result if isinstance(result, list) else [result])
                    self.active_sessions[session_id]["completed_tasks"].add(task.method)
                    await self.db.record_task_completion(session_id, task.method)
            
            # Create checkpoint after each level
            await self.checkpoint_manager.save_checkpoint(session_id, self.active_sessions[session_id])
        
        return all_results
    
    def _group_tasks_by_level(self, tasks: List[AnalysisTask]) -> List[List[AnalysisTask]]:
        """Group tasks by dependency level for parallel execution"""
        task_map = {task.method: task for task in tasks}
        levels = []
        completed = set()
        remaining = set(task.method for task in tasks)
        
        while remaining:
            current_level = []
            for task_name in list(remaining):
                task = task_map[task_name]
                if all(dep in completed for dep in task.dependencies):
                    current_level.append(task)
                    remaining.remove(task_name)
                    completed.add(task_name)
            
            if not current_level:
                # Handle circular dependencies
                task_name = remaining.pop()
                current_level.append(task_map[task_name])
                completed.add(task_name)
            
            levels.append(current_level)
        
        return levels
    
    async def _execute_cpu_task(self, task: AnalysisTask, session_id: str, semaphore: asyncio.Semaphore) -> List[Dict[str, Any]]:
        """Execute CPU-bound task"""
        async with semaphore:
            start_time = time.time()
            
            try:
                # Get appropriate tool
                tool = getattr(self, task.tool_name)
                
                # Execute task
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self.cpu_executor,
                    tool.execute_method,
                    task.method,
                    task.file_path
                )
                
                elapsed = time.time() - start_time
                self.logger.debug(f"Task {task.method} completed in {elapsed:.2f}s")
                
                # Store results
                if result:
                    await self.db.store_findings(session_id, task.method, result)
                    await self.graph_tracker.add_findings(session_id, task.method, result)
                
                return result if isinstance(result, list) else [result] if result else []
                
            except Exception as e:
                elapsed = time.time() - start_time
                self.logger.error(f"Task {task.method} failed after {elapsed:.2f}s: {e}")
                raise
    
    async def _execute_gpu_task(self, task: AnalysisTask, session_id: str, semaphore: asyncio.Semaphore) -> List[Dict[str, Any]]:
        """Execute GPU-bound task"""
        async with semaphore:
            # Ensure GPU is available
            gpu_id = await self.gpu_manager.acquire_gpu()
            if gpu_id is None:
                raise RuntimeError("No GPU available for task")
            
            try:
                start_time = time.time()
                
                # Get appropriate tool
                tool = getattr(self, task.tool_name)
                
                # Execute task with GPU
                result = await tool.execute_method_async(task.method, task.file_path, gpu_id)
                
                elapsed = time.time() - start_time
                self.logger.debug(f"GPU task {task.method} completed in {elapsed:.2f}s")
                
                # Store results
                if result:
                    await self.db.store_findings(session_id, task.method, result)
                    await self.graph_tracker.add_findings(session_id, task.method, result)
                
                return result if isinstance(result, list) else [result] if result else []
                
            finally:
                await self.gpu_manager.release_gpu(gpu_id)
    
    async def _correlate_findings(self, results: List[Dict[str, Any]], session_id: str) -> List[Dict[str, Any]]:
        """Correlate and cross-reference findings"""
        if not results:
            return results
        
        # Group findings by type and confidence
        findings_by_type = {}
        for result in results:
            finding_type = result.get("type", "unknown")
            if finding_type not in findings_by_type:
                findings_by_type[finding_type] = []
            findings_by_type[finding_type].append(result)
        
        # Look for correlated evidence
        correlated = []
        for finding_type, findings in findings_by_type.items():
            if len(findings) > 1:
                # Multiple tools found the same type of evidence
                correlation = {
                    "type": "correlation",
                    "finding_type": finding_type,
                    "evidence_count": len(findings),
                    "methods": [f.get("method", "unknown") for f in findings],
                    "confidence": min(f.get("confidence", 0.5) for f in findings) + 0.2,
                    "details": f"Multiple methods ({len(findings)}) detected {finding_type}",
                    "correlated_findings": findings
                }
                correlated.append(correlation)
        
        # Add correlation data to graph
        if correlated:
            await self.graph_tracker.add_correlations(session_id, correlated)
        
        return results + correlated
    
    async def _ai_enhance_findings(self, results: List[Dict[str, Any]], file_path: Path, session_id: str) -> List[Dict[str, Any]]:
        """Use AI to enhance and interpret findings"""
        if not results:
            return results
        
        try:
            # Use LLM to analyze findings context
            context_analysis = await self.llm_analyzer.analyze_findings_context(results, file_path)
            
            # Use multimodal AI for deeper file understanding
            file_analysis = await self.multimodal_classifier.analyze_file_comprehensive(file_path)
            
            # Combine AI insights
            ai_insights = {
                "type": "ai_analysis",
                "method": "ai_enhancement",
                "confidence": 0.8,
                "context_analysis": context_analysis,
                "file_analysis": file_analysis,
                "summary": context_analysis.get("summary", ""),
                "recommendations": context_analysis.get("recommendations", []),
                "risk_assessment": context_analysis.get("risk_level", "low")
            }
            
            # Store AI insights
            await self.db.store_findings(session_id, "ai_enhancement", [ai_insights])
            
            return results + [ai_insights]
            
        except Exception as e:
            self.logger.error(f"AI enhancement failed: {e}")
            return results
    
    async def cleanup(self):
        """Cleanup resources"""
        self.cpu_executor.shutdown(wait=True)
        self.gpu_executor.shutdown(wait=True)
        self.process_executor.shutdown(wait=True)
        await self.gpu_manager.cleanup()
        
        if self.cloud:
            await self.cloud.cleanup()