#!/usr/bin/env python3
"""
Orchestrator integration for Cascade Analyzer
Add this code to your existing core/orchestrator.py
"""

# Add these imports to the top of orchestrator.py
from tools.cascade_analyzer import CascadeAnalyzer

class StegOrchestrator:
    """Enhanced orchestrator with cascade analysis support"""
    
    def __init__(self, config):
        # ... existing initialization code ...
        
        # Add cascade analyzer initialization
        try:
            self.cascade_analyzer = CascadeAnalyzer(config)
            self.logger.info("Cascade analyzer initialized successfully")
        except Exception as e:
            self.logger.warning(f"Failed to initialize cascade analyzer: {e}")
            self.cascade_analyzer = None
    
    # Add this method to create cascade analysis tasks
    def _create_cascade_analysis_tasks(self, file_path: Path, completed: Set[str]) -> List[AnalysisTask]:
        """Create cascade analysis tasks for recursive extraction"""
        tasks = []
        
        if self.cascade_analyzer and "cascade_analyze" not in completed:
            # Only run cascade on image files or when specifically requested
            if (file_path.suffix.lower() in {'.png', '.bmp', '.gif', '.tiff', '.tif', '.webp'} or
                getattr(self.config, 'force_cascade', False)):
                
                tasks.append(AnalysisTask(
                    file_path=file_path,
                    method="cascade_analyze", 
                    tool_name="cascade_analyzer",
                    priority=1,  # High priority for comprehensive analysis
                    dependencies=["basic_analysis"],
                    estimated_time=60.0  # Cascade can take a while
                ))
        
        return tasks
    
    # Modify the existing _create_analysis_tasks method to include cascade
    def _create_analysis_tasks(self, file_path: Path, completed_set: Set[str]) -> List[AnalysisTask]:
        """Create analysis tasks based on file type and available tools"""
        tasks = []
        file_ext = file_path.suffix.lower()
        
        # ... existing task creation code ...
        
        # Add cascade analysis tasks
        cascade_tasks = self._create_cascade_analysis_tasks(file_path, completed_set)
        tasks.extend(cascade_tasks)
        
        return sorted(tasks, key=lambda t: t.priority)
    
    # Modify the existing analyze method to handle cascade
    async def analyze(self, file_path: Path, session_id: str = None) -> List[Dict[str, Any]]:
        """Enhanced analyze method with cascade support"""
        
        # ... existing analysis code ...
        
        # If cascade mode is enabled, run cascade analysis
        if (hasattr(self.config, 'cascade_mode') and self.config.cascade_mode and
            self.cascade_analyzer):
            
            self.logger.info("Running cascade analysis mode")
            cascade_results = await self.cascade_analyzer.cascade_analyze(file_path, session_id)
            
            # Merge cascade results with regular results
            all_results.extend(cascade_results)
            
            # Store cascade results in database
            if session_id:
                for result in cascade_results:
                    await self.db.store_result(session_id, file_path, result)
        
        return all_results
    
    # Add cascade-specific analysis method
    async def analyze_cascade(self, file_path: Path, session_id: str = None, max_depth: int = None) -> Dict[str, Any]:
        """Run pure cascade analysis on a file"""
        
        if not self.cascade_analyzer:
            raise ValueError("Cascade analyzer not available")
        
        self.logger.info(f"Starting cascade analysis on {file_path}")
        
        # Override max depth if specified
        if max_depth:
            original_depth = self.cascade_analyzer.max_depth
            self.cascade_analyzer.max_depth = max_depth
        
        try:
            # Run cascade analysis
            results = await self.cascade_analyzer.cascade_analyze(file_path, session_id)
            
            # Store results
            if session_id:
                for result in results:
                    await self.db.store_result(session_id, file_path, result)
            
            # Generate cascade report
            cascade_summary = self._generate_cascade_summary(results)
            
            return {
                'file_path': str(file_path),
                'session_id': session_id,
                'cascade_results': results,
                'summary': cascade_summary,
                'analysis_complete': True
            }
            
        finally:
            # Restore original depth
            if max_depth:
                self.cascade_analyzer.max_depth = original_depth
    
    def _generate_cascade_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary of cascade analysis results"""
        
        if not results:
            return {'total_files': 0, 'total_findings': 0, 'max_depth': 0}
        
        total_files = len(set(r.get('file_hash', '') for r in results))
        total_findings = sum(1 for r in results if r.get('zsteg_findings', 0) > 0)
        max_depth = max(r.get('depth', 0) for r in results)
        total_extracted = sum(r.get('binwalk_extractions', 0) for r in results)
        
        # File tree structure
        file_tree = {}
        for result in results:
            if result.get('cascade_tree'):
                file_tree.update(result['cascade_tree'])
        
        return {
            'total_files_analyzed': total_files,
            'total_zsteg_findings': total_findings,
            'total_extracted_files': total_extracted,
            'max_depth_reached': max_depth,
            'extraction_tree': file_tree,
            'high_confidence_results': [
                r for r in results if r.get('confidence', 0) > 0.7
            ]
        }
    
    # Update the get_available_tools method
    async def get_available_tools(self) -> Dict[str, bool]:
        """Get status of all available tools including cascade"""
        tools = {
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
            'gpu_manager': self.gpu_manager is not None,
            'cascade_analyzer': self.cascade_analyzer is not None
        }
        
        # Add cascade tool details
        if self.cascade_analyzer:
            cascade_info = self.cascade_analyzer.get_tool_info()
            tools['cascade_details'] = cascade_info
        
        return tools


# Example of how to modify the AnalysisTask class to support cascade
class AnalysisTask:
    """Enhanced analysis task with cascade support"""
    
    def __init__(self, file_path: Path, method: str, tool_name: str,
                 priority: int = 1, dependencies: List[str] = None, 
                 estimated_time: float = 1.0, cascade_depth: int = None):
        self.file_path = file_path
        self.method = method
        self.tool_name = tool_name
        self.priority = priority
        self.dependencies = dependencies or []
        self.estimated_time = estimated_time
        self.cascade_depth = cascade_depth  # For cascade-specific tasks
        self.created_at = time.time()


# Add cascade-specific execution logic
async def execute_cascade_task(self, task: AnalysisTask, session_id: str) -> List[Dict[str, Any]]:
    """Execute cascade analysis task"""
    
    if task.tool_name != "cascade_analyzer" or not self.cascade_analyzer:
        return []
    
    try:
        self.logger.info(f"Executing cascade task: {task.method} on {task.file_path.name}")
        
        # Call the appropriate cascade method
        if task.method == "cascade_analyze":
            results = await self.cascade_analyzer.cascade_analyze(task.file_path, session_id)
        else:
            self.logger.warning(f"Unknown cascade method: {task.method}")
            return []
        
        # Store task completion
        await self.db.mark_task_complete(session_id, task.file_path, task.method)
        
        return results
        
    except Exception as e:
        self.logger.error(f"Cascade task failed: {e}")
        return [{
            "type": "cascade_error",
            "method": task.method,
            "tool_name": "cascade_analyzer",
            "confidence": 0.0,
            "details": f"Cascade task execution failed: {str(e)}",
            "file_path": str(task.file_path)
        }]
