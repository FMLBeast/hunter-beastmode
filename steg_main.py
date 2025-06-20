#!/usr/bin/env python3
"""
StegAnalyzer - Advanced Steganography Detection & Analysis Framework
GPU-Powered, AI-Augmented, Massively Parallel Steganography Analysis Tool
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse
import json
from datetime import datetime

from core.orchestrator import StegOrchestrator
from core.config import Config
from core.database import DatabaseManager
from core.dashboard import Dashboard
from core.reporter import ReportGenerator
from utils.logger import setup_logging
from utils.system_check import SystemChecker

class StegAnalyzer:
    def __init__(self, config_path: str = "config/default.json"):
        self.config = Config(config_path)
        self.db = DatabaseManager(self.config.database)
        self.orchestrator = StegOrchestrator(self.config, self.db)
        self.dashboard = Dashboard(self.config, self.db)
        self.reporter = ReportGenerator(self.config, self.db)
        self.logger = logging.getLogger(__name__)
        
    async def analyze_file(self, file_path: str, target_dir: str = None) -> Dict[str, Any]:
        """Analyze a single file for steganographic content"""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # Create analysis session
        session_id = await self.db.create_session(
            target_path=str(file_path),
            target_dir=target_dir,
            config=self.config.to_dict()
        )
        
        self.logger.info(f"Starting analysis session {session_id} for {file_path}")
        
        # Start dashboard if enabled
        if self.config.dashboard.enabled:
            await self.dashboard.start(session_id)
        
        try:
            # Run orchestrated analysis
            results = await self.orchestrator.analyze(file_path, session_id)
            
            # Generate report
            report = await self.reporter.generate_report(session_id)
            
            self.logger.info(f"Analysis complete. Found {len(results)} findings.")
            return {
                "session_id": session_id,
                "results": results,
                "report_path": report["path"],
                "dashboard_url": self.dashboard.url if self.config.dashboard.enabled else None
            }
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            await self.db.update_session_status(session_id, "failed", str(e))
            raise
        finally:
            if self.config.dashboard.enabled:
                await self.dashboard.stop()
    
    async def analyze_directory(self, directory_path: str) -> Dict[str, Any]:
        """Analyze all files in a directory recursively"""
        directory_path = Path(directory_path)
        if not directory_path.exists() or not directory_path.is_dir():
            raise NotADirectoryError(f"Directory not found: {directory_path}")
        
        # Get all files
        files = []
        for pattern in self.config.analysis.file_patterns:
            files.extend(directory_path.rglob(pattern))
        
        if not files:
            self.logger.warning(f"No files found in {directory_path}")
            return {"results": [], "files_processed": 0}
        
        self.logger.info(f"Found {len(files)} files to analyze")
        
        # Create batch session
        session_id = await self.db.create_session(
            target_path=str(directory_path),
            target_dir=str(directory_path),
            config=self.config.to_dict(),
            batch_mode=True
        )
        
        # Start dashboard
        if self.config.dashboard.enabled:
            await self.dashboard.start(session_id)
        
        try:
            # Process files in parallel batches
            batch_size = self.config.orchestrator.max_concurrent_files
            results = []
            
            for i in range(0, len(files), batch_size):
                batch = files[i:i + batch_size]
                batch_results = await asyncio.gather(*[
                    self.orchestrator.analyze(file_path, session_id)
                    for file_path in batch
                ], return_exceptions=True)
                
                for file_path, result in zip(batch, batch_results):
                    if isinstance(result, Exception):
                        self.logger.error(f"Failed to analyze {file_path}: {result}")
                    else:
                        results.extend(result)
            
            # Generate final report
            report = await self.reporter.generate_report(session_id)
            
            return {
                "session_id": session_id,
                "files_processed": len(files),
                "results": results,
                "report_path": report["path"],
                "dashboard_url": self.dashboard.url if self.config.dashboard.enabled else None
            }
            
        except Exception as e:
            self.logger.error(f"Batch analysis failed: {e}")
            await self.db.update_session_status(session_id, "failed", str(e))
            raise
        finally:
            if self.config.dashboard.enabled:
                await self.dashboard.stop()
    
    async def resume_session(self, session_id: str) -> Dict[str, Any]:
        """Resume a previously interrupted analysis session"""
        session = await self.db.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        if session["status"] == "completed":
            self.logger.info(f"Session {session_id} already completed")
            return await self.get_session_results(session_id)
        
        self.logger.info(f"Resuming session {session_id}")
        
        # Restore configuration
        self.config.update(session["config"])
        
        # Start dashboard
        if self.config.dashboard.enabled:
            await self.dashboard.start(session_id)
        
        try:
            # Resume analysis
            if session["batch_mode"]:
                return await self._resume_batch_analysis(session_id)
            else:
                target_path = Path(session["target_path"])
                results = await self.orchestrator.analyze(target_path, session_id, resume=True)
                report = await self.reporter.generate_report(session_id)
                
                return {
                    "session_id": session_id,
                    "results": results,
                    "report_path": report["path"],
                    "dashboard_url": self.dashboard.url if self.config.dashboard.enabled else None
                }
        finally:
            if self.config.dashboard.enabled:
                await self.dashboard.stop()
    
    async def _resume_batch_analysis(self, session_id: str) -> Dict[str, Any]:
        """Resume batch analysis from checkpoint"""
        # Get incomplete files
        incomplete_files = await self.db.get_incomplete_files(session_id)
        
        if not incomplete_files:
            self.logger.info("All files already processed")
            report = await self.reporter.generate_report(session_id)
            return {
                "session_id": session_id,
                "files_processed": 0,
                "results": [],
                "report_path": report["path"]
            }
        
        # Continue processing
        batch_size = self.config.orchestrator.max_concurrent_files
        results = []
        
        for i in range(0, len(incomplete_files), batch_size):
            batch = incomplete_files[i:i + batch_size]
            batch_results = await asyncio.gather(*[
                self.orchestrator.analyze(Path(file_path), session_id, resume=True)
                for file_path in batch
            ], return_exceptions=True)
            
            for file_path, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    self.logger.error(f"Failed to analyze {file_path}: {result}")
                else:
                    results.extend(result)
        
        report = await self.reporter.generate_report(session_id)
        
        return {
            "session_id": session_id,
            "files_processed": len(incomplete_files),
            "results": results,
            "report_path": report["path"]
        }
    
    async def get_session_results(self, session_id: str) -> Dict[str, Any]:
        """Get results for a completed session"""
        session = await self.db.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        results = await self.db.get_session_results(session_id)
        
        return {
            "session_id": session_id,
            "status": session["status"],
            "results": results,
            "created_at": session["created_at"],
            "completed_at": session.get("completed_at")
        }
    
    async def list_sessions(self) -> List[Dict[str, Any]]:
        """List all analysis sessions"""
        return await self.db.list_sessions()
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.db.close()
        if hasattr(self, 'dashboard'):
            await self.dashboard.stop()

async def main():
    parser = argparse.ArgumentParser(
        description="StegAnalyzer - Advanced Steganography Detection Framework"
    )
    parser.add_argument("target", help="File or directory to analyze")
    parser.add_argument("-c", "--config", default="config/default.json", help="Configuration file")
    parser.add_argument("-o", "--output", help="Output directory for results")
    parser.add_argument("-r", "--resume", help="Resume session ID")
    parser.add_argument("-l", "--list-sessions", action="store_true", help="List all sessions")
    parser.add_argument("-s", "--session-results", help="Get results for session ID")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--check-system", action="store_true", help="Check system requirements")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)
    
    # System check
    if args.check_system:
        checker = SystemChecker()
        status = await checker.check_all()
        print(json.dumps(status, indent=2))
        return
    
    try:
        analyzer = StegAnalyzer(args.config)
        
        if args.list_sessions:
            sessions = await analyzer.list_sessions()
            print(json.dumps(sessions, indent=2, default=str))
            return
        
        if args.session_results:
            results = await analyzer.get_session_results(args.session_results)
            print(json.dumps(results, indent=2, default=str))
            return
        
        if args.resume:
            results = await analyzer.resume_session(args.resume)
        elif Path(args.target).is_file():
            results = await analyzer.analyze_file(args.target, args.output)
        elif Path(args.target).is_dir():
            results = await analyzer.analyze_directory(args.target)
        else:
            print(f"Error: {args.target} is not a valid file or directory")
            return 1
        
        print(f"\nAnalysis completed!")
        print(f"Session ID: {results['session_id']}")
        print(f"Results found: {len(results.get('results', []))}")
        if 'report_path' in results:
            print(f"Report: {results['report_path']}")
        if 'dashboard_url' in results and results['dashboard_url']:
            print(f"Dashboard: {results['dashboard_url']}")
            
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
        return 130
    except Exception as e:
        logging.error(f"Analysis failed: {e}")
        return 1
    finally:
        if 'analyzer' in locals():
            await analyzer.cleanup()

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))