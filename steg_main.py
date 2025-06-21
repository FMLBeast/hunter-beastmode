#!/usr/bin/env python3
"""
StegAnalyzer - Advanced Steganography Detection & Analysis Framework
GPU-Powered, AI-Augmented, Massively Parallel Steganography Analysis Tool
"""

import asyncio
import logging
import sys
import os
import json
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse
from datetime import datetime

# Core imports
from config.steg_config import Config
from core.database import DatabaseManager
from core.orchestrator import StegOrchestrator

# Optional imports with fallbacks
try:
    from core.dashboard import Dashboard
except ImportError:
    Dashboard = None

try:
    from core.reporter import ReportGenerator
except ImportError:
    ReportGenerator = None

try:
    from core.cascading_analyzer import CascadingAnalyzer
except ImportError:
    CascadingAnalyzer = None

def setup_logging(level=logging.INFO):
    """Setup logging configuration"""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s"
    )
    logging.getLogger("steg_main").info("Logging initialized.")

class SystemChecker:
    """System requirements checker"""
    
    @staticmethod
    async def check_all():
        """Check all system requirements"""
        status = {
            "python_version": sys.version,
            "platform": sys.platform,
            "tools": {},
            "dependencies": {},
            "gpu": {"available": False, "cuda": False}
        }
        
        # Check tools
        import subprocess
        tools = ["steghide", "outguess", "zsteg", "binwalk", "exiftool", "strings"]
        for tool in tools:
            try:
                result = subprocess.run([tool, "--version"], capture_output=True, timeout=5)
                status["tools"][tool] = result.returncode == 0
            except:
                status["tools"][tool] = False
        
        # Check Python dependencies
        deps = ["PIL", "numpy", "scipy", "magic", "yara"]
        for dep in deps:
            try:
                __import__(dep)
                status["dependencies"][dep] = True
            except ImportError:
                status["dependencies"][dep] = False
        
        # Check GPU
        try:
            import torch
            status["gpu"]["available"] = torch.cuda.is_available()
            status["gpu"]["cuda"] = torch.version.cuda if torch.cuda.is_available() else None
        except ImportError:
            pass
        
        return status

class StegAnalyzer:
    """Main steganography analyzer class"""
    
    def __init__(self, config_path: str = "config/default.json"):
        self.logger = logging.getLogger(__name__)
        
        # Load configuration properly
        try:
            self.config = self._load_config(config_path)
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            self.config = Config()  # Use default config
        
        # Initialize database
        self.db = DatabaseManager(self.config.database)
        
        # Initialize orchestrator
        self.orchestrator = StegOrchestrator(self.config, self.db)
        
        # Initialize dashboard if available
        self.dashboard = None
        if Dashboard and getattr(self.config.dashboard, 'enabled', False):
            try:
                self.dashboard = Dashboard(self.config, self.db)
            except Exception as e:
                self.logger.warning(f"Dashboard initialization failed: {e}")
        
        # Initialize reporter if available
        self.reporter = None
        if ReportGenerator:
            try:
                self.reporter = ReportGenerator(self.config, self.db)
            except Exception as e:
                self.logger.warning(f"Reporter initialization failed: {e}")
    
    def _load_config(self, config_path: str) -> Config:
        """Load configuration file properly"""
        config_file = Path(config_path)
        
        if config_file.exists():
            # Load config using proper method
            config = Config()
            try:
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                
                # Update config objects properly
                for section, data in config_data.items():
                    if hasattr(config, section):
                        config_obj = getattr(config, section)
                        if hasattr(config_obj, '__dict__'):
                            # Update dataclass attributes
                            for key, value in data.items():
                                if hasattr(config_obj, key):
                                    setattr(config_obj, key, value)
                
                self.logger.info(f"Configuration loaded from {config_path}")
                return config
                
            except Exception as e:
                self.logger.error(f"Failed to parse config {config_path}: {e}")
                return Config()  # Return default config
        else:
            # Create default config
            default_config = Config()
            config_file.parent.mkdir(parents=True, exist_ok=True)
            try:
                default_config.save_config()
                self.logger.info(f"Created default config at {config_path}")
            except Exception as e:
                self.logger.warning(f"Could not save default config: {e}")
            return default_config
    
    async def analyze_file(self, file_path: str, target_dir: str = None) -> Dict[str, Any]:
        """Analyze a single file for steganographic content"""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Create analysis session
        session_id = str(uuid.uuid4())
        try:
            session_id = await self.db.create_session(
                target_path=str(file_path),
                target_dir=target_dir,
                config=getattr(self.config, 'to_dict', lambda: {})()
            )
        except Exception as e:
            self.logger.warning(f"Could not create session in database: {e}")
            session_id = "auto-generated"
        
        self.logger.info(f"Analyzing file: {file_path}")
        
        # Start dashboard if available
        dashboard_url = None
        if self.dashboard:
            try:
                dashboard_url = await self.dashboard.start(session_id)
            except Exception as e:
                self.logger.warning(f"Dashboard start failed: {e}")
        
        # Run orchestrated analysis
        results = []
        try:
            results = await self.orchestrator.analyze(file_path, session_id)
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            results = []
        
        # Generate report if available
        report_info = {"path": "No report generated"}
        if self.reporter:
            try:
                report = await self.reporter.generate_report(session_id)
                if isinstance(report, dict) and 'path' in report:
                    report_info = report
            except Exception as e:
                self.logger.error(f"Report generation failed: {e}")
        
        self.logger.info(f"Analysis complete. Found {len(results)} findings.")
        
        return {
            "session_id": session_id,
            "results": results,
            "report_path": report_info.get("path", "No report generated"),
            "dashboard_url": dashboard_url
        }
    
    async def analyze_directory(self, directory_path: str) -> Dict[str, Any]:
        """Analyze all files in a directory"""
        directory_path = Path(directory_path)
        if not directory_path.exists() or not directory_path.is_dir():
            raise ValueError(f"Directory not found: {directory_path}")
        
        # Create session for directory analysis
        session_id = str(uuid.uuid4())
        try:
            session_id = await self.db.create_session(
                target_path=str(directory_path),
                target_dir=str(directory_path),
                config=getattr(self.config, 'to_dict', lambda: {})()
            )
        except Exception as e:
            self.logger.warning(f"Could not create session: {e}")
            session_id = "auto-generated"
        
        self.logger.info(f"Analyzing directory: {directory_path}")
        
        # Find all files to analyze
        file_patterns = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff', '*.tif', 
                        '*.pdf', '*.wav', '*.mp3', '*.flac', '*.ogg', '*.zip', '*.rar']
        
        files_to_analyze = []
        for pattern in file_patterns:
            files_to_analyze.extend(directory_path.glob(f"**/{pattern}"))
        
        self.logger.info(f"Found {len(files_to_analyze)} files to analyze")
        
        results = []
        for file_path in files_to_analyze:
            try:
                file_results = await self.orchestrator.analyze(file_path, session_id)
                if isinstance(file_results, list):
                    results.extend(file_results)
                elif isinstance(file_results, Exception):
                    self.logger.error(f"Failed to analyze {file_path}: {file_results}")
                else:
                    results.extend(file_results)
            except Exception as e:
                self.logger.error(f"Error analyzing {file_path}: {e}")
        
        # Generate report
        report_info = {"path": "No report generated"}
        if self.reporter:
            try:
                report = await self.reporter.generate_report(session_id)
                if isinstance(report, dict):
                    report_info = report
            except Exception as e:
                self.logger.error(f"Report generation failed: {e}")
        
        return {
            "session_id": session_id,
            "files_processed": len(files_to_analyze),
            "results": results,
            "report_path": report_info.get("path", "No report generated")
        }
    
    async def get_session_results(self, session_id: str) -> Dict[str, Any]:
        """Get results for a completed session"""
        try:
            session = await self.db.get_session(session_id)
            if not session:
                raise ValueError(f"Session {session_id} not found")
            
            results = await self.db.get_session_results(session_id)
            
            return {
                "session_id": session_id,
                "status": session.get("status", "unknown"),
                "results": results,
                "created_at": session.get("created_at"),
                "completed_at": session.get("completed_at")
            }
        except Exception as e:
            self.logger.error(f"Error getting session results: {e}")
            return {"session_id": session_id, "error": str(e)}
    
    async def list_sessions(self) -> List[Dict[str, Any]]:
        """List all analysis sessions"""
        try:
            return await self.db.list_sessions()
        except Exception as e:
            self.logger.error(f"Error listing sessions: {e}")
            return []
    
    async def resume_session(self, session_id: str) -> Dict[str, Any]:
        """Resume an interrupted analysis session"""
        try:
            session = await self.db.get_session(session_id)
            if not session:
                raise ValueError(f"Session {session_id} not found")
            
            # Get incomplete files from session
            incomplete_files = await self.db.get_incomplete_files(session_id)
            
            results = []
            for file_info in incomplete_files:
                file_path = Path(file_info['file_path'])
                if file_path.exists():
                    try:
                        file_results = await self.orchestrator.analyze(file_path, session_id)
                        if isinstance(file_results, list):
                            results.extend(file_results)
                    except Exception as e:
                        self.logger.error(f"Failed to analyze {file_path}: {e}")
            
            # Generate report
            report_info = {"path": "No report generated"}
            if self.reporter:
                try:
                    report = await self.reporter.generate_report(session_id)
                    if isinstance(report, dict):
                        report_info = report
                except Exception as e:
                    self.logger.error(f"Report generation failed: {e}")
            
            return {
                "session_id": session_id,
                "files_processed": len(incomplete_files),
                "results": results,
                "report_path": report_info.get("path", "No report generated")
            }
        except Exception as e:
            self.logger.error(f"Error resuming session: {e}")
            return {"session_id": session_id, "error": str(e)}
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            await self.db.close()
        except:
            pass
        
        if self.dashboard:
            try:
                await self.dashboard.stop()
            except:
                pass

async def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(
        description="StegAnalyzer - Advanced Steganography Detection Framework"
    )
    
    # Target argument (optional for some operations)
    parser.add_argument("target", nargs="?", help="File or directory to analyze")
    
    # Configuration
    parser.add_argument("-c", "--config", default="config/default.json", help="Configuration file")
    parser.add_argument("-o", "--output", help="Output directory for results")
    parser.add_argument("-f", "--format", default="html", choices=["html", "json", "pdf", "markdown"], help="Report format")
    
    # Session management
    parser.add_argument("-r", "--resume", help="Resume session ID")
    parser.add_argument("-l", "--list-sessions", action="store_true", help="List all sessions")
    parser.add_argument("-s", "--session-results", help="Get results for session ID")
    
    # Analysis options
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--cascade", action="store_true", help="Enable recursive cascade analysis")
    parser.add_argument("--max-depth", type=int, default=10, help="Max recursion depth for cascade analysis")
    parser.add_argument("--max-files", type=int, default=5000, help="Max number of files to process during cascade")
    
    # System operations
    parser.add_argument("--check-system", action="store_true", help="Check system requirements")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)
    
    # Handle operations that don't require a target
    if args.check_system:
        checker = SystemChecker()
        status = await checker.check_all()
        print(json.dumps(status, indent=2))
        return 0
    
    if args.list_sessions:
        try:
            analyzer = StegAnalyzer(args.config)
            sessions = await analyzer.list_sessions()
            print(json.dumps(sessions, indent=2, default=str))
            await analyzer.cleanup()
            return 0
        except Exception as e:
            print(f"Error listing sessions: {e}")
            return 1
    
    if args.session_results:
        try:
            analyzer = StegAnalyzer(args.config)
            results = await analyzer.get_session_results(args.session_results)
            print(json.dumps(results, indent=2, default=str))
            await analyzer.cleanup()
            return 0
        except Exception as e:
            print(f"Error getting session results: {e}")
            return 1
    
    if args.resume:
        try:
            analyzer = StegAnalyzer(args.config)
            results = await analyzer.resume_session(args.resume)
            print(f"\nAnalysis resumed!")
            print(f"Session ID: {results['session_id']}")
            print(f"Results found: {len(results.get('results', []))}")
            if 'report_path' in results:
                print(f"Report: {results['report_path']}")
            await analyzer.cleanup()
            return 0
        except Exception as e:
            print(f"Error resuming session: {e}")
            return 1
    
    # For analysis operations, target is required
    if not args.target:
        print("Error: target file or directory is required for analysis")
        print("Use --help for usage information")
        return 1
    
    try:
        analyzer = StegAnalyzer(args.config)
        
        # Override config with command line arguments
        if args.output:
            analyzer.config.report.output_dir = args.output
        if args.format:
            analyzer.config.report.format = args.format
        
        target_path = Path(args.target)
        
        if target_path.is_file():
            results = await analyzer.analyze_file(str(target_path))
            
            # If cascade mode, perform deep analysis on extracted files
            if args.cascade and CascadingAnalyzer:
                try:
                    print("\n🌳 Starting cascading extraction analysis...")
                    cascading = CascadingAnalyzer(
                        analyzer.orchestrator,
                        max_depth=args.max_depth,
                        max_files=args.max_files
                    )
                    tree = await cascading.analyze_cascading(target_path, results.get('session_id'))
                    print(f"\n🎉 Cascading analysis complete! Check output for detailed results.")
                except Exception as e:
                    print(f"Error during cascade analysis: {e}")
                    
        elif target_path.is_dir():
            results = await analyzer.analyze_directory(str(target_path))
        else:
            print(f"Error: {args.target} is not a valid file or directory")
            return 1
        
        # Print results summary
        print(f"\nAnalysis completed!")
        print(f"Session ID: {results['session_id']}")
        print(f"Results found: {len(results.get('results', []))}")
        if results.get('report_path') != "No report generated":
            print(f"Report: {results['report_path']}")
        if results.get('dashboard_url'):
            print(f"Dashboard: {results['dashboard_url']}")
        
        await analyzer.cleanup()
        return 0
        
    except Exception as e:
        logging.getLogger("steg_main").error(f"Application error: {e}")
        return 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)