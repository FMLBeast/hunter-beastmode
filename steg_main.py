#!/usr/bin/env python3
"""
StegAnalyzer Main Entry Point
Advanced Steganography Detection and Analysis System
"""

import os
import sys
import asyncio
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import json

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Core imports
from core.orchestrator import StegOrchestrator
from core.database import DatabaseManager
from core.dashboard import create_dashboard
from core.reporter import ReportGenerator
from config.steg_config import Config

# Version info
__version__ = "2.0.0"
__author__ = "StegAnalyzer Team"

def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration"""
    log_format = "%(asctime)s %(levelname)s %(name)s - %(message)s"
    
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file or "logs/steganalyzer.log")
        ]
    )
    
    logger = logging.getLogger("steg_main")
    logger.info("Logging initialized.")
    return logger

def load_configuration(config_path: str = None) -> Config:
    """Load configuration from file"""
    if config_path and Path(config_path).exists():
        config = Config.from_file(config_path)
        logging.info(f"Configuration loaded from {config_path}")
    else:
        # Try default locations
        default_configs = [
            "config/default.json",
            "config/vast_ai_optimized.json", 
            "config/high_performance.json"
        ]
        
        config = None
        for default_config in default_configs:
            if Path(default_config).exists():
                config = Config.from_file(default_config)
                logging.info(f"Configuration loaded from {default_config}")
                break
        
        if not config:
            config = Config()  # Use default configuration
            logging.info("Using default configuration")
    
    return config

async def analyze_file(file_path: Path, orchestrator: StegOrchestrator, 
                      cascade: bool = False, session_id: str = None) -> Dict[str, Any]:
    """Analyze a single file"""
    if not file_path.exists():
        return {"error": f"File not found: {file_path}"}
    
    try:
        if cascade:
            logging.info(f"Running cascade analysis on {file_path}")
            result = await orchestrator.analyze_cascade(file_path, session_id)
        else:
            logging.info(f"Running standard analysis on {file_path}")
            results = await orchestrator.analyze(file_path, session_id)
            result = {
                "type": "standard_analysis",
                "results": results,
                "total_results": len(results)
            }
        
        return result
        
    except Exception as e:
        logging.error(f"Analysis failed for {file_path}: {e}")
        return {"error": str(e)}

async def analyze_directory(directory: Path, orchestrator: StegOrchestrator,
                           cascade: bool = False, pattern: str = None) -> List[Dict[str, Any]]:
    """Analyze all files in a directory"""
    results = []
    
    # Define file patterns to analyze
    if pattern:
        patterns = pattern.split(',')
    else:
        patterns = ['*.png', '*.jpg', '*.jpeg', '*.gif', '*.bmp', '*.tiff', 
                   '*.wav', '*.mp3', '*.flac', '*.pdf', '*.zip']
    
    files_to_analyze = []
    for pat in patterns:
        files_to_analyze.extend(directory.glob(pat))
        files_to_analyze.extend(directory.rglob(pat))  # Recursive
    
    # Remove duplicates
    files_to_analyze = list(set(files_to_analyze))
    
    logging.info(f"Found {len(files_to_analyze)} files to analyze")
    
    for i, file_path in enumerate(files_to_analyze):
        logging.info(f"Analyzing file {i+1}/{len(files_to_analyze)}: {file_path.name}")
        
        session_id = f"batch_{int(time.time())}_{i}"
        result = await analyze_file(file_path, orchestrator, cascade, session_id)
        
        result["file_index"] = i + 1
        result["total_files"] = len(files_to_analyze)
        results.append(result)
    
    return results

async def start_dashboard(config: Config):
    """Start the web dashboard"""
    try:
        dashboard = create_dashboard(config)
        
        if dashboard:
            host = config.dashboard.host
            port = config.dashboard.port
            
            logging.info(f"Starting dashboard on http://{host}:{port}")
            await dashboard.start(host, port)
        else:
            logging.warning("Dashboard could not be created")
            
    except Exception as e:
        logging.error(f"Dashboard failed to start: {e}")

def generate_report(results: Dict[str, Any], output_path: Path, config: Config):
    """Generate analysis report"""
    try:
        reporter = ReportGenerator(config)
        
        if config.report.format.lower() == "html":
            report_file = output_path / f"report_{int(time.time())}.html"
            reporter.generate_html_report(results, report_file)
        elif config.report.format.lower() == "json":
            report_file = output_path / f"report_{int(time.time())}.json"
            with open(report_file, 'w') as f:
                json.dump(results, f, indent=2)
        else:
            report_file = output_path / f"report_{int(time.time())}.txt"
            with open(report_file, 'w') as f:
                f.write(str(results))
        
        logging.info(f"Report generated: {report_file}")
        return report_file
        
    except Exception as e:
        logging.error(f"Report generation failed: {e}")
        return None

async def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(
        description="StegAnalyzer - Advanced Steganography Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 steg_main.py image.png                    # Analyze single image
  python3 steg_main.py image.png --cascade          # Deep cascade analysis
  python3 steg_main.py /path/to/images/ --recursive # Analyze directory
  python3 steg_main.py --dashboard                  # Start web dashboard
  python3 steg_main.py --check-system               # Check system tools
        """
    )
    
    # File/directory arguments
    parser.add_argument("target", nargs="?", help="File or directory to analyze")
    
    # Analysis options
    parser.add_argument("--cascade", action="store_true", 
                       help="Enable cascade analysis (deep recursive extraction)")
    parser.add_argument("--recursive", "-r", action="store_true",
                       help="Recursively analyze directory")
    parser.add_argument("--pattern", type=str,
                       help="File patterns to match (e.g., '*.png,*.jpg')")
    
    # Configuration options
    parser.add_argument("--config", type=str,
                       help="Configuration file path")
    parser.add_argument("--output", "-o", type=str, default="reports",
                       help="Output directory for reports")
    parser.add_argument("--format", choices=["html", "json", "txt"], default="html",
                       help="Report format")
    
    # System options
    parser.add_argument("--dashboard", action="store_true",
                       help="Start web dashboard")
    parser.add_argument("--check-system", action="store_true",
                       help="Check system tools and dependencies")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    
    # Session management
    parser.add_argument("--session-id", type=str,
                       help="Custom session ID")
    parser.add_argument("--resume", type=str,
                       help="Resume previous session")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.debug else ("INFO" if args.verbose else "WARNING")
    logger = setup_logging(log_level)
    
    try:
        # Load configuration
        config = load_configuration(args.config)
        
        # Override report format if specified
        if hasattr(config, 'report'):
            config.report.format = args.format
            config.report.output_dir = args.output
        
        # Check system if requested
        if args.check_system:
            from tools.system_check import SystemCheck
            checker = SystemCheck()
            results = checker.run_comprehensive_check()
            print(json.dumps(results, indent=2))
            return
        
        # Start dashboard if requested
        if args.dashboard:
            await start_dashboard(config)
            return
        
        # Validate target
        if not args.target:
            logger.error("No target file or directory specified")
            parser.print_help()
            return
        
        target_path = Path(args.target)
        if not target_path.exists():
            logger.error(f"Target not found: {target_path}")
            return
        
        # Initialize database
        db_manager = DatabaseManager(config)
        
        # Initialize orchestrator
        orchestrator = StegOrchestrator(config, db_manager)
        
        # Create output directory
        output_dir = Path(args.output)
        output_dir.mkdir(exist_ok=True)
        
        # Run analysis
        if target_path.is_file():
            logger.info(f"Analyzing file: {target_path}")
            result = await analyze_file(target_path, orchestrator, args.cascade, args.session_id)
            
        elif target_path.is_dir():
            logger.info(f"Analyzing directory: {target_path}")
            results = await analyze_directory(target_path, orchestrator, args.cascade, args.pattern)
            result = {
                "type": "directory_analysis",
                "directory": str(target_path),
                "total_files": len(results),
                "results": results
            }
        else:
            logger.error(f"Invalid target: {target_path}")
            return
        
        # Generate report
        report_file = generate_report(result, output_dir, config)
        
        # Print summary
        print("\nAnalysis completed!")
        print(f"Session ID: {args.session_id or 'auto-generated'}")
        
        if isinstance(result, dict):
            if "results" in result:
                print(f"Results found: {len(result['results'])}")
            if "error" in result:
                print(f"Error: {result['error']}")
        
        if report_file:
            print(f"Report: {report_file}")
        else:
            print("Report: No report generated")
        
        # Start dashboard if enabled in config
        if hasattr(config, 'dashboard') and config.dashboard.enabled:
            print(f"Dashboard: http://{config.dashboard.host}:{config.dashboard.port}")
        
        # Cleanup
        orchestrator.cleanup()
        if hasattr(db_manager, 'close'):
            await db_manager.close()
        
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)
