#!/usr/bin/env python3
"""
CLI integration for Cascade Analyzer
Add these commands and functions to your existing steg_main.py
"""

import argparse
import asyncio
from pathlib import Path
import json
import sys
from typing import Optional

# Add these imports to your existing steg_main.py
from core.orchestrator import StegOrchestrator
from config.steg_config import Config, apply_cascade_preset

class StegAnalyzerCLI:
    """Enhanced CLI with cascade analysis support"""
    
    def __init__(self):
        self.config = Config()
        self.orchestrator = None
    
    def setup_cascade_parser(self, parser):
        """Add cascade-specific arguments to the main parser"""
        
        # Cascade analysis command
        cascade_parser = parser.add_parser('cascade', 
                                         help='Run recursive cascade analysis')
        cascade_parser.add_argument('file_path', 
                                  help='Path to file for cascade analysis')
        cascade_parser.add_argument('--max-depth', type=int, default=10,
                                  help='Maximum recursion depth (default: 10)')
        cascade_parser.add_argument('--preset', 
                                  choices=['fast', 'balanced', 'thorough', 'extreme'],
                                  help='Use configuration preset')
        cascade_parser.add_argument('--enable-exotic', action='store_true',
                                  help='Enable exotic zsteg parameters')
        cascade_parser.add_argument('--disable-zsteg', action='store_true',
                                  help='Disable zsteg analysis')
        cascade_parser.add_argument('--disable-binwalk', action='store_true',
                                  help='Disable binwalk extraction')
        cascade_parser.add_argument('--output-dir', type=str,
                                  help='Custom output directory for extractions')
        cascade_parser.add_argument('--no-extract', action='store_true',
                                  help='Don\'t save extracted files')
        cascade_parser.add_argument('--format', choices=['json', 'text', 'html'],
                                  default='text', help='Output format')
        cascade_parser.set_defaults(func=self.run_cascade_analysis)
        
        # Add cascade options to regular analyze command
        analyze_parser = None
        for action in parser._subparsers._actions:
            if hasattr(action, 'choices') and 'analyze' in action.choices:
                analyze_parser = action.choices['analyze']
                break
        
        if analyze_parser:
            analyze_parser.add_argument('--cascade', action='store_true',
                                      help='Enable cascade analysis mode')
            analyze_parser.add_argument('--cascade-depth', type=int, default=10,
                                      help='Cascade maximum depth')
    
    async def run_cascade_analysis(self, args):
        """Run cascade analysis command"""
        
        file_path = Path(args.file_path)
        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            return 1
        
        print(f"🚀 Starting cascade analysis on {file_path.name}")
        print("=" * 60)
        
        # Apply preset if specified
        if args.preset:
            apply_cascade_preset(self.config, args.preset)
            print(f"📋 Applied {args.preset} preset")
        
        # Apply command line overrides
        if args.max_depth:
            self.config.cascade.max_depth = args.max_depth
        
        if args.enable_exotic:
            self.config.cascade.enable_exotic_params = True
        
        if args.disable_zsteg:
            self.config.cascade.enable_zsteg = False
        
        if args.disable_binwalk:
            self.config.cascade.enable_binwalk = False
        
        if args.output_dir:
            self.config.cascade.extraction_dir = args.output_dir
        
        if args.no_extract:
            self.config.cascade.save_extracts = False
        
        # Initialize orchestrator
        self.orchestrator = StegOrchestrator(self.config)
        
        # Check tool availability
        available_tools = await self.orchestrator.get_available_tools()
        if not available_tools.get('cascade_analyzer'):
            print("❌ Cascade analyzer not available")
            return 1
        
        cascade_info = available_tools.get('cascade_details', {})
        print(f"🔧 Tools available:")
        print(f"   zsteg: {'✅' if cascade_info.get('requirements', {}).get('zsteg') else '❌'}")
        print(f"   binwalk: {'✅' if cascade_info.get('requirements', {}).get('binwalk') else '❌'}")
        print(f"   Max depth: {cascade_info.get('max_depth', 'Unknown')}")
        print(f"   zsteg parameters: {cascade_info.get('zsteg_parameters', 'Unknown')}")
        print()
        
        try:
            # Run cascade analysis
            session_id = f"cascade_{int(time.time())}"
            result = await self.orchestrator.analyze_cascade(file_path, session_id, args.max_depth)
            
            # Output results
            if args.format == 'json':
                print(json.dumps(result, indent=2))
            elif args.format == 'html':
                await self._output_cascade_html(result, file_path)
            else:
                await self._output_cascade_text(result)
            
            return 0
            
        except Exception as e:
            print(f"❌ Cascade analysis failed: {e}")
            return 1
    
    async def _output_cascade_text(self, result):
        """Output cascade results in text format"""
        
        print("📊 Cascade Analysis Results")
        print("=" * 60)
        
        summary = result.get('summary', {})
        print(f"📁 Files analyzed: {summary.get('total_files_analyzed', 0)}")
        print(f"🔍 zsteg findings: {summary.get('total_zsteg_findings', 0)}")
        print(f"📦 Files extracted: {summary.get('total_extracted_files', 0)}")
        print(f"📏 Max depth reached: {summary.get('max_depth_reached', 0)}")
        print()
        
        # High confidence results
        high_conf = summary.get('high_confidence_results', [])
        if high_conf:
            print("🎯 High Confidence Findings:")
            for i, finding in enumerate(high_conf[:10], 1):  # Show top 10
                print(f"  {i}. {finding.get('details', 'No details')}")
                print(f"     Confidence: {finding.get('confidence', 0):.2f}")
                print(f"     File: {Path(finding.get('file_path', '')).name}")
                print(f"     Depth: {finding.get('depth', 0)}")
                print()
        
        # Extraction tree
        tree = summary.get('extraction_tree', {})
        if tree:
            print("🌳 Extraction Tree:")
            self._print_extraction_tree(tree, indent=0)
        
        # Show extracted files
        cascade_results = result.get('cascade_results', [])
        extracted_files = []
        for res in cascade_results:
            extracted_files.extend(res.get('extracted_files', []))
        
        if extracted_files:
            print(f"\n📂 Extracted Files ({len(extracted_files)}):")
            for i, file_path in enumerate(extracted_files[:20], 1):  # Show first 20
                print(f"  {i}. {Path(file_path).name}")
            
            if len(extracted_files) > 20:
                print(f"  ... and {len(extracted_files) - 20} more files")
    
    def _print_extraction_tree(self, tree, indent=0, visited=None):
        """Print extraction tree recursively"""
        if visited is None:
            visited = set()
        
        for parent_hash, child_hashes in tree.items():
            if parent_hash in visited:
                continue
            visited.add(parent_hash)
            
            print("  " * indent + f"├─ {parent_hash[:8]}...")
            
            for child_hash in child_hashes:
                if child_hash not in visited:
                    print("  " * (indent + 1) + f"├─ {child_hash[:8]}...")
                    if child_hash in tree:
                        self._print_extraction_tree({child_hash: tree[child_hash]}, 
                                                  indent + 2, visited)
    
    async def _output_cascade_html(self, result, file_path):
        """Output cascade results in HTML format"""
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Cascade Analysis - {file_path.name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .summary {{ background: #f0f0f0; padding: 10px; border-radius: 5px; }}
        .finding {{ margin: 10px 0; padding: 10px; border-left: 3px solid #007acc; }}
        .high-conf {{ border-left-color: #28a745; }}
        .tree {{ font-family: monospace; background: #f8f9fa; padding: 10px; }}
        .files {{ columns: 3; column-gap: 20px; }}
    </style>
</head>
<body>
    <h1>🚀 Cascade Analysis Results</h1>
    <h2>File: {file_path.name}</h2>
    
    <div class="summary">
        <h3>📊 Summary</h3>
        <p><strong>Files analyzed:</strong> {result.get('summary', {}).get('total_files_analyzed', 0)}</p>
        <p><strong>zsteg findings:</strong> {result.get('summary', {}).get('total_zsteg_findings', 0)}</p>
        <p><strong>Files extracted:</strong> {result.get('summary', {}).get('total_extracted_files', 0)}</p>
        <p><strong>Max depth:</strong> {result.get('summary', {}).get('max_depth_reached', 0)}</p>
    </div>
    
    <h3>🎯 High Confidence Findings</h3>
"""
        
        high_conf = result.get('summary', {}).get('high_confidence_results', [])
        for finding in high_conf:
            html_content += f"""
    <div class="finding high-conf">
        <strong>{finding.get('details', 'No details')}</strong><br>
        Confidence: {finding.get('confidence', 0):.2f}<br>
        File: {Path(finding.get('file_path', '')).name}<br>
        Depth: {finding.get('depth', 0)}
    </div>
"""
        
        html_content += """
    <h3>📂 Extracted Files</h3>
    <div class="files">
"""
        
        cascade_results = result.get('cascade_results', [])
        extracted_files = []
        for res in cascade_results:
            extracted_files.extend(res.get('extracted_files', []))
        
        for file_path in extracted_files:
            html_content += f"<p>{Path(file_path).name}</p>\n"
        
        html_content += """
    </div>
</body>
</html>
"""
        
        output_file = Path(f"cascade_report_{int(time.time())}.html")
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        print(f"📄 HTML report saved: {output_file}")

# Add these functions to enhance your existing analyze command
async def enhanced_analyze_with_cascade(args):
    """Enhanced analyze function with cascade support"""
    
    # ... existing analyze code ...
    
    # Add cascade analysis if requested
    if getattr(args, 'cascade', False):
        print("\n🔄 Running cascade analysis...")
        
        config.cascade_mode = True
        if hasattr(args, 'cascade_depth'):
            config.cascade.max_depth = args.cascade_depth
        
        # Re-run with cascade enabled
        orchestrator = StegOrchestrator(config)
        cascade_results = await orchestrator.analyze_cascade(file_path, session_id)
        
        print(f"🎯 Cascade found {len(cascade_results.get('cascade_results', []))} additional results")

# Add these to your main CLI setup
def setup_main_parser():
    """Setup main argument parser with cascade support"""
    
    parser = argparse.ArgumentParser(description="StegAnalyzer - Advanced Steganography Detection")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # ... existing command setup ...
    
    # Add cascade command
    cli = StegAnalyzerCLI()
    cli.setup_cascade_parser(subparsers)
    
    return parser

# Example usage commands to add to your CLI help
CASCADE_EXAMPLES = """
Cascade Analysis Examples:

# Basic cascade analysis
python steg_main.py cascade image.png

# Deep cascade analysis
python steg_main.py cascade image.png --max-depth 20

# Fast cascade with preset
python steg_main.py cascade image.png --preset fast

# Thorough cascade with exotic parameters
python steg_main.py cascade image.png --preset thorough --enable-exotic

# Cascade without file extraction
python steg_main.py cascade image.png --no-extract

# Regular analysis with cascade mode
python steg_main.py analyze image.png --cascade --cascade-depth 15

# HTML output
python steg_main.py cascade image.png --format html
"""

if __name__ == "__main__":
    # Add this to your main function
    import time
    
    parser = setup_main_parser()
    args = parser.parse_args()
    
    if args.command == 'cascade':
        cli = StegAnalyzerCLI()
        exit_code = asyncio.run(cli.run_cascade_analysis(args))
        sys.exit(exit_code)
    elif args.command == 'examples' and getattr(args, 'cascade_examples', False):
        print(CASCADE_EXAMPLES)
    else:
        # ... existing command handling ...
        pass
