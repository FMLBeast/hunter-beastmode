"""
Report Generator - Creates comprehensive analysis reports in multiple formats
"""

import logging
import json
import time
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import hashlib
import base64
import tempfile

class ReportGenerator:
    def __init__(self, config, database):
        self.config = config.report
        self.db = database
        self.logger = logging.getLogger(__name__)
        
        # Ensure output directory exists
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Template directory
        self.template_dir = Path(self.config.template_dir)
        
        # Report configuration
        self.supported_formats = ['html', 'json', 'markdown', 'txt']
        self.default_format = self.config.format if self.config.format in self.supported_formats else 'html'
    
    async def generate_report(self, session_id: str, format: str = None) -> Dict[str, Any]:
        """Generate comprehensive analysis report"""
        start_time = time.time()
        
        try:
            # Use default format if none specified
            report_format = format or self.default_format
            
            # Gather all report data
            report_data = await self._gather_report_data(session_id)
            
            # Generate report based on format
            if report_format == 'html':
                report_path = await self._generate_html_report(session_id, report_data)
            elif report_format == 'json':
                report_path = await self._generate_json_report(session_id, report_data)
            elif report_format == 'markdown':
                report_path = await self._generate_markdown_report(session_id, report_data)
        