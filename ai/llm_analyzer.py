"""
Fix for LLM Analyzer Anthropic client initialization
Add this to your ai/llm_analyzer.py file or replace the initialization section
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import json

try:
    from anthropic import AsyncAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

class LLMAnalyzer:
    def __init__(self, config):
        self.config = config.llm if hasattr(config, 'llm') else config
        self.logger = logging.getLogger(__name__)
        self.client = None
        
        # Initialize Anthropic client if available and configured
        if ANTHROPIC_AVAILABLE and hasattr(self.config, 'anthropic_api_key') and self.config.anthropic_api_key:
            try:
                # Fixed initialization - remove proxies parameter
                self.client = AsyncAnthropic(
                    api_key=self.config.anthropic_api_key,
                    # Remove any proxy-related parameters that might be causing issues
                )
                self.logger.info("LLM analyzer initialized with Anthropic")
            except Exception as e:
                self.logger.error(f"Failed to initialize Anthropic client: {e}")
                self.client = None
        else:
            self.logger.warning("LLM provider anthropic requires API key")

    async def analyze_file(self, file_path: Path, findings: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Main analysis method that the orchestrator expects"""
        if not self.client:
            return []
            
        try:
            # Read file info
            file_info = self._get_file_context(file_path)
            
            # Create analysis prompt
            prompt = self._create_analysis_prompt(file_path, file_info, findings or [])
            
            # Call Anthropic API
            response = await self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1024,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Parse response
            analysis_result = self._parse_llm_response(response.content[0].text, file_path)
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"LLM analysis failed for {file_path}: {e}")
            return [{
                "type": "error",
                "method": "llm_analysis",
                "tool_name": "llm_analyzer",
                "confidence": 0.0,
                "details": f"LLM analysis failed: {str(e)}",
                "file_path": str(file_path)
            }]

    async def analyze(self, file_path: Path) -> List[Dict[str, Any]]:
        """Alternative method name for compatibility"""
        return await self.analyze_file(file_path)

    def _get_file_context(self, file_path: Path) -> Dict[str, Any]:
        """Get basic file information"""
        try:
            stat = file_path.stat()
            return {
                'size': stat.st_size,
                'extension': file_path.suffix,
                'name': file_path.name
            }
        except Exception:
            return {}

    def _create_analysis_prompt(self, file_path: Path, file_info: Dict[str, Any], findings: List[Dict[str, Any]]) -> str:
        """Create analysis prompt for LLM"""
        prompt = f"""Analyze this file for potential steganography:

File: {file_path.name}
Size: {file_info.get('size', 'unknown')} bytes
Extension: {file_info.get('extension', 'unknown')}

Previous findings: {len(findings)} detections found

Based on the file characteristics and any findings, assess the likelihood of steganography and provide analysis.
Focus on file size, extension mismatches, and unusual patterns.

Respond with a confidence score (0.0-1.0) and brief explanation."""

        return prompt

    def _parse_llm_response(self, response_text: str, file_path: Path) -> List[Dict[str, Any]]:
        """Parse LLM response into structured findings"""
        try:
            # Simple parsing - look for confidence indicators
            confidence = 0.3  # Default medium confidence
            
            if "high confidence" in response_text.lower() or "likely" in response_text.lower():
                confidence = 0.8
            elif "low confidence" in response_text.lower() or "unlikely" in response_text.lower():
                confidence = 0.2
            elif "suspicious" in response_text.lower():
                confidence = 0.6
                
            return [{
                "type": "llm_analysis",
                "method": "llm_analysis", 
                "tool_name": "llm_analyzer",
                "confidence": confidence,
                "details": response_text[:200],  # Truncate for storage
                "file_path": str(file_path)
            }]
            
        except Exception as e:
            self.logger.error(f"Failed to parse LLM response: {e}")
            return []

    async def close(self):
        """Cleanup method"""
        if self.client:
            try:
                await self.client.close()
            except Exception as e:
                self.logger.debug(f"Error closing LLM client: {e}")
