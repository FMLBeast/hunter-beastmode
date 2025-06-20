"""
LLM Analyzer - Large Language Model Integration for Intelligent Analysis
Supports multiple providers: Anthropic, OpenAI, Hugging Face, Local models
"""

import asyncio
import logging
import json
import re
import base64
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import hashlib
import time
from dataclasses import dataclass
import aiohttp
import tempfile

# Provider-specific imports
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import llamacpp
    LLAMACPP_AVAILABLE = True
except ImportError:
    LLAMACPP_AVAILABLE = False

@dataclass
class AnalysisPrompt:
    system: str
    user: str
    max_tokens: int = 4096
    temperature: float = 0.1

class LLMAnalyzer:
    def __init__(self, config):
        self.config = config.llm
        self.logger = logging.getLogger(__name__)
        
        # Initialize provider-specific clients
        self.anthropic_client = None
        self.openai_client = None
        self.local_model = None
        self.local_tokenizer = None
        
        # Request tracking
        self.request_count = 0
        self.total_tokens = 0
        self.session_cache = {}
        
        # Initialize the configured provider
        asyncio.create_task(self._initialize_provider())
    
    async def _initialize_provider(self):
        """Initialize the configured LLM provider"""
        try:
            if self.config.provider == "anthropic" and ANTHROPIC_AVAILABLE:
                self.anthropic_client = anthropic.AsyncAnthropic(
                    api_key=self.config.api_key,
                    base_url=self.config.base_url if self.config.base_url else None
                )
                self.logger.info("Initialized Anthropic client")
                
            elif self.config.provider == "openai" and OPENAI_AVAILABLE:
                self.openai_client = openai.AsyncOpenAI(
                    api_key=self.config.api_key,
                    base_url=self.config.base_url if self.config.base_url else None
                )
                self.logger.info("Initialized OpenAI client")
                
            elif self.config.provider == "huggingface" and TRANSFORMERS_AVAILABLE:
                await self._load_huggingface_model()
                
            elif self.config.provider == "local" and self.config.local_model_path:
                await self._load_local_model()
                
            else:
                self.logger.warning(f"Provider {self.config.provider} not available or not configured")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM provider: {e}")
    
    async def _load_huggingface_model(self):
        """Load Hugging Face model"""
        try:
            model_name = self.config.model_name or "microsoft/DialoGPT-medium"
            
            self.local_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.local_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            if self.local_tokenizer.pad_token is None:
                self.local_tokenizer.pad_token = self.local_tokenizer.eos_token
            
            self.logger.info(f"Loaded Hugging Face model: {model_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to load Hugging Face model: {e}")
    
    async def _load_local_model(self):
        """Load local model (e.g., GGUF format)"""
        try:
            if LLAMACPP_AVAILABLE:
                self.local_model = llamacpp.Llama(
                    model_path=self.config.local_model_path,
                    n_gpu_layers=self.config.gpu_layers,
                    n_ctx=self.config.context_window,
                    verbose=False
                )
                self.logger.info(f"Loaded local model: {self.config.local_model_path}")
            else:
                self.logger.error("llama-cpp-python not available for local model")
                
        except Exception as e:
            self.logger.error(f"Failed to load local model: {e}")
    
    async def analyze_findings_context(self, findings: List[Dict[str, Any]], file_path: Path) -> Dict[str, Any]:
        """Analyze findings in context and provide insights"""
        
        prompt = self._create_findings_analysis_prompt(findings, file_path)
        
        try:
            response = await self._query_llm(prompt)
            
            # Parse structured response
            analysis = self._parse_analysis_response(response)
            
            # Add metadata
            analysis["llm_provider"] = self.config.provider
            analysis["model_name"] = self.config.model_name
            analysis["timestamp"] = time.time()
            analysis["findings_count"] = len(findings)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"LLM analysis failed: {e}")
            return {
                "summary": "LLM analysis failed",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _create_findings_analysis_prompt(self, findings: List[Dict[str, Any]], file_path: Path) -> AnalysisPrompt:
        """Create prompt for analyzing findings"""
        
        findings_summary = self._summarize_findings(findings)
        file_info = self._get_file_context(file_path)
        
        system_prompt = """You are an expert steganography analyst. Analyze the provided findings and file information to:

1. Assess the likelihood of steganographic content
2. Identify the most significant findings
3. Suggest additional analysis approaches
4. Provide a risk assessment
5. Generate actionable recommendations

Respond in JSON format with these fields:
- summary: Brief analysis summary
- confidence: Overall confidence score (0.0-1.0)
- risk_level: "low", "medium", "high", or "critical"
- key_findings: List of most important findings
- steganography_indicators: Specific indicators found
- recommended_actions: List of recommended next steps
- technical_analysis: Detailed technical insights
- false_positive_likelihood: Assessment of false positive risk"""

        user_prompt = f"""File Information:
- Path: {file_path}
- Type: {file_info.get('type', 'unknown')}
- Size: {file_info.get('size', 0)} bytes
- Extension: {file_path.suffix}

Analysis Results Summary:
- Total findings: {len(findings)}
- Methods used: {', '.join(set(f.get('method', 'unknown') for f in findings))}
- Tools used: {', '.join(set(f.get('tool_name', 'unknown') for f in findings))}

Detailed Findings:
{findings_summary}

Please analyze these findings and provide your assessment."""

        return AnalysisPrompt(
            system=system_prompt,
            user=user_prompt,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature
        )
    
    def _summarize_findings(self, findings: List[Dict[str, Any]]) -> str:
        """Create a concise summary of findings for LLM analysis"""
        if not findings:
            return "No findings to analyze."
        
        summary_parts = []
        
        # Group findings by type
        findings_by_type = {}
        for finding in findings:
            finding_type = finding.get('type', 'unknown')
            if finding_type not in findings_by_type:
                findings_by_type[finding_type] = []
            findings_by_type[finding_type].append(finding)
        
        # Summarize each type
        for finding_type, type_findings in findings_by_type.items():
            count = len(type_findings)
            
            # Get confidence range
            confidences = [f.get('confidence', 0.0) for f in type_findings]
            min_conf = min(confidences) if confidences else 0.0
            max_conf = max(confidences) if confidences else 0.0
            
            # Get methods
            methods = set(f.get('method', 'unknown') for f in type_findings)
            
            # Sample details from highest confidence finding
            best_finding = max(type_findings, key=lambda x: x.get('confidence', 0.0))
            details = best_finding.get('details', 'No details')
            
            summary_parts.append(f"""
Type: {finding_type}
Count: {count}
Confidence Range: {min_conf:.2f} - {max_conf:.2f}
Methods: {', '.join(methods)}
Best Finding: {details[:200]}""")
        
        return '\n'.join(summary_parts)
    
    def _get_file_context(self, file_path: Path) -> Dict[str, Any]:
        """Get contextual information about the file"""
        context = {}
        
        try:
            stat = file_path.stat()
            context['size'] = stat.st_size
            context['modified'] = stat.st_mtime
            
            # Try to determine file type
            import magic
            context['type'] = magic.from_file(str(file_path), mime=True)
            
        except Exception as e:
            self.logger.debug(f"Could not get file context: {e}")
        
        return context
    
    async def _query_llm(self, prompt: AnalysisPrompt) -> str:
        """Query the configured LLM with the prompt"""
        self.request_count += 1
        
        if self.config.provider == "anthropic" and self.anthropic_client:
            return await self._query_anthropic(prompt)
        elif self.config.provider == "openai" and self.openai_client:
            return await self._query_openai(prompt)
        elif self.config.provider == "huggingface" and self.local_model:
            return await self._query_huggingface(prompt)
        elif self.config.provider == "local" and self.local_model:
            return await self._query_local(prompt)
        else:
            raise RuntimeError(f"No available LLM provider: {self.config.provider}")
    
    async def _query_anthropic(self, prompt: AnalysisPrompt) -> str:
        """Query Anthropic Claude"""
        try:
            message = await self.anthropic_client.messages.create(
                model=self.config.model_name,
                max_tokens=prompt.max_tokens,
                temperature=prompt.temperature,
                system=prompt.system,
                messages=[
                    {"role": "user", "content": prompt.user}
                ]
            )
            
            self.total_tokens += message.usage.input_tokens + message.usage.output_tokens
            return message.content[0].text
            
        except Exception as e:
            self.logger.error(f"Anthropic API error: {e}")
            raise
    
    async def _query_openai(self, prompt: AnalysisPrompt) -> str:
        """Query OpenAI GPT"""
        try:
            response = await self.openai_client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": prompt.system},
                    {"role": "user", "content": prompt.user}
                ],
                max_tokens=prompt.max_tokens,
                temperature=prompt.temperature
            )
            
            self.total_tokens += response.usage.total_tokens
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"OpenAI API error: {e}")
            raise
    
    async def _query_huggingface(self, prompt: AnalysisPrompt) -> str:
        """Query Hugging Face model"""
        try:
            # Combine system and user prompts
            full_prompt = f"System: {prompt.system}\n\nUser: {prompt.user}\n\nAssistant:"
            
            # Tokenize
            inputs = self.local_tokenizer.encode(full_prompt, return_tensors="pt", truncation=True, max_length=self.config.context_window-prompt.max_tokens)
            
            # Generate
            with torch.no_grad():
                outputs = self.local_model.generate(
                    inputs,
                    max_new_tokens=prompt.max_tokens,
                    temperature=prompt.temperature,
                    do_sample=True,
                    pad_token_id=self.local_tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.local_tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
            
            self.total_tokens += len(inputs[0]) + len(outputs[0]) - len(inputs[0])
            return response.strip()
            
        except Exception as e:
            self.logger.error(f"Hugging Face model error: {e}")
            raise
    
    async def _query_local(self, prompt: AnalysisPrompt) -> str:
        """Query local GGUF model"""
        try:
            full_prompt = f"System: {prompt.system}\n\nUser: {prompt.user}\n\nAssistant:"
            
            response = self.local_model(
                full_prompt,
                max_tokens=prompt.max_tokens,
                temperature=prompt.temperature,
                stop=["User:", "System:"],
                echo=False
            )
            
            self.total_tokens += response["usage"]["total_tokens"]
            return response["choices"][0]["text"].strip()
            
        except Exception as e:
            self.logger.error(f"Local model error: {e}")
            raise
    
    def _parse_analysis_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured format"""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())
                
                # Validate required fields
                required_fields = ['summary', 'confidence', 'risk_level']
                for field in required_fields:
                    if field not in analysis:
                        analysis[field] = self._get_default_value(field)
                
                # Ensure confidence is in valid range
                if not 0.0 <= analysis.get('confidence', 0.0) <= 1.0:
                    analysis['confidence'] = 0.5
                
                # Ensure risk_level is valid
                if analysis.get('risk_level') not in ['low', 'medium', 'high', 'critical']:
                    analysis['risk_level'] = 'medium'
                
                return analysis
            else:
                # Fallback: parse unstructured response
                return self._parse_unstructured_response(response)
                
        except json.JSONDecodeError:
            return self._parse_unstructured_response(response)
        except Exception as e:
            self.logger.error(f"Failed to parse LLM response: {e}")
            return {
                "summary": "Failed to parse LLM response",
                "confidence": 0.0,
                "risk_level": "medium",
                "error": str(e),
                "raw_response": response[:1000]
            }
    
    def _parse_unstructured_response(self, response: str) -> Dict[str, Any]:
        """Parse unstructured LLM response"""
        # Extract key information using regex
        analysis = {
            "summary": self._extract_summary(response),
            "confidence": self._extract_confidence(response),
            "risk_level": self._extract_risk_level(response),
            "recommendations": self._extract_recommendations(response),
            "raw_response": response[:2000]
        }
        
        return analysis
    
    def _extract_summary(self, text: str) -> str:
        """Extract summary from unstructured text"""
        # Look for summary patterns
        summary_patterns = [
            r'summary[:\-]\s*(.+?)(?:\n|$)',
            r'analysis[:\-]\s*(.+?)(?:\n|$)',
            r'conclusion[:\-]\s*(.+?)(?:\n|$)'
        ]
        
        for pattern in summary_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Fallback: first sentence
        sentences = text.split('.')
        if sentences:
            return sentences[0].strip()
        
        return "No summary available"
    
    def _extract_confidence(self, text: str) -> float:
        """Extract confidence score from text"""
        confidence_patterns = [
            r'confidence[:\-]\s*(\d+(?:\.\d+)?)',
            r'score[:\-]\s*(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)%?\s*confidence'
        ]
        
        for pattern in confidence_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = float(match.group(1))
                # Convert percentage to decimal if needed
                if value > 1.0:
                    value = value / 100.0
                return max(0.0, min(1.0, value))
        
        return 0.5  # Default confidence
    
    def _extract_risk_level(self, text: str) -> str:
        """Extract risk level from text"""
        risk_patterns = [
            r'risk[:\-]\s*(low|medium|high|critical)',
            r'(low|medium|high|critical)\s*risk',
            r'threat\s*level[:\-]\s*(low|medium|high|critical)'
        ]
        
        for pattern in risk_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).lower()
        
        # Infer from keywords
        text_lower = text.lower()
        if any(keyword in text_lower for keyword in ['high', 'serious', 'critical', 'danger']):
            return 'high'
        elif any(keyword in text_lower for keyword in ['medium', 'moderate', 'possible']):
            return 'medium'
        elif any(keyword in text_lower for keyword in ['low', 'minimal', 'unlikely']):
            return 'low'
        
        return 'medium'  # Default
    
    def _extract_recommendations(self, text: str) -> List[str]:
        """Extract recommendations from text"""
        recommendations = []
        
        # Look for numbered lists
        numbered_pattern = r'\d+\.\s*(.+?)(?:\n|$)'
        matches = re.findall(numbered_pattern, text)
        if matches:
            recommendations.extend([match.strip() for match in matches])
        
        # Look for bullet points
        bullet_pattern = r'[•\-\*]\s*(.+?)(?:\n|$)'
        matches = re.findall(bullet_pattern, text)
        if matches:
            recommendations.extend([match.strip() for match in matches])
        
        # Look for recommendation sections
        rec_section_pattern = r'recommend(?:ation)?s?[:\-]\s*(.+?)(?:\n\n|$)'
        match = re.search(rec_section_pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            rec_text = match.group(1).strip()
            recommendations.extend([rec.strip() for rec in rec_text.split('.') if rec.strip()])
        
        return recommendations[:10]  # Limit to top 10
    
    def _get_default_value(self, field: str) -> Any:
        """Get default value for missing fields"""
        defaults = {
            'summary': 'Analysis completed',
            'confidence': 0.5,
            'risk_level': 'medium',
            'key_findings': [],
            'steganography_indicators': [],
            'recommended_actions': [],
            'technical_analysis': '',
            'false_positive_likelihood': 0.5
        }
        return defaults.get(field, None)
    
    async def analyze_file_content(self, file_path: Path, content_sample: str = None) -> Dict[str, Any]:
        """Analyze file content directly with LLM"""
        if not content_sample:
            # Read a sample of the file
            try:
                with open(file_path, 'rb') as f:
                    raw_content = f.read(4096)  # First 4KB
                
                # Try to decode as text
                try:
                    content_sample = raw_content.decode('utf-8', errors='ignore')
                except:
                    # If binary, show hex representation
                    content_sample = raw_content.hex()[:2000]
                    
            except Exception as e:
                self.logger.error(f"Could not read file content: {e}")
                return {"error": str(e)}
        
        prompt = AnalysisPrompt(
            system="""You are analyzing file content for potential steganographic elements. Look for:
1. Hidden messages or encoded data
2. Unusual patterns or structures
3. Potential steganography techniques
4. Suspicious metadata or headers
5. Base64 or other encoding schemes

Provide analysis in JSON format with fields: analysis, indicators, confidence, recommendations.""",
            
            user=f"""File: {file_path}
Content sample:
{content_sample}

Analyze this content for steganographic indicators.""",
            
            max_tokens=2048
        )
        
        try:
            response = await self._query_llm(prompt)
            return self._parse_analysis_response(response)
        except Exception as e:
            return {"error": str(e)}
    
    async def explain_finding(self, finding: Dict[str, Any]) -> str:
        """Get LLM explanation of a specific finding"""
        prompt = AnalysisPrompt(
            system="You are an expert steganography analyst. Explain the provided finding in clear, technical terms that would be useful for both technical and non-technical audiences.",
            
            user=f"""Please explain this steganography analysis finding:

Type: {finding.get('type', 'unknown')}
Method: {finding.get('method', 'unknown')}
Tool: {finding.get('tool_name', 'unknown')}
Confidence: {finding.get('confidence', 0.0)}
Details: {finding.get('details', 'No details')}

Provide a clear explanation of:
1. What this finding means
2. How significant it is
3. What it might indicate
4. Any limitations or caveats""",
            
            max_tokens=1024
        )
        
        try:
            response = await self._query_llm(prompt)
            return response.strip()
        except Exception as e:
            return f"Could not generate explanation: {e}"
    
    async def suggest_next_steps(self, findings: List[Dict[str, Any]], file_path: Path) -> List[str]:
        """Get LLM suggestions for next analysis steps"""
        findings_summary = self._summarize_findings(findings)
        
        prompt = AnalysisPrompt(
            system="You are a steganography analysis expert. Based on the current findings, suggest the next most valuable analysis steps to take.",
            
            user=f"""File: {file_path}

Current findings:
{findings_summary}

What analysis steps would you recommend next? Provide 3-5 specific, actionable recommendations.""",
            
            max_tokens=1024
        )
        
        try:
            response = await self._query_llm(prompt)
            # Extract recommendations
            recommendations = self._extract_recommendations(response)
            return recommendations if recommendations else [response.strip()]
        except Exception as e:
            return [f"Could not generate recommendations: {e}"]
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get LLM usage statistics"""
        return {
            "provider": self.config.provider,
            "model": self.config.model_name,
            "requests": self.request_count,
            "total_tokens": self.total_tokens,
            "cache_hits": len(self.session_cache)
        }
    
    async def cleanup(self):
        """Cleanup LLM resources"""
        # Close API clients
        if hasattr(self.anthropic_client, 'close'):
            await self.anthropic_client.close()
        
        if hasattr(self.openai_client, 'close'):
            await self.openai_client.close()
        
        # Clear local models from memory
        if self.local_model:
            del self.local_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        if self.local_tokenizer:
            del self.local_tokenizer
        
        self.session_cache.clear()
        self.logger.info("LLM analyzer cleaned up")