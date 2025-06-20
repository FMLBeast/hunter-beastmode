"""
Cloud Integrations - External API Integration for Enhanced Analysis
Supports VirusTotal, NSRL, Hash Lookups, Malware Bazaar, and other cloud services
"""

import asyncio
import aiohttp
import logging
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from urllib.parse import urlencode
import base64

class CloudIntegrations:
    def __init__(self, config):
        self.config = config.cloud
        self.logger = logging.getLogger(__name__)
        
        # Rate limiting
        self.rate_limiter = RateLimiter(self.config.rate_limit)
        
        # API endpoints
        self.endpoints = {
            'virustotal': {
                'base_url': 'https://www.virustotal.com/vtapi/v2',
                'file_report': '/file/report',
                'file_scan': '/file/scan',
                'url_scan': '/url/scan',
                'url_report': '/url/report'
            },
            'nsrl': {
                'base_url': 'https://hashlookup.circl.lu',
                'lookup': '/lookup'
            },
            'hashlookup': {
                'base_url': 'https://hashlookup.circl.lu',
                'lookup': '/lookup'
            },
            'malware_bazaar': {
                'base_url': 'https://mb-api.abuse.ch/api/v1',
                'query_hash': '/query'
            },
            'hybrid_analysis': {
                'base_url': 'https://www.hybrid-analysis.com/api/v2',
                'search_hash': '/search/hash'
            }
        }
        
        # Session for HTTP requests
        self.session = None
        
        # Cache for API responses
        self.cache = {}
        self.cache_ttl = 3600  # 1 hour
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.cleanup()
    
    async def initialize(self):
        """Initialize HTTP session and connections"""
        connector = aiohttp.TCPConnector(
            limit=10,
            limit_per_host=5,
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )
    
    async def cleanup(self):
        """Cleanup HTTP session"""
        if self.session:
            await self.session.close()
    
    async def analyze_file_hash(self, file_path: Path) -> List[Dict[str, Any]]:
        """Analyze file using cloud services"""
        results = []
        
        if not self.config.enabled:
            return results
        
        try:
            # Calculate file hashes
            hashes = await self._calculate_file_hashes(file_path)
            
            # Query each enabled service
            if self.config.virustotal_api_key:
                vt_results = await self._query_virustotal(hashes, file_path)
                if vt_results:
                    results.extend(vt_results)
            
            if self.config.nsrl_enabled:
                nsrl_results = await self._query_nsrl(hashes)
                if nsrl_results:
                    results.extend(nsrl_results)
            
            if self.config.hashlookup_enabled:
                hashlookup_results = await self._query_hashlookup(hashes)
                if hashlookup_results:
                    results.extend(hashlookup_results)
            
            if self.config.malware_bazaar_enabled:
                mb_results = await self._query_malware_bazaar(hashes)
                if mb_results:
                    results.extend(mb_results)
            
            if self.config.hybrid_analysis_api_key:
                ha_results = await self._query_hybrid_analysis(hashes)
                if ha_results:
                    results.extend(ha_results)
                    
        except Exception as e:
            self.logger.error(f"Cloud analysis failed for {file_path}: {e}")
        
        return results
    
    async def _calculate_file_hashes(self, file_path: Path) -> Dict[str, str]:
        """Calculate multiple hash types for the file"""
        hashes = {}
        
        try:
            # Initialize hash objects
            md5_hash = hashlib.md5()
            sha1_hash = hashlib.sha1()
            sha256_hash = hashlib.sha256()
            
            # Read file in chunks
            with open(file_path, 'rb') as f:
                while chunk := f.read(8192):
                    md5_hash.update(chunk)
                    sha1_hash.update(chunk)
                    sha256_hash.update(chunk)
            
            hashes = {
                'md5': md5_hash.hexdigest(),
                'sha1': sha1_hash.hexdigest(),
                'sha256': sha256_hash.hexdigest()
            }
            
        except Exception as e:
            self.logger.error(f"Hash calculation failed for {file_path}: {e}")
        
        return hashes
    
    async def _query_virustotal(self, hashes: Dict[str, str], file_path: Path) -> List[Dict[str, Any]]:
        """Query VirusTotal API"""
        results = []
        
        if not self.config.virustotal_api_key:
            return results
        
        try:
            await self.rate_limiter.wait()
            
            # Check cache first
            cache_key = f"vt_{hashes['sha256']}"
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                return cached_result
            
            # Query VirusTotal
            params = {
                'apikey': self.config.virustotal_api_key,
                'resource': hashes['sha256']
            }
            
            url = f"{self.endpoints['virustotal']['base_url']}{self.endpoints['virustotal']['file_report']}"
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get('response_code') == 1:
                        # File found in VirusTotal
                        vt_result = self._process_virustotal_response(data, file_path)
                        results.append(vt_result)
                        
                        # Cache result
                        self._cache_result(cache_key, results)
                    
                    elif data.get('response_code') == 0:
                        # File not found - might want to submit for analysis
                        results.append({
                            "type": "virustotal_unknown",
                            "method": "virustotal_lookup",
                            "tool_name": "virustotal",
                            "confidence": 0.3,
                            "details": "File not found in VirusTotal database",
                            "file_path": str(file_path),
                            "recommendation": "Consider submitting file for analysis"
                        })
                
                elif response.status == 204:
                    # Rate limit exceeded
                    self.logger.warning("VirusTotal rate limit exceeded")
                
                else:
                    self.logger.error(f"VirusTotal API error: {response.status}")
                    
        except Exception as e:
            self.logger.error(f"VirusTotal query failed: {e}")
        
        return results
    
    def _process_virustotal_response(self, data: Dict, file_path: Path) -> Dict[str, Any]:
        """Process VirusTotal API response"""
        scans = data.get('scans', {})
        total_scans = data.get('total', 0)
        positives = data.get('positives', 0)
        
        # Calculate detection ratio
        detection_ratio = positives / total_scans if total_scans > 0 else 0
        
        # Get detected malware names
        detected_threats = []
        for engine, result in scans.items():
            if result.get('detected'):
                threat_name = result.get('result', 'Unknown')
                detected_threats.append({
                    'engine': engine,
                    'threat': threat_name
                })
        
        # Determine confidence and risk level
        if positives == 0:
            confidence = 0.9
            risk_level = "clean"
            details = f"File is clean according to {total_scans} antivirus engines"
        elif positives <= 2:
            confidence = 0.6
            risk_level = "low"
            details = f"File flagged by {positives}/{total_scans} engines (possible false positive)"
        elif positives <= 5:
            confidence = 0.8
            risk_level = "medium"
            details = f"File flagged by {positives}/{total_scans} engines"
        else:
            confidence = 0.95
            risk_level = "high"
            details = f"File flagged by {positives}/{total_scans} engines - likely malicious"
        
        return {
            "type": "virustotal_analysis",
            "method": "virustotal_lookup",
            "tool_name": "virustotal",
            "confidence": confidence,
            "details": details,
            "scan_date": data.get('scan_date'),
            "total_scans": total_scans,
            "positive_detections": positives,
            "detection_ratio": detection_ratio,
            "risk_level": risk_level,
            "detected_threats": detected_threats[:10],  # Limit to top 10
            "permalink": data.get('permalink'),
            "file_path": str(file_path)
        }
    
    async def _query_nsrl(self, hashes: Dict[str, str]) -> List[Dict[str, Any]]:
        """Query NSRL (National Software Reference Library) database"""
        results = []
        
        try:
            await self.rate_limiter.wait()
            
            # Try SHA-1 first (NSRL primary hash)
            for hash_type in ['sha1', 'md5', 'sha256']:
                if hash_type not in hashes:
                    continue
                
                cache_key = f"nsrl_{hashes[hash_type]}"
                cached_result = self._get_cached_result(cache_key)
                if cached_result:
                    return cached_result
                
                url = f"{self.endpoints['nsrl']['base_url']}/lookup/sha1/{hashes[hash_type]}"
                
                async with self.session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        nsrl_result = {
                            "type": "nsrl_known_file",
                            "method": "nsrl_lookup",
                            "tool_name": "nsrl",
                            "confidence": 0.9,
                            "details": f"File found in NSRL database: {data.get('ProductName', 'Unknown')}",
                            "product_name": data.get('ProductName'),
                            "file_name": data.get('FileName'),
                            "file_size": data.get('FileSize'),
                            "manufacturer": data.get('MfgCode'),
                            "hash_type": hash_type,
                            "hash_value": hashes[hash_type]
                        }
                        
                        results.append(nsrl_result)
                        self._cache_result(cache_key, results)
                        break  # Found in NSRL, no need to check other hashes
                    
                    elif response.status == 404:
                        # Not found in NSRL - this is normal for many files
                        continue
                    
                    else:
                        self.logger.debug(f"NSRL API error: {response.status}")
                        
        except Exception as e:
            self.logger.error(f"NSRL query failed: {e}")
        
        return results
    
    async def _query_hashlookup(self, hashes: Dict[str, str]) -> List[Dict[str, Any]]:
        """Query CIRCL Hash Lookup service"""
        results = []
        
        try:
            await self.rate_limiter.wait()
            
            # Try different hash types
            for hash_type in ['sha1', 'md5', 'sha256']:
                if hash_type not in hashes:
                    continue
                
                cache_key = f"hashlookup_{hashes[hash_type]}"
                cached_result = self._get_cached_result(cache_key)
                if cached_result:
                    results.extend(cached_result)
                    continue
                
                url = f"{self.endpoints['hashlookup']['base_url']}/lookup/{hash_type}/{hashes[hash_type]}"
                
                async with self.session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Process hashlookup response
                        hashlookup_result = {
                            "type": "hashlookup_known_file",
                            "method": "hashlookup_lookup",
                            "tool_name": "hashlookup",
                            "confidence": 0.8,
                            "details": f"File found in CIRCL hashlookup database",
                            "hash_type": hash_type,
                            "hash_value": hashes[hash_type],
                            "file_info": data
                        }
                        
                        # Add specific information if available
                        if 'FileName' in data:
                            hashlookup_result['file_name'] = data['FileName']
                        if 'FileSize' in data:
                            hashlookup_result['file_size'] = data['FileSize']
                        if 'ProductName' in data:
                            hashlookup_result['product_name'] = data['ProductName']
                        
                        result_list = [hashlookup_result]
                        results.extend(result_list)
                        self._cache_result(cache_key, result_list)
                        break  # Found, no need to check other hashes
                    
                    elif response.status == 404:
                        # Not found
                        continue
                    
                    else:
                        self.logger.debug(f"Hashlookup API error: {response.status}")
                        
        except Exception as e:
            self.logger.error(f"Hashlookup query failed: {e}")
        
        return results
    
    async def _query_malware_bazaar(self, hashes: Dict[str, str]) -> List[Dict[str, Any]]:
        """Query Malware Bazaar database"""
        results = []
        
        try:
            await self.rate_limiter.wait()
            
            # Try SHA-256 first (Malware Bazaar prefers SHA-256)
            for hash_type in ['sha256', 'sha1', 'md5']:
                if hash_type not in hashes:
                    continue
                
                cache_key = f"mb_{hashes[hash_type]}"
                cached_result = self._get_cached_result(cache_key)
                if cached_result:
                    results.extend(cached_result)
                    continue
                
                # Malware Bazaar API request
                url = self.endpoints['malware_bazaar']['base_url'] + self.endpoints['malware_bazaar']['query_hash']
                
                data = {
                    'query': 'get_info',
                    'hash': hashes[hash_type]
                }
                
                async with self.session.post(url, data=data) as response:
                    if response.status == 200:
                        result_data = await response.json()
                        
                        if result_data.get('query_status') == 'ok':
                            malware_info = result_data.get('data', [])
                            
                            if malware_info:
                                mb_result = self._process_malware_bazaar_response(malware_info[0], hash_type, hashes[hash_type])
                                result_list = [mb_result]
                                results.extend(result_list)
                                self._cache_result(cache_key, result_list)
                                break
                        
                        elif result_data.get('query_status') == 'hash_not_found':
                            continue  # Try next hash type
                    
                    else:
                        self.logger.debug(f"Malware Bazaar API error: {response.status}")
                        
        except Exception as e:
            self.logger.error(f"Malware Bazaar query failed: {e}")
        
        return results
    
    def _process_malware_bazaar_response(self, data: Dict, hash_type: str, hash_value: str) -> Dict[str, Any]:
        """Process Malware Bazaar API response"""
        return {
            "type": "malware_bazaar_detection",
            "method": "malware_bazaar_lookup",
            "tool_name": "malware_bazaar",
            "confidence": 0.95,
            "details": f"File found in Malware Bazaar database - {data.get('signature', 'Unknown malware')}",
            "malware_family": data.get('signature'),
            "file_name": data.get('file_name'),
            "file_size": data.get('file_size'),
            "file_type": data.get('file_type'),
            "first_seen": data.get('first_seen'),
            "last_seen": data.get('last_seen'),
            "delivery_method": data.get('delivery_method'),
            "intelligence": data.get('intelligence', {}),
            "hash_type": hash_type,
            "hash_value": hash_value,
            "risk_level": "high"
        }
    
    async def _query_hybrid_analysis(self, hashes: Dict[str, str]) -> List[Dict[str, Any]]:
        """Query Hybrid Analysis API"""
        results = []
        
        if not self.config.hybrid_analysis_api_key:
            return results
        
        try:
            await self.rate_limiter.wait()
            
            headers = {
                'api-key': self.config.hybrid_analysis_api_key,
                'User-Agent': 'StegAnalyzer'
            }
            
            # Search by SHA-256 hash
            hash_value = hashes.get('sha256')
            if not hash_value:
                return results
            
            cache_key = f"ha_{hash_value}"
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                return cached_result
            
            url = f"{self.endpoints['hybrid_analysis']['base_url']}{self.endpoints['hybrid_analysis']['search_hash']}"
            params = {'hash': hash_value}
            
            async with self.session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get('count', 0) > 0:
                        # Process results
                        for result in data.get('result', []):
                            ha_result = self._process_hybrid_analysis_response(result, hash_value)
                            results.append(ha_result)
                        
                        self._cache_result(cache_key, results)
                
                elif response.status == 403:
                    self.logger.warning("Hybrid Analysis API access denied")
                
                else:
                    self.logger.debug(f"Hybrid Analysis API error: {response.status}")
                    
        except Exception as e:
            self.logger.error(f"Hybrid Analysis query failed: {e}")
        
        return results
    
    def _process_hybrid_analysis_response(self, data: Dict, hash_value: str) -> Dict[str, Any]:
        """Process Hybrid Analysis API response"""
        verdict = data.get('verdict', 'unknown')
        threat_score = data.get('threat_score', 0)
        
        # Determine confidence based on threat score
        if threat_score >= 80:
            confidence = 0.95
            risk_level = "high"
        elif threat_score >= 50:
            confidence = 0.8
            risk_level = "medium"
        elif threat_score >= 20:
            confidence = 0.6
            risk_level = "low"
        else:
            confidence = 0.4
            risk_level = "clean"
        
        return {
            "type": "hybrid_analysis_detection",
            "method": "hybrid_analysis_lookup",
            "tool_name": "hybrid_analysis",
            "confidence": confidence,
            "details": f"Hybrid Analysis verdict: {verdict} (score: {threat_score}/100)",
            "verdict": verdict,
            "threat_score": threat_score,
            "risk_level": risk_level,
            "analysis_date": data.get('analysis_start_time'),
            "environment": data.get('environment_description'),
            "file_type": data.get('type'),
            "hash_value": hash_value
        }
    
    def _get_cached_result(self, cache_key: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached API result if still valid"""
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return cached_data
            else:
                del self.cache[cache_key]
        return None
    
    def _cache_result(self, cache_key: str, result: List[Dict[str, Any]]):
        """Cache API result"""
        self.cache[cache_key] = (result, time.time())
    
    async def submit_file_for_analysis(self, file_path: Path) -> Dict[str, Any]:
        """Submit file to cloud services for analysis"""
        results = {}
        
        if not self.config.enabled:
            return results
        
        try:
            # Submit to VirusTotal if API key available
            if self.config.virustotal_api_key:
                vt_submission = await self._submit_to_virustotal(file_path)
                if vt_submission:
                    results['virustotal'] = vt_submission
                    
        except Exception as e:
            self.logger.error(f"File submission failed: {e}")
        
        return results
    
    async def _submit_to_virustotal(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Submit file to VirusTotal for analysis"""
        try:
            await self.rate_limiter.wait()
            
            url = f"{self.endpoints['virustotal']['base_url']}{self.endpoints['virustotal']['file_scan']}"
            
            data = aiohttp.FormData()
            data.add_field('apikey', self.config.virustotal_api_key)
            
            # Add file
            with open(file_path, 'rb') as f:
                data.add_field('file', f, filename=file_path.name)
            
            async with self.session.post(url, data=data) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    return {
                        'scan_id': result.get('scan_id'),
                        'permalink': result.get('permalink'),
                        'response_code': result.get('response_code'),
                        'verbose_msg': result.get('verbose_msg')
                    }
                
                else:
                    self.logger.error(f"VirusTotal submission failed: {response.status}")
                    
        except Exception as e:
            self.logger.error(f"VirusTotal submission error: {e}")
        
        return None
    
    async def get_ip_reputation(self, ip_address: str) -> List[Dict[str, Any]]:
        """Get IP address reputation from multiple sources"""
        results = []
        
        # This could integrate with various IP reputation services
        # Implementation would depend on available APIs
        
        return results
    
    async def get_url_reputation(self, url: str) -> List[Dict[str, Any]]:
        """Get URL reputation from multiple sources"""
        results = []
        
        if not self.config.enabled:
            return results
        
        try:
            # VirusTotal URL analysis
            if self.config.virustotal_api_key:
                vt_url_result = await self._analyze_url_virustotal(url)
                if vt_url_result:
                    results.append(vt_url_result)
                    
        except Exception as e:
            self.logger.error(f"URL reputation check failed: {e}")
        
        return results
    
    async def _analyze_url_virustotal(self, url: str) -> Optional[Dict[str, Any]]:
        """Analyze URL with VirusTotal"""
        try:
            await self.rate_limiter.wait()
            
            # First, submit URL for scanning
            scan_url = f"{self.endpoints['virustotal']['base_url']}{self.endpoints['virustotal']['url_scan']}"
            
            data = {
                'apikey': self.config.virustotal_api_key,
                'url': url
            }
            
            async with self.session.post(scan_url, data=data) as response:
                if response.status == 200:
                    scan_result = await response.json()
                    
                    # Wait a moment, then get report
                    await asyncio.sleep(5)
                    
                    report_url = f"{self.endpoints['virustotal']['base_url']}{self.endpoints['virustotal']['url_report']}"
                    
                    params = {
                        'apikey': self.config.virustotal_api_key,
                        'resource': url
                    }
                    
                    async with self.session.get(report_url, params=params) as report_response:
                        if report_response.status == 200:
                            report_data = await report_response.json()
                            
                            if report_data.get('response_code') == 1:
                                return self._process_url_analysis_response(report_data, url)
                            
        except Exception as e:
            self.logger.error(f"VirusTotal URL analysis failed: {e}")
        
        return None
    
    def _process_url_analysis_response(self, data: Dict, url: str) -> Dict[str, Any]:
        """Process VirusTotal URL analysis response"""
        scans = data.get('scans', {})
        total_scans = data.get('total', 0)
        positives = data.get('positives', 0)
        
        detection_ratio = positives / total_scans if total_scans > 0 else 0
        
        if positives == 0:
            confidence = 0.8
            risk_level = "safe"
            details = f"URL is clean according to {total_scans} engines"
        elif positives <= 2:
            confidence = 0.6
            risk_level = "low"
            details = f"URL flagged by {positives}/{total_scans} engines"
        else:
            confidence = 0.9
            risk_level = "high"
            details = f"URL flagged by {positives}/{total_scans} engines - likely malicious"
        
        return {
            "type": "url_reputation",
            "method": "virustotal_url_analysis",
            "tool_name": "virustotal",
            "confidence": confidence,
            "details": details,
            "url": url,
            "total_scans": total_scans,
            "positive_detections": positives,
            "detection_ratio": detection_ratio,
            "risk_level": risk_level,
            "scan_date": data.get('scan_date'),
            "permalink": data.get('permalink')
        }
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get API usage statistics"""
        return {
            "cache_size": len(self.cache),
            "rate_limiter_calls": getattr(self.rate_limiter, 'call_count', 0),
            "enabled_services": self._get_enabled_services()
        }
    
    def _get_enabled_services(self) -> List[str]:
        """Get list of enabled cloud services"""
        services = []
        
        if self.config.virustotal_api_key:
            services.append("VirusTotal")
        if self.config.nsrl_enabled:
            services.append("NSRL")
        if self.config.hashlookup_enabled:
            services.append("CIRCL Hash Lookup")
        if self.config.malware_bazaar_enabled:
            services.append("Malware Bazaar")
        if self.config.hybrid_analysis_api_key:
            services.append("Hybrid Analysis")
        
        return services

class RateLimiter:
    """Simple rate limiter for API calls"""
    
    def __init__(self, calls_per_second: float):
        self.calls_per_second = calls_per_second
        self.min_interval = 1.0 / calls_per_second if calls_per_second > 0 else 1.0
        self.last_call = 0.0
        self.call_count = 0
    
    async def wait(self):
        """Wait if necessary to respect rate limit"""
        if self.calls_per_second <= 0:
            return
        
        current_time = time.time()
        time_since_last = current_time - self.last_call
        
        if time_since_last < self.min_interval:
            wait_time = self.min_interval - time_since_last
            await asyncio.sleep(wait_time)
        
        self.last_call = time.time()
        self.call_count += 1