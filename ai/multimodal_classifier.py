"""
Multimodal Classifier - Advanced AI-powered file analysis
Combines vision models, OCR, CLIP, and multimodal transformers for comprehensive file understanding
"""

import asyncio
import logging
import numpy as np
import torch
import cv2
from PIL import Image
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import base64
import json

# Vision and multimodal models
try:
    import transformers
    from transformers import (
        CLIPProcessor, CLIPModel,
        TrOCRProcessor, VisionEncoderDecoderModel,
        BlipProcessor, BlipForConditionalGeneration
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# OCR libraries
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False

# Additional vision libraries
try:
    from torchvision import transforms
    import torchvision.models as models
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False

class MultimodalClassifier:
    def __init__(self, config):
        self.config = config.multimodal
        self.logger = logging.getLogger(__name__)
        self.device = self._setup_device()
        
        # Model cache
        self.models = {}
        self.processors = {}
        
        # OCR readers
        self.ocr_readers = {}
        
        # Image preprocessing
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize models
        asyncio.create_task(self._load_models())
    
    def _setup_device(self):
        """Setup computing device"""
        if not self.config.gpu_acceleration or not torch.cuda.is_available():
            return torch.device('cpu')
        
        return torch.device('cuda:0')
    
    async def _load_models(self):
        """Load all multimodal models"""
        if not TRANSFORMERS_AVAILABLE:
            self.logger.warning("Transformers not available, multimodal analysis limited")
            return
        
        try:
            # Load CLIP model for image-text understanding
            if self.config.enabled:
                await self._load_clip_model()
                await self._load_trocr_model()
                await self._load_blip_model()
                await self._load_ocr_models()
                
        except Exception as e:
            self.logger.error(f"Failed to load multimodal models: {e}")
    
    async def _load_clip_model(self):
        """Load CLIP model for image-text similarity"""
        try:
            model_name = self.config.clip_model
            self.processors['clip'] = CLIPProcessor.from_pretrained(model_name)
            self.models['clip'] = CLIPModel.from_pretrained(model_name).to(self.device)
            self.models['clip'].eval()
            self.logger.info(f"Loaded CLIP model: {model_name}")
        except Exception as e:
            self.logger.error(f"Failed to load CLIP model: {e}")
    
    async def _load_trocr_model(self):
        """Load TrOCR model for advanced OCR"""
        try:
            model_name = self.config.trocr_model
            self.processors['trocr'] = TrOCRProcessor.from_pretrained(model_name)
            self.models['trocr'] = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device)
            self.models['trocr'].eval()
            self.logger.info(f"Loaded TrOCR model: {model_name}")
        except Exception as e:
            self.logger.error(f"Failed to load TrOCR model: {e}")
    
    async def _load_blip_model(self):
        """Load BLIP model for image captioning and QA"""
        try:
            model_name = "Salesforce/blip-image-captioning-base"
            self.processors['blip'] = BlipProcessor.from_pretrained(model_name)
            self.models['blip'] = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)
            self.models['blip'].eval()
            self.logger.info(f"Loaded BLIP model: {model_name}")
        except Exception as e:
            self.logger.error(f"Failed to load BLIP model: {e}")
    
    async def _load_ocr_models(self):
        """Load OCR models"""
        try:
            # EasyOCR
            if EASYOCR_AVAILABLE:
                self.ocr_readers['easyocr'] = easyocr.Reader(['en'], gpu=self.device.type == 'cuda')
                self.logger.info("Loaded EasyOCR model")
        except Exception as e:
            self.logger.error(f"Failed to load OCR models: {e}")
    
    async def analyze_file_comprehensive(self, file_path: Path) -> Dict[str, Any]:
        """Comprehensive multimodal analysis of a file"""
        results = {
            "file_path": str(file_path),
            "analysis_type": "multimodal",
            "timestamp": np.datetime64('now').isoformat(),
            "models_used": [],
            "findings": {}
        }
        
        try:
            # Determine file type and apply appropriate analysis
            file_type = self._detect_file_type(file_path)
            results["detected_file_type"] = file_type
            
            if file_type.startswith('image/'):
                image_analysis = await self._analyze_image_multimodal(file_path)
                results["findings"].update(image_analysis)
                
            elif file_type.startswith('text/'):
                text_analysis = await self._analyze_text_multimodal(file_path)
                results["findings"].update(text_analysis)
                
            elif file_type.startswith('audio/'):
                audio_analysis = await self._analyze_audio_multimodal(file_path)
                results["findings"].update(audio_analysis)
                
            else:
                # Binary file analysis
                binary_analysis = await self._analyze_binary_multimodal(file_path)
                results["findings"].update(binary_analysis)
            
            # Cross-modal analysis if multiple content types detected
            cross_modal = await self._perform_cross_modal_analysis(file_path, results["findings"])
            if cross_modal:
                results["findings"]["cross_modal"] = cross_modal
                
        except Exception as e:
            self.logger.error(f"Multimodal analysis failed for {file_path}: {e}")
            results["error"] = str(e)
        
        return results
    
    def _detect_file_type(self, file_path: Path) -> str:
        """Detect file MIME type"""
        try:
            import magic
            return magic.from_file(str(file_path), mime=True)
        except:
            # Fallback to extension-based detection
            ext = file_path.suffix.lower()
            if ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
                return 'image/jpeg'
            elif ext in ['.txt', '.md']:
                return 'text/plain'
            elif ext in ['.wav', '.mp3', '.flac']:
                return 'audio/wav'
            else:
                return 'application/octet-stream'
    
    async def _analyze_image_multimodal(self, file_path: Path) -> Dict[str, Any]:
        """Comprehensive multimodal image analysis"""
        findings = {}
        
        try:
            # Load image
            image = Image.open(file_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # CLIP-based analysis
            if 'clip' in self.models:
                clip_analysis = await self._clip_image_analysis(image)
                findings["clip_analysis"] = clip_analysis
            
            # Image captioning with BLIP
            if 'blip' in self.models:
                caption_analysis = await self._generate_image_caption(image)
                findings["caption_analysis"] = caption_analysis
            
            # Advanced OCR analysis
            ocr_analysis = await self._comprehensive_ocr_analysis(image)
            findings["ocr_analysis"] = ocr_analysis
            
            # Visual feature analysis
            visual_features = await self._extract_visual_features(image)
            findings["visual_features"] = visual_features
            
            # Steganography-specific analysis
            stego_indicators = await self._detect_image_stego_indicators(image)
            findings["steganography_indicators"] = stego_indicators
            
        except Exception as e:
            self.logger.error(f"Image multimodal analysis failed: {e}")
            findings["error"] = str(e)
        
        return findings
    
    async def _clip_image_analysis(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze image using CLIP model"""
        analysis = {}
        
        try:
            # Suspicious text queries for steganography detection
            suspicious_queries = [
                "hidden message",
                "secret text", 
                "encoded data",
                "steganography",
                "concealed information",
                "encrypted text",
                "base64 encoded text",
                "hexadecimal data",
                "QR code",
                "barcode",
                "watermark",
                "digital signature"
            ]
            
            # Process image and text
            inputs = self.processors['clip'](
                text=suspicious_queries,
                images=image,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.models['clip'](**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
            
            # Get probabilities for each query
            query_scores = {}
            for i, query in enumerate(suspicious_queries):
                score = float(probs[0][i])
                query_scores[query] = score
            
            # Find highest scoring queries
            top_matches = sorted(query_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            
            analysis["query_scores"] = query_scores
            analysis["top_matches"] = top_matches
            analysis["max_score"] = max(query_scores.values())
            analysis["suspicious_content_detected"] = analysis["max_score"] > 0.3
            
        except Exception as e:
            self.logger.error(f"CLIP analysis failed: {e}")
            analysis["error"] = str(e)
        
        return analysis
    
    async def _generate_image_caption(self, image: Image.Image) -> Dict[str, Any]:
        """Generate descriptive caption for image"""
        analysis = {}
        
        try:
            # Generate caption
            inputs = self.processors['blip'](image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                out = self.models['blip'].generate(**inputs, max_length=50, num_beams=5)
            
            caption = self.processors['blip'].decode(out[0], skip_special_tokens=True)
            
            analysis["caption"] = caption
            analysis["caption_length"] = len(caption)
            
            # Analyze caption for suspicious content
            suspicious_keywords = [
                "text", "writing", "document", "code", "numbers", 
                "symbols", "pattern", "grid", "data", "message"
            ]
            
            found_keywords = [kw for kw in suspicious_keywords if kw in caption.lower()]
            analysis["suspicious_keywords"] = found_keywords
            analysis["text_content_suggested"] = len(found_keywords) > 0
            
        except Exception as e:
            self.logger.error(f"Image captioning failed: {e}")
            analysis["error"] = str(e)
        
        return analysis
    
    async def _comprehensive_ocr_analysis(self, image: Image.Image) -> Dict[str, Any]:
        """Comprehensive OCR analysis using multiple engines"""
        analysis = {
            "engines_used": [],
            "extracted_text": {},
            "combined_text": "",
            "text_confidence": {},
            "text_locations": {}
        }
        
        # Convert PIL to numpy array for OCR
        image_np = np.array(image)
        
        # EasyOCR
        if 'easyocr' in self.ocr_readers:
            try:
                easyocr_results = self.ocr_readers['easyocr'].readtext(image_np)
                
                easyocr_text = ""
                easyocr_confidences = []
                easyocr_locations = []
                
                for (bbox, text, confidence) in easyocr_results:
                    easyocr_text += text + " "
                    easyocr_confidences.append(confidence)
                    easyocr_locations.append({
                        "text": text,
                        "bbox": bbox,
                        "confidence": confidence
                    })
                
                analysis["engines_used"].append("easyocr")
                analysis["extracted_text"]["easyocr"] = easyocr_text.strip()
                analysis["text_confidence"]["easyocr"] = np.mean(easyocr_confidences) if easyocr_confidences else 0
                analysis["text_locations"]["easyocr"] = easyocr_locations
                
            except Exception as e:
                self.logger.error(f"EasyOCR failed: {e}")
        
        # Tesseract OCR
        if PYTESSERACT_AVAILABLE:
            try:
                tesseract_text = pytesseract.image_to_string(image)
                tesseract_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
                
                # Calculate average confidence
                confidences = [int(conf) for conf in tesseract_data['conf'] if int(conf) > 0]
                avg_confidence = np.mean(confidences) / 100 if confidences else 0
                
                analysis["engines_used"].append("tesseract")
                analysis["extracted_text"]["tesseract"] = tesseract_text.strip()
                analysis["text_confidence"]["tesseract"] = avg_confidence
                
            except Exception as e:
                self.logger.error(f"Tesseract OCR failed: {e}")
        
        # TrOCR (Transformer-based OCR)
        if 'trocr' in self.models:
            try:
                pixel_values = self.processors['trocr'](image, return_tensors="pt").pixel_values.to(self.device)
                
                with torch.no_grad():
                    generated_ids = self.models['trocr'].generate(pixel_values, max_length=200)
                
                trocr_text = self.processors['trocr'].batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                analysis["engines_used"].append("trocr")
                analysis["extracted_text"]["trocr"] = trocr_text.strip()
                analysis["text_confidence"]["trocr"] = 0.8  # TrOCR doesn't provide confidence scores
                
            except Exception as e:
                self.logger.error(f"TrOCR failed: {e}")
        
        # Combine all extracted text
        all_texts = [text for text in analysis["extracted_text"].values() if text]
        analysis["combined_text"] = " ".join(all_texts)
        
        # Analyze extracted text for suspicious patterns
        text_analysis = await self._analyze_extracted_text_patterns(analysis["combined_text"])
        analysis["pattern_analysis"] = text_analysis
        
        return analysis
    
    async def _analyze_extracted_text_patterns(self, text: str) -> Dict[str, Any]:
        """Analyze extracted text for suspicious patterns"""
        import re
        
        patterns = {
            "base64": {
                "pattern": re.compile(r'[A-Za-z0-9+/]{20,}={0,2}'),
                "description": "Base64 encoded data"
            },
            "hex": {
                "pattern": re.compile(r'[0-9A-Fa-f]{16,}'),
                "description": "Hexadecimal data"
            },
            "urls": {
                "pattern": re.compile(r'https?://[^\s]+'),
                "description": "URLs"
            },
            "emails": {
                "pattern": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
                "description": "Email addresses"
            },
            "coordinates": {
                "pattern": re.compile(r'[-+]?\d*\.?\d+,\s*[-+]?\d*\.?\d+'),
                "description": "GPS coordinates"
            },
            "phone_numbers": {
                "pattern": re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),
                "description": "Phone numbers"
            },
            "crypto_addresses": {
                "pattern": re.compile(r'\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b'),
                "description": "Bitcoin addresses"
            },
            "flags": {
                "pattern": re.compile(r'flag\{[^}]+\}', re.IGNORECASE),
                "description": "CTF flags"
            }
        }
        
        analysis = {
            "text_length": len(text),
            "patterns_found": {},
            "suspicious_score": 0.0
        }
        
        for pattern_name, pattern_info in patterns.items():
            matches = pattern_info["pattern"].findall(text)
            if matches:
                analysis["patterns_found"][pattern_name] = {
                    "count": len(matches),
                    "matches": matches[:5],  # First 5 matches
                    "description": pattern_info["description"]
                }
                
                # Increase suspicious score based on pattern type
                if pattern_name in ["base64", "hex", "crypto_addresses", "flags"]:
                    analysis["suspicious_score"] += 0.3
                else:
                    analysis["suspicious_score"] += 0.1
        
        analysis["suspicious_score"] = min(analysis["suspicious_score"], 1.0)
        analysis["contains_encoded_data"] = any(p in analysis["patterns_found"] for p in ["base64", "hex"])
        
        return analysis
    
    async def _extract_visual_features(self, image: Image.Image) -> Dict[str, Any]:
        """Extract visual features for analysis"""
        features = {}
        
        try:
            # Convert to numpy array
            img_array = np.array(image)
            
            # Basic image statistics
            features["dimensions"] = {
                "width": image.width,
                "height": image.height,
                "channels": len(img_array.shape)
            }
            
            features["color_statistics"] = {
                "mean_brightness": float(np.mean(img_array)),
                "std_brightness": float(np.std(img_array))
            }
            
            # Edge detection
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            features["edge_analysis"] = {
                "edge_density": float(edge_density),
                "has_structured_content": edge_density > 0.05
            }
            
            # Texture analysis using Local Binary Pattern
            try:
                from skimage.feature import local_binary_pattern
                
                lbp = local_binary_pattern(gray, 8, 1, method='uniform')
                lbp_hist, _ = np.histogram(lbp.ravel(), bins=10)
                lbp_uniformity = np.std(lbp_hist)
                
                features["texture_analysis"] = {
                    "lbp_uniformity": float(lbp_uniformity),
                    "texture_complexity": "high" if lbp_uniformity > 50 else "low"
                }
                
            except ImportError:
                pass
            
            # Color distribution analysis
            if img_array.shape[-1] == 3:  # RGB image
                color_hist = []
                for channel in range(3):
                    hist, _ = np.histogram(img_array[:, :, channel], bins=32, range=(0, 256))
                    color_hist.extend(hist.tolist())
                
                features["color_distribution"] = {
                    "histogram": color_hist,
                    "dominant_colors": self._find_dominant_colors(img_array)
                }
            
        except Exception as e:
            self.logger.error(f"Visual feature extraction failed: {e}")
            features["error"] = str(e)
        
        return features
    
    def _find_dominant_colors(self, img_array: np.ndarray, k: int = 5) -> List[List[int]]:
        """Find dominant colors in image"""
        try:
            from sklearn.cluster import KMeans
            
            # Reshape image to be a list of pixels
            pixels = img_array.reshape(-1, 3)
            
            # Sample pixels if image is large
            if len(pixels) > 10000:
                indices = np.random.choice(len(pixels), 10000, replace=False)
                pixels = pixels[indices]
            
            # Apply K-means clustering
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            # Get dominant colors
            colors = kmeans.cluster_centers_.astype(int).tolist()
            return colors
            
        except ImportError:
            return []
        except Exception:
            return []
    
    async def _detect_image_stego_indicators(self, image: Image.Image) -> Dict[str, Any]:
        """Detect steganography indicators in image"""
        indicators = {
            "visual_anomalies": [],
            "metadata_anomalies": [],
            "statistical_anomalies": [],
            "overall_suspicion": 0.0
        }
        
        try:
            img_array = np.array(image)
            
            # Check for LSB modification indicators
            if len(img_array.shape) == 3:
                for channel in range(3):
                    lsb_analysis = self._analyze_lsb_channel(img_array[:, :, channel])
                    if lsb_analysis["suspicious"]:
                        indicators["statistical_anomalies"].append({
                            "type": "lsb_anomaly",
                            "channel": channel,
                            "score": lsb_analysis["score"]
                        })
                        indicators["overall_suspicion"] += 0.2
            
            # Check for unusual aspect ratios or dimensions
            aspect_ratio = image.width / image.height
            if aspect_ratio > 5 or aspect_ratio < 0.2:
                indicators["visual_anomalies"].append({
                    "type": "unusual_aspect_ratio",
                    "ratio": aspect_ratio
                })
                indicators["overall_suspicion"] += 0.1
            
            # Check image metadata
            if hasattr(image, '_getexif') and image._getexif():
                exif_data = image._getexif()
                if exif_data and len(exif_data) > 20:
                    indicators["metadata_anomalies"].append({
                        "type": "excessive_metadata",
                        "field_count": len(exif_data)
                    })
                    indicators["overall_suspicion"] += 0.1
            
            # Compress and check for size anomalies
            original_size = len(img_array.tobytes())
            
            # Create a temporary compressed version
            with tempfile.NamedTemporaryFile(suffix='.jpg') as temp_file:
                image.save(temp_file.name, 'JPEG', quality=85)
                compressed_image = Image.open(temp_file.name)
                compressed_size = len(np.array(compressed_image).tobytes())
            
            compression_ratio = original_size / compressed_size if compressed_size > 0 else 1
            if compression_ratio < 1.5:  # Poor compression might indicate hidden data
                indicators["statistical_anomalies"].append({
                    "type": "poor_compression",
                    "ratio": compression_ratio
                })
                indicators["overall_suspicion"] += 0.15
            
            indicators["overall_suspicion"] = min(indicators["overall_suspicion"], 1.0)
            
        except Exception as e:
            self.logger.error(f"Steganography indicator detection failed: {e}")
            indicators["error"] = str(e)
        
        return indicators
    
    def _analyze_lsb_channel(self, channel_data: np.ndarray) -> Dict[str, Any]:
        """Analyze LSB of image channel for anomalies"""
        try:
            # Extract LSB plane
            lsb_plane = channel_data & 1
            
            # Calculate entropy of LSB plane
            unique, counts = np.unique(lsb_plane, return_counts=True)
            probabilities = counts / counts.sum()
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
            
            # High entropy in LSB plane suggests embedded data
            suspicious = entropy > 0.9
            
            return {
                "entropy": float(entropy),
                "suspicious": suspicious,
                "score": float(entropy)
            }
            
        except Exception:
            return {"entropy": 0.0, "suspicious": False, "score": 0.0}
    
    async def _analyze_text_multimodal(self, file_path: Path) -> Dict[str, Any]:
        """Analyze text files using multimodal approaches"""
        findings = {}
        
        try:
            # Read text content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text_content = f.read()
            
            # Text pattern analysis
            pattern_analysis = await self._analyze_extracted_text_patterns(text_content)
            findings["pattern_analysis"] = pattern_analysis
            
            # Language detection and analysis
            lang_analysis = await self._analyze_text_language(text_content)
            findings["language_analysis"] = lang_analysis
            
            # Semantic analysis if CLIP available
            if 'clip' in self.models:
                semantic_analysis = await self._analyze_text_semantics(text_content)
                findings["semantic_analysis"] = semantic_analysis
            
        except Exception as e:
            self.logger.error(f"Text multimodal analysis failed: {e}")
            findings["error"] = str(e)
        
        return findings
    
    async def _analyze_text_language(self, text: str) -> Dict[str, Any]:
        """Analyze text language characteristics"""
        analysis = {
            "text_length": len(text),
            "line_count": len(text.split('\n')),
            "word_count": len(text.split()),
            "character_distribution": {},
            "entropy": 0.0
        }
        
        # Character frequency analysis
        char_counts = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        # Calculate entropy
        total_chars = len(text)
        if total_chars > 0:
            probabilities = [count / total_chars for count in char_counts.values()]
            analysis["entropy"] = float(-np.sum([p * np.log2(p) for p in probabilities if p > 0]))
        
        # Analyze character types
        analysis["character_distribution"] = {
            "alphabetic": sum(1 for c in text if c.isalpha()) / total_chars if total_chars > 0 else 0,
            "numeric": sum(1 for c in text if c.isdigit()) / total_chars if total_chars > 0 else 0,
            "punctuation": sum(1 for c in text if c in '.,;:!?()[]{}') / total_chars if total_chars > 0 else 0,
            "whitespace": sum(1 for c in text if c.isspace()) / total_chars if total_chars > 0 else 0,
            "special": sum(1 for c in text if not c.isalnum() and not c.isspace() and c not in '.,;:!?()[]{}') / total_chars if total_chars > 0 else 0
        }
        
        # Determine if text looks encoded/encrypted
        analysis["looks_encoded"] = (
            analysis["entropy"] > 7.0 or
            analysis["character_distribution"]["special"] > 0.3 or
            analysis["character_distribution"]["numeric"] > 0.7
        )
        
        return analysis
    
    async def _analyze_text_semantics(self, text: str) -> Dict[str, Any]:
        """Analyze text semantics using CLIP"""
        analysis = {}
        
        try:
            # Define semantic queries
            semantic_queries = [
                "encrypted message",
                "hidden information", 
                "secret code",
                "steganographic content",
                "base64 encoded data",
                "programming code",
                "configuration file",
                "log file",
                "normal text document"
            ]
            
            # Truncate text if too long
            text_sample = text[:1000] if len(text) > 1000 else text
            
            # Process with CLIP
            inputs = self.processors['clip'](
                text=[text_sample],
                text_pair=semantic_queries,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.models['clip'](**inputs)
                # For text-text similarity, we need to compute similarity differently
                text_embeds = outputs.text_embeds
                similarities = torch.cosine_similarity(
                    text_embeds[0:1].repeat(len(semantic_queries), 1),
                    text_embeds[1:],
                    dim=1
                )
            
            # Get similarity scores
            query_scores = {}
            for i, query in enumerate(semantic_queries):
                score = float(similarities[i])
                query_scores[query] = score
            
            analysis["semantic_scores"] = query_scores
            analysis["most_similar"] = max(query_scores.items(), key=lambda x: x[1])
            analysis["encoded_content_likelihood"] = max(
                query_scores.get("encrypted message", 0),
                query_scores.get("base64 encoded data", 0),
                query_scores.get("secret code", 0)
            )
            
        except Exception as e:
            self.logger.error(f"Text semantic analysis failed: {e}")
            analysis["error"] = str(e)
        
        return analysis
    
    async def _analyze_audio_multimodal(self, file_path: Path) -> Dict[str, Any]:
        """Analyze audio files (placeholder for future implementation)"""
        return {
            "message": "Audio multimodal analysis not yet implemented",
            "file_path": str(file_path)
        }
    
    async def _analyze_binary_multimodal(self, file_path: Path) -> Dict[str, Any]:
        """Analyze binary files for embedded content"""
        findings = {}
        
        try:
            # Read binary data
            with open(file_path, 'rb') as f:
                binary_data = f.read(8192)  # First 8KB
            
            # Entropy analysis
            entropy = self._calculate_binary_entropy(binary_data)
            findings["entropy_analysis"] = {
                "entropy": entropy,
                "likely_compressed_or_encrypted": entropy > 7.5
            }
            
            # Look for embedded images/text within binary
            embedded_content = await self._detect_embedded_content(binary_data)
            findings["embedded_content"] = embedded_content
            
            # File signature analysis
            signature_analysis = self._analyze_file_signatures(binary_data)
            findings["signature_analysis"] = signature_analysis
            
        except Exception as e:
            self.logger.error(f"Binary multimodal analysis failed: {e}")
            findings["error"] = str(e)
        
        return findings
    
    def _calculate_binary_entropy(self, data: bytes) -> float:
        """Calculate entropy of binary data"""
        if len(data) == 0:
            return 0.0
        
        # Count byte frequencies
        byte_counts = [0] * 256
        for byte in data:
            byte_counts[byte] += 1
        
        # Calculate entropy
        entropy = 0.0
        length = len(data)
        
        for count in byte_counts:
            if count > 0:
                probability = count / length
                entropy -= probability * np.log2(probability)
        
        return entropy
    
    async def _detect_embedded_content(self, binary_data: bytes) -> Dict[str, Any]:
        """Detect embedded content in binary data"""
        content = {
            "embedded_images": [],
            "embedded_text": [],
            "suspicious_patterns": []
        }
        
        # Look for image signatures
        image_signatures = {
            b'\xFF\xD8\xFF': 'JPEG',
            b'\x89PNG': 'PNG',
            b'GIF8': 'GIF',
            b'BM': 'BMP'
        }
        
        for sig, img_type in image_signatures.items():
            pos = binary_data.find(sig)
            if pos > 0:  # Not at the beginning
                content["embedded_images"].append({
                    "type": img_type,
                    "position": pos,
                    "signature": sig.hex()
                })
        
        # Look for text patterns
        try:
            text_content = binary_data.decode('utf-8', errors='ignore')
            if len(text_content) > 100:
                # Check if there's readable text
                readable_ratio = sum(1 for c in text_content if c.isprintable()) / len(text_content)
                if readable_ratio > 0.7:
                    content["embedded_text"].append({
                        "length": len(text_content),
                        "readable_ratio": readable_ratio,
                        "preview": text_content[:200]
                    })
        except:
            pass
        
        return content
    
    def _analyze_file_signatures(self, binary_data: bytes) -> Dict[str, Any]:
        """Analyze file signatures in binary data"""
        signatures = {
            "detected_formats": [],
            "multiple_formats": False
        }
        
        # Common file signatures
        file_sigs = {
            b'%PDF': 'PDF',
            b'PK\x03\x04': 'ZIP',
            b'Rar!': 'RAR',
            b'\x7FELF': 'ELF',
            b'MZ': 'PE'
        }
        
        detected = []
        for sig, format_name in file_sigs.items():
            if sig in binary_data:
                detected.append(format_name)
        
        signatures["detected_formats"] = detected
        signatures["multiple_formats"] = len(detected) > 1
        
        return signatures
    
    async def _perform_cross_modal_analysis(self, file_path: Path, findings: Dict[str, Any]) -> Dict[str, Any]:
        """Perform cross-modal analysis to correlate findings"""
        cross_modal = {
            "correlations": [],
            "confidence_score": 0.0,
            "recommendations": []
        }
        
        try:
            # Check for correlations between different analysis types
            if "ocr_analysis" in findings and "visual_features" in findings:
                # Correlate OCR findings with visual features
                ocr_data = findings["ocr_analysis"]
                visual_data = findings["visual_features"]
                
                if (ocr_data.get("pattern_analysis", {}).get("suspicious_score", 0) > 0.5 and
                    visual_data.get("edge_analysis", {}).get("has_structured_content", False)):
                    
                    cross_modal["correlations"].append({
                        "type": "ocr_visual_correlation",
                        "description": "Suspicious text patterns correlate with structured visual content",
                        "confidence": 0.8
                    })
                    cross_modal["confidence_score"] += 0.3
            
            # Check for steganography indicators
            if "steganography_indicators" in findings:
                stego_indicators = findings["steganography_indicators"]
                if stego_indicators.get("overall_suspicion", 0) > 0.5:
                    cross_modal["correlations"].append({
                        "type": "steganography_correlation",
                        "description": "Multiple steganography indicators detected",
                        "confidence": stego_indicators["overall_suspicion"]
                    })
                    cross_modal["confidence_score"] += 0.4
            
            # Generate recommendations
            if cross_modal["confidence_score"] > 0.5:
                cross_modal["recommendations"].extend([
                    "Perform deeper steganographic analysis",
                    "Examine file with specialized forensic tools",
                    "Check for additional hidden layers or channels"
                ])
            
            cross_modal["confidence_score"] = min(cross_modal["confidence_score"], 1.0)
            
        except Exception as e:
            self.logger.error(f"Cross-modal analysis failed: {e}")
            cross_modal["error"] = str(e)
        
        return cross_modal
    
    def cleanup(self):
        """Cleanup models and free GPU memory"""
        # Clear models
        for model in self.models.values():
            if hasattr(model, 'cpu'):
                model.cpu()
            del model
        
        self.models.clear()
        self.processors.clear()
        
        # Clear OCR readers
        self.ocr_readers.clear()
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.info("Multimodal classifier cleaned up")