"""
GPU Manager - Manages GPU resources and allocation for ML/AI tasks
"""

import logging
import asyncio
import time
import threading
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from contextlib import asynccontextmanager
import psutil
from concurrent.futures import ThreadPoolExecutor

@dataclass
class GPUDevice:
    id: int
    name: str
    memory_total: int
    memory_free: int
    memory_used: int
    utilization: float
    temperature: float
    power_usage: float
    available: bool

@dataclass
class GPUTask:
    task_id: str
    session_id: str
    method: str
    estimated_memory: int
    estimated_time: float
    priority: int
    created_at: float

class GPUManager:
    def __init__(self, config=None):
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # GPU availability and management
        self.gpu_available = False
        self.gpu_devices = []
        self.device_locks = {}
        self.task_queue = asyncio.Queue()
        self.active_tasks = {}
        
        # Resource tracking
        self.memory_threshold = 0.9  # Use up to 90% of GPU memory
        self.max_concurrent_tasks = 2
        self.task_timeout = 3600  # 1 hour timeout
        
        # Performance monitoring
        self.task_history = []
        self.performance_stats = {}
        
        # Semaphores for resource control
        self.gpu_semaphore = None
        self.memory_semaphore = None
        
        # Initialize GPU detection
        self._initialize_gpu_detection()
        
        # Start monitoring thread
        self.monitoring_thread = None
        self.shutdown_event = threading.Event()
        self._start_monitoring()
    
    def _initialize_gpu_detection(self):
        """Initialize GPU detection and setup"""
        try:
            # Try to detect NVIDIA GPUs first
            nvidia_available = self._detect_nvidia_gpus()
            
            if not nvidia_available:
                # Try AMD GPUs
                amd_available = self._detect_amd_gpus()
                
                if not amd_available:
                    # Try Intel GPUs
                    intel_available = self._detect_intel_gpus()
                    
                    if not intel_available:
                        self.logger.warning("No GPUs detected or GPU libraries not available")
                        self.gpu_available = False
                        return
            
            if self.gpu_devices:
                self.gpu_available = True
                self.gpu_semaphore = asyncio.Semaphore(min(len(self.gpu_devices), self.max_concurrent_tasks))
                self.logger.info(f"Initialized GPU manager with {len(self.gpu_devices)} devices")
                
                # Initialize device locks
                for device in self.gpu_devices:
                    self.device_locks[device.id] = asyncio.Lock()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize GPU detection: {e}")
            self.gpu_available = False
    
    def _detect_nvidia_gpus(self) -> bool:
        """Detect NVIDIA GPUs using various methods"""
        try:
            # Try pynvml first (most reliable)
            try:
                import pynvml
                pynvml.nvmlInit()
                
                device_count = pynvml.nvmlDeviceGetCount()
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                    
                    # Get memory info
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    
                    # Get utilization
                    try:
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        utilization = util.gpu
                    except:
                        utilization = 0.0
                    
                    # Get temperature
                    try:
                        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    except:
                        temp = 0.0
                    
                    # Get power usage
                    try:
                        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                    except:
                        power = 0.0
                    
                    device = GPUDevice(
                        id=i,
                        name=name,
                        memory_total=memory_info.total,
                        memory_free=memory_info.free,
                        memory_used=memory_info.used,
                        utilization=utilization,
                        temperature=temp,
                        power_usage=power,
                        available=True
                    )
                    
                    self.gpu_devices.append(device)
                
                return len(self.gpu_devices) > 0
                
            except ImportError:
                self.logger.debug("pynvml not available, trying PyTorch")
                
            # Try PyTorch CUDA
            try:
                import torch
                if torch.cuda.is_available():
                    device_count = torch.cuda.device_count()
                    for i in range(device_count):
                        name = torch.cuda.get_device_name(i)
                        props = torch.cuda.get_device_properties(i)
                        
                        device = GPUDevice(
                            id=i,
                            name=name,
                            memory_total=props.total_memory,
                            memory_free=props.total_memory,  # Approximate
                            memory_used=0,
                            utilization=0.0,
                            temperature=0.0,
                            power_usage=0.0,
                            available=True
                        )
                        
                        self.gpu_devices.append(device)
                    
                    return len(self.gpu_devices) > 0
                    
            except ImportError:
                self.logger.debug("PyTorch not available")
            
            # Try nvidia-ml-py
            try:
                import nvidia_ml_py3 as nvml
                nvml.nvmlInit()
                
                device_count = nvml.nvmlDeviceGetCount()
                for i in range(device_count):
                    handle = nvml.nvmlDeviceGetHandleByIndex(i)
                    name = nvml.nvmlDeviceGetName(handle).decode('utf-8')
                    
                    memory_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                    
                    device = GPUDevice(
                        id=i,
                        name=name,
                        memory_total=memory_info.total,
                        memory_free=memory_info.free,
                        memory_used=memory_info.used,
                        utilization=0.0,
                        temperature=0.0,
                        power_usage=0.0,
                        available=True
                    )
                    
                    self.gpu_devices.append(device)
                
                return len(self.gpu_devices) > 0
                
            except ImportError:
                self.logger.debug("nvidia-ml-py3 not available")
            
            return False
            
        except Exception as e:
            self.logger.debug(f"NVIDIA GPU detection failed: {e}")
            return False
    
    def _detect_amd_gpus(self) -> bool:
        """Detect AMD GPUs"""
        try:
            # Try ROCm if available
            import subprocess
            result = subprocess.run(['rocm-smi', '--showid'], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for i, line in enumerate(lines[1:]):  # Skip header
                    if 'GPU' in line:
                        device = GPUDevice(
                            id=i,
                            name=f"AMD GPU {i}",
                            memory_total=0,  # Would need more detailed detection
                            memory_free=0,
                            memory_used=0,
                            utilization=0.0,
                            temperature=0.0,
                            power_usage=0.0,
                            available=True
                        )
                        self.gpu_devices.append(device)
                
                return len(self.gpu_devices) > 0
            
        except (FileNotFoundError, subprocess.TimeoutExpired, ImportError):
            self.logger.debug("AMD GPU detection failed")
        
        return False
    
    def _detect_intel_gpus(self) -> bool:
        """Detect Intel GPUs"""
        try:
            # Basic Intel GPU detection (limited support)
            import subprocess
            result = subprocess.run(['lspci'], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                intel_gpus = [line for line in result.stdout.split('\n') if 'Intel' in line and 'VGA' in line]
                
                for i, gpu_line in enumerate(intel_gpus):
                    device = GPUDevice(
                        id=i + 1000,  # Offset to avoid conflicts
                        name=f"Intel GPU {i}",
                        memory_total=0,
                        memory_free=0,
                        memory_used=0,
                        utilization=0.0,
                        temperature=0.0,
                        power_usage=0.0,
                        available=False  # Limited support
                    )
                    self.gpu_devices.append(device)
                
                return len(self.gpu_devices) > 0
            
        except (FileNotFoundError, subprocess.TimeoutExpired):
            self.logger.debug("Intel GPU detection failed")
        
        return False
    
    def _start_monitoring(self):
        """Start GPU monitoring thread"""
        if self.gpu_available:
            self.monitoring_thread = threading.Thread(target=self._monitor_gpus, daemon=True)
            self.monitoring_thread.start()
    
    def _monitor_gpus(self):
        """Monitor GPU status in background thread"""
        while not self.shutdown_event.is_set():
            try:
                self._update_gpu_status()
                time.sleep(5)  # Update every 5 seconds
            except Exception as e:
                self.logger.debug(f"GPU monitoring error: {e}")
                time.sleep(10)  # Wait longer on error
    
    def _update_gpu_status(self):
        """Update GPU device status"""
        if not self.gpu_available:
            return
        
        try:
            # Update NVIDIA GPUs
            try:
                import pynvml
                for device in self.gpu_devices:
                    if device.id < 1000:  # NVIDIA devices
                        handle = pynvml.nvmlDeviceGetHandleByIndex(device.id)
                        
                        # Update memory info
                        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        device.memory_free = memory_info.free
                        device.memory_used = memory_info.used
                        
                        # Update utilization
                        try:
                            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                            device.utilization = util.gpu
                        except:
                            pass
                        
                        # Update temperature
                        try:
                            device.temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                        except:
                            pass
                        
                        # Update power
                        try:
                            device.power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                        except:
                            pass
                        
                        # Check availability
                        memory_usage_ratio = device.memory_used / device.memory_total
                        device.available = (memory_usage_ratio < self.memory_threshold and 
                                          device.utilization < 95.0 and 
                                          device.temperature < 85.0)
            
            except ImportError:
                pass
            
        except Exception as e:
            self.logger.debug(f"Failed to update GPU status: {e}")
    
    async def allocate_gpu(self, task: GPUTask) -> Optional[int]:
        """Allocate GPU for a task"""
        if not self.gpu_available:
            return None
        
        try:
            # Wait for GPU semaphore
            await self.gpu_semaphore.acquire()
            
            # Find best available GPU
            best_device = self._find_best_gpu(task.estimated_memory)
            
            if best_device is None:
                self.gpu_semaphore.release()
                return None
            
            # Lock the device
            async with self.device_locks[best_device.id]:
                # Double-check availability
                if not best_device.available:
                    self.gpu_semaphore.release()
                    return None
                
                # Allocate task
                self.active_tasks[task.task_id] = {
                    "task": task,
                    "device_id": best_device.id,
                    "start_time": time.time(),
                    "estimated_end": time.time() + task.estimated_time
                }
                
                self.logger.info(f"Allocated GPU {best_device.id} for task {task.task_id}")
                return best_device.id
        
        except Exception as e:
            self.logger.error(f"Failed to allocate GPU: {e}")
            if self.gpu_semaphore:
                self.gpu_semaphore.release()
            return None
    
    async def release_gpu(self, task_id: str):
        """Release GPU allocation for a task"""
        try:
            if task_id in self.active_tasks:
                task_info = self.active_tasks[task_id]
                device_id = task_info["device_id"]
                
                # Record performance
                end_time = time.time()
                actual_time = end_time - task_info["start_time"]
                
                self.task_history.append({
                    "task_id": task_id,
                    "device_id": device_id,
                    "start_time": task_info["start_time"],
                    "end_time": end_time,
                    "actual_time": actual_time,
                    "estimated_time": task_info["task"]["estimated_time"]
                })
                
                # Remove from active tasks
                del self.active_tasks[task_id]
                
                # Release semaphore
                self.gpu_semaphore.release()
                
                self.logger.info(f"Released GPU {device_id} for task {task_id}")
                
                # Clean up old history
                if len(self.task_history) > 1000:
                    self.task_history = self.task_history[-500:]
        
        except Exception as e:
            self.logger.error(f"Failed to release GPU for task {task_id}: {e}")
    
    @asynccontextmanager
    async def gpu_context(self, task: GPUTask):
        """Context manager for GPU allocation"""
        device_id = None
        try:
            device_id = await self.allocate_gpu(task)
            if device_id is None:
                raise RuntimeError("Failed to allocate GPU")
            yield device_id
        finally:
            if device_id is not None:
                await self.release_gpu(task.task_id)
    
    def _find_best_gpu(self, required_memory: int) -> Optional[GPUDevice]:
        """Find the best available GPU for a task"""
        available_devices = [d for d in self.gpu_devices if d.available]
        
        if not available_devices:
            return None
        
        # Filter by memory requirement
        suitable_devices = [
            d for d in available_devices 
            if d.memory_free >= required_memory
        ]
        
        if not suitable_devices:
            return None
        
        # Sort by utilization (prefer less utilized)
        suitable_devices.sort(key=lambda d: (d.utilization, d.memory_used))
        
        return suitable_devices[0]
    
    def get_gpu_status(self) -> Dict[str, Any]:
        """Get current GPU status"""
        if not self.gpu_available:
            return {
                "available": False,
                "devices": [],
                "active_tasks": 0
            }
        
        return {
            "available": True,
            "devices": [
                {
                    "id": d.id,
                    "name": d.name,
                    "memory_total": d.memory_total,
                    "memory_free": d.memory_free,
                    "memory_used": d.memory_used,
                    "memory_usage_percent": (d.memory_used / d.memory_total * 100) if d.memory_total > 0 else 0,
                    "utilization": d.utilization,
                    "temperature": d.temperature,
                    "power_usage": d.power_usage,
                    "available": d.available
                }
                for d in self.gpu_devices
            ],
            "active_tasks": len(self.active_tasks),
            "queue_size": self.task_queue.qsize() if hasattr(self.task_queue, 'qsize') else 0
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get GPU performance statistics"""
        if not self.task_history:
            return {"no_data": True}
        
        total_tasks = len(self.task_history)
        total_time = sum(t["actual_time"] for t in self.task_history)
        avg_time = total_time / total_tasks if total_tasks > 0 else 0
        
        # Calculate time estimation accuracy
        time_errors = []
        for task in self.task_history:
            if task["estimated_time"] > 0:
                error = abs(task["actual_time"] - task["estimated_time"]) / task["estimated_time"]
                time_errors.append(error)
        
        avg_error = sum(time_errors) / len(time_errors) if time_errors else 0
        
        return {
            "total_tasks_completed": total_tasks,
            "average_task_time": avg_time,
            "total_gpu_time": total_time,
            "time_estimation_error": avg_error,
            "device_usage": {
                device.id: sum(1 for t in self.task_history if t["device_id"] == device.id)
                for device in self.gpu_devices
            }
        }
    
    def estimate_task_time(self, method: str, file_size: int) -> float:
        """Estimate task execution time based on history"""
        # Base estimates by method type
        base_estimates = {
            "cnn_steg_detection": 30.0,
            "noiseprint": 45.0,
            "deep_stego": 60.0,
            "anomaly_detection": 25.0,
            "multimodal_classification": 20.0,
            "llm_content_analysis": 90.0
        }
        
        base_time = base_estimates.get(method, 30.0)
        
        # Adjust based on file size (rough heuristic)
        size_factor = min(file_size / (1024 * 1024), 10.0)  # Cap at 10x for very large files
        
        # Look for similar tasks in history
        similar_tasks = [
            t for t in self.task_history 
            if t.get("method") == method
        ]
        
        if similar_tasks:
            avg_historical = sum(t["actual_time"] for t in similar_tasks) / len(similar_tasks)
            # Blend historical and base estimate
            estimated_time = (base_time + avg_historical) / 2
        else:
            estimated_time = base_time
        
        return estimated_time * (1 + size_factor * 0.1)
    
    def estimate_memory_usage(self, method: str, file_size: int) -> int:
        """Estimate GPU memory usage for a task"""
        # Base memory estimates in bytes
        base_memory = {
            "cnn_steg_detection": 512 * 1024 * 1024,  # 512MB
            "noiseprint": 1024 * 1024 * 1024,         # 1GB
            "deep_stego": 768 * 1024 * 1024,          # 768MB
            "anomaly_detection": 256 * 1024 * 1024,   # 256MB
            "multimodal_classification": 512 * 1024 * 1024,  # 512MB
            "llm_content_analysis": 2048 * 1024 * 1024       # 2GB
        }
        
        base_mem = base_memory.get(method, 512 * 1024 * 1024)
        
        # Adjust based on file size
        size_factor = file_size / (1024 * 1024)  # Size in MB
        additional_memory = int(size_factor * 1024 * 1024 * 0.1)  # 10% of file size
        
        return base_mem + additional_memory
    
    def can_run_task(self, method: str, file_size: int) -> Tuple[bool, str]:
        """Check if a task can run on available GPUs"""
        if not self.gpu_available:
            return False, "No GPUs available"
        
        required_memory = self.estimate_memory_usage(method, file_size)
        
        available_devices = [d for d in self.gpu_devices if d.available]
        if not available_devices:
            return False, "No available GPU devices"
        
        suitable_devices = [d for d in available_devices if d.memory_free >= required_memory]
        if not suitable_devices:
            return False, f"Insufficient GPU memory (need {required_memory // (1024*1024)}MB)"
        
        if len(self.active_tasks) >= self.max_concurrent_tasks:
            return False, "Maximum concurrent GPU tasks reached"
        
        return True, "GPU available"
    
    async def shutdown(self):
        """Shutdown GPU manager"""
        self.shutdown_event.set()
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        
        # Cancel active tasks
        for task_id in list(self.active_tasks.keys()):
            await self.release_gpu(task_id)
        
        self.logger.info("GPU manager shutdown completed")
