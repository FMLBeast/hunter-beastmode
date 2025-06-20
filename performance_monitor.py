#!/usr/bin/env python3
"""
Real-time performance monitor for StegAnalyzer cascade analysis
"""

import time
import json
import psutil
import subprocess
from pathlib import Path
from datetime import datetime

class PerformanceMonitor:
    def __init__(self):
        self.log_file = Path("logs/performance_realtime.json")
        self.log_file.parent.mkdir(exist_ok=True)
        self.start_time = time.time()
        
    def get_gpu_stats(self):
        """Get GPU statistics if available"""
        try:
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                gpu_data = result.stdout.strip().split(',')
                return {
                    'gpu_utilization': float(gpu_data[0]),
                    'gpu_memory_used': float(gpu_data[1]),
                    'gpu_memory_total': float(gpu_data[2])
                }
        except:
            pass
        return None
    
    def get_cascade_stats(self):
        """Get cascade-specific statistics"""
        temp_dirs = list(Path("/tmp").glob("*cascading_analysis*"))
        total_temp_files = 0
        total_temp_size = 0
        
        for temp_dir in temp_dirs:
            try:
                for file in temp_dir.rglob("*"):
                    if file.is_file():
                        total_temp_files += 1
                        total_temp_size += file.stat().st_size
            except:
                continue
        
        return {
            'temp_directories': len(temp_dirs),
            'temp_files': total_temp_files,
            'temp_size_mb': total_temp_size / (1024 * 1024)
        }
    
    def monitor_loop(self):
        """Main monitoring loop"""
        print("📊 Performance monitoring started...")
        
        while True:
            try:
                # System stats
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('.')
                
                # Network I/O
                net_io = psutil.net_io_counters()
                
                # Process count
                python_processes = len([p for p in psutil.process_iter(['name']) 
                                       if p.info['name'] == 'python'])
                
                # GPU stats
                gpu_stats = self.get_gpu_stats()
                
                # Cascade-specific stats
                cascade_stats = self.get_cascade_stats()
                
                # Create stats object
                stats = {
                    'timestamp': datetime.now().isoformat(),
                    'uptime_seconds': time.time() - self.start_time,
                    'system': {
                        'cpu_percent': cpu_percent,
                        'memory_percent': memory.percent,
                        'memory_used_gb': memory.used / (1024**3),
                        'memory_available_gb': memory.available / (1024**3),
                        'disk_percent': disk.percent,
                        'disk_free_gb': disk.free / (1024**3)
                    },
                    'network': {
                        'bytes_sent': net_io.bytes_sent,
                        'bytes_recv': net_io.bytes_recv
                    },
                    'processes': {
                        'python_count': python_processes,
                        'total_count': len(list(psutil.process_iter()))
                    },
                    'cascade': cascade_stats
                }
                
                if gpu_stats:
                    stats['gpu'] = gpu_stats
                
                # Log to file
                with open(self.log_file, 'a') as f:
                    f.write(json.dumps(stats) + "\n")
                
                # Console output every 30 seconds
                if int(time.time()) % 30 == 0:
                    print(f"📊 CPU: {cpu_percent:5.1f}% | "
                          f"RAM: {memory.percent:5.1f}% | "
                          f"Temp files: {cascade_stats['temp_files']:,}")
                
                time.sleep(5)  # Monitor every 5 seconds
                
            except KeyboardInterrupt:
                print("\n📊 Performance monitoring stopped")
                break
            except Exception as e:
                print(f"⚠️  Monitor error: {e}")
                time.sleep(10)

if __name__ == "__main__":
    monitor = PerformanceMonitor()
    monitor.monitor_loop()
