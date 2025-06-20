#!/usr/bin/env python3
"""
Cascade Analysis Monitor
Monitors comprehensive cascade analysis progress
"""
import time
import json
import psutil
from pathlib import Path
from datetime import datetime

def monitor_cascade():
    """Monitor cascade analysis"""
    log_file = Path("logs/cascade_monitor.json")
    log_file.parent.mkdir(exist_ok=True)
    
    print("📊 Cascade analysis monitoring started...")
    
    while True:
        try:
            # System stats
            stats = {
                "timestamp": datetime.now().isoformat(),
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('.').percent
            }
            
            # Count analysis directories
            analysis_dirs = list(Path('.').glob('comprehensive_results_*/'))
            if analysis_dirs:
                latest_dir = max(analysis_dirs, key=lambda p: p.stat().st_mtime)
                completed_analyses = len(list(latest_dir.glob('*_analysis')))
                stats['completed_analyses'] = completed_analyses
            
            # Count temp zsteg directories
            temp_dirs = list(Path('/tmp').glob('*zsteg*'))
            stats['temp_zsteg_dirs'] = len(temp_dirs)
            
            # Log stats
            with open(log_file, 'a') as f:
                f.write(json.dumps(stats) + "\n")
            
            # Console output every 30 seconds
            if int(time.time()) % 30 == 0:
                completed = stats.get('completed_analyses', 0)
                print(f"📊 CPU: {stats['cpu_percent']:.1f}% | "
                      f"RAM: {stats['memory_percent']:.1f}% | "
                      f"Completed: {completed}")
            
            time.sleep(5)
            
        except KeyboardInterrupt:
            print("\n📊 Monitoring stopped")
            break
        except Exception as e:
            print(f"Monitor error: {e}")
            time.sleep(10)

if __name__ == "__main__":
    monitor_cascade()
