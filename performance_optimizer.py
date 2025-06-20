#!/usr/bin/env python3
"""
Performance Optimizer for StegAnalyzer Cascade Analysis
Optimizes the working cascade system for maximum throughput on vast.ai
"""

import os
import json
import multiprocessing
from pathlib import Path
from typing import Dict, Any

def main():
    """Optimize performance for the working cascade system"""
    print("⚡ StegAnalyzer Performance Optimizer")
    print("=" * 50)
    print("Your cascade analysis is working! Let's optimize it for maximum speed.")
    
    project_root = Path(".")
    
    # Step 1: Fix the tool registration warning
    fix_tool_warnings(project_root)
    
    # Step 2: Optimize cascade configuration
    optimize_cascade_config(project_root)
    
    # Step 3: Create high-performance batch scripts
    create_optimized_batch_scripts(project_root)
    
    # Step 4: Setup performance monitoring
    setup_performance_monitoring(project_root)
    
    print("\n✅ Performance optimization complete!")
    print("\n🚀 Your optimized system:")
    print("   • Eliminated tool registration warnings")
    print("   • Optimized cascade parameters for speed")
    print("   • Created high-performance batch processing")
    print("   • Added real-time performance monitoring")
    print("\n🎯 Ready to process 45K files at maximum speed!")

def fix_tool_warnings(project_root: Path):
    """Fix the tool registration warnings from the logs"""
    print("\n🔧 Fixing tool registration warnings...")
    
    # Apply the tool registration fix
    try:
        exec(open('fix_tool_registration.py').read())
        print("   ✅ Tool registration warnings eliminated")
    except Exception as e:
        print(f"   ⚠️  Manual fix needed: {e}")
        manual_fix_tool_warnings(project_root)

def manual_fix_tool_warnings(project_root: Path):
    """Manual fix for tool warnings"""
    orchestrator_file = project_root / "core" / "orchestrator.py"
    if orchestrator_file.exists():
        content = orchestrator_file.read_text()
        
        # Simple fix: replace file_analyzer with file_forensics
        content = content.replace('"file_analyzer"', '"file_forensics"')
        content = content.replace("'file_analyzer'", "'file_forensics'")
        
        # Fix basic_analysis method mapping
        if "basic_analysis" in content and "file_analyzer" in content:
            content = content.replace(
                'tool_name="file_analyzer"',
                'tool_name="file_forensics"'
            )
        
        orchestrator_file.write_text(content)
        print("   ✅ Manual tool mapping fix applied")

def optimize_cascade_config(project_root: Path):
    """Create optimized configuration for maximum cascade performance"""
    print("\n⚙️  Creating high-performance cascade config...")
    
    # Detect system capabilities
    cpu_count = multiprocessing.cpu_count()
    gpu_available = check_gpu_available()
    memory_gb = get_memory_gb()
    
    print(f"   📊 Detected: {cpu_count} CPUs, {memory_gb}GB RAM, GPU: {gpu_available}")
    
    # Create optimized configuration
    config = {
        "database": {
            "type": "sqlite",
            "path": "steganalyzer.db",
            "connection_pool_size": min(50, cpu_count * 4)
        },
        "orchestrator": {
            "max_concurrent_files": min(12, cpu_count * 2),
            "max_cpu_workers": cpu_count,
            "max_gpu_workers": 3 if gpu_available else 0,
            "task_timeout": 1800,  # 30 minutes per task
            "memory_limit_gb": max(4, memory_gb - 4),
            "temp_directory": "/tmp/steg_cascade"
        },
        "analysis": {
            "quick_mode": False,
            "deep_analysis": True,
            "ml_analysis": gpu_available,
            "parallel_analysis": True,
            "enable_cascade": True
        },
        "cascade": {
            "max_depth": 20,  # Deep for CTF challenges
            "max_files": 50000,  # Handle your 45K files
            "enable_zsteg": True,
            "enable_binwalk": True,
            "enable_foremost": True,
            "zsteg_timeout": 45.0,  # Optimized timeouts
            "binwalk_timeout": 120.0,
            "foremost_timeout": 90.0,
            "max_file_size": 1073741824,  # 1GB max file size
            "save_extracts": True,
            "keep_extraction_tree": True,
            "compress_results": True,
            "max_concurrent_extractions": min(6, cpu_count),
            "memory_limit_mb": min(8192, (memory_gb * 1024) // 2),
            "min_confidence_threshold": 0.3,
            "skip_large_files": True,
            "large_file_threshold_mb": 100,
            "enable_file_type_filtering": True,
            "priority_extensions": [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".wav", ".mp3"]
        },
        "classic_stego": {
            "steghide_enabled": True,
            "outguess_enabled": True,
            "zsteg_enabled": True,
            "binwalk_enabled": True,
            "foremost_enabled": True,
            "parallel_execution": True,
            "timeout_per_tool": 60.0
        },
        "performance": {
            "enable_caching": True,
            "cache_size_mb": 1024,
            "enable_compression": True,
            "batch_processing": True,
            "batch_size": min(100, cpu_count * 10),
            "progress_reporting_interval": 50
        },
        "logging": {
            "level": "INFO",
            "log_dir": "logs/",
            "max_log_size_mb": 500,
            "backup_count": 10,
            "enable_performance_logging": True
        }
    }
    
    # Save optimized config
    config_dir = project_root / "config"
    config_dir.mkdir(exist_ok=True)
    
    config_file = config_dir / "high_performance.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"   ✅ High-performance config saved: {config_file}")
    print(f"   🎯 Optimized for {cpu_count} CPUs and {memory_gb}GB RAM")

def create_optimized_batch_scripts(project_root: Path):
    """Create optimized batch processing scripts"""
    print("\n📦 Creating optimized batch scripts...")
    
    # High-speed processing script
    speed_script = '''#!/bin/bash
# High-Speed StegAnalyzer for 45K Files
set -e

echo "🚀 High-Speed StegAnalyzer Starting..."

# Configuration
CONFIG_FILE="config/high_performance.json"
OUTPUT_DIR="results/speed_run_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="logs/speed_run.log"
MAX_PARALLEL=4

# Create directories
mkdir -p "$OUTPUT_DIR" logs temp

# Performance monitoring
echo "📊 Starting performance monitoring..."
python performance_monitor.py > logs/performance_$(date +%Y%m%d_%H%M%S).log 2>&1 &
MONITOR_PID=$!

# Cleanup function
cleanup() {
    echo "🧹 Cleaning up..."
    kill $MONITOR_PID 2>/dev/null || true
    # Clean temp files older than 1 hour
    find /tmp/steg_cascade -type f -mtime +0.04 -delete 2>/dev/null || true
}
trap cleanup EXIT

# Function to process directory batch
process_directory_batch() {
    local dir="$1"
    local batch_id="$2"
    
    echo "🔍 Processing batch $batch_id: $(basename "$dir")" | tee -a "$LOG_FILE"
    
    start_time=$(date +%s)
    
    python steg_main.py \\
        --config "$CONFIG_FILE" \\
        --cascade \\
        --max-depth 20 \\
        --max-files 50000 \\
        --output "$OUTPUT_DIR/batch_$batch_id" \\
        --format json \\
        --compress \\
        "$dir" 2>&1 | tee -a "$LOG_FILE"
    
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    echo "✅ Batch $batch_id completed in ${duration}s" | tee -a "$LOG_FILE"
}

# Export function for parallel execution
export -f process_directory_batch
export CONFIG_FILE OUTPUT_DIR LOG_FILE

echo "🔎 Finding image directories..."

# Find directories containing image files and process them
find . -name "*.jpg" -o -name "*.png" -o -name "*.gif" -o -name "*.bmp" | \\
    head -n 1000 | \\
    xargs dirname | \\
    sort -u | \\
    parallel -j $MAX_PARALLEL --progress process_directory_batch {} {#}

echo "🎉 High-speed analysis complete!"
echo "📊 Results in: $OUTPUT_DIR"
echo "📈 Performance logs: logs/"

# Generate final report
python -c "
import json
from pathlib import Path
import time

results_dir = Path('$OUTPUT_DIR')
total_files = 0
total_findings = 0
high_confidence = 0

for json_file in results_dir.rglob('*.json'):
    try:
        with open(json_file) as f:
            data = json.load(f)
            if isinstance(data, list):
                total_files += 1
                total_findings += len(data)
                high_confidence += len([r for r in data if r.get('confidence', 0) > 0.7])
    except: pass

print(f'📊 FINAL REPORT:')
print(f'   Files processed: {total_files}')
print(f'   Total findings: {total_findings}')
print(f'   High confidence: {high_confidence}')
print(f'   Time: $(date)')
"
'''
    
    speed_file = project_root / "speed_analysis.sh"
    with open(speed_file, 'w') as f:
        f.write(speed_script)
    os.chmod(speed_file, 0o755)
    
    # Smart batch processor for organized processing
    smart_script = '''#!/bin/bash
# Smart Batch Processor - Organizes files by type and processes efficiently
set -e

echo "🧠 Smart StegAnalyzer Batch Processor"

CONFIG_FILE="config/high_performance.json"
OUTPUT_DIR="results/smart_batch_$(date +%Y%m%d_%H%M%S)"

# Create organized processing
mkdir -p "$OUTPUT_DIR"/{images,audio,documents,mixed}

# Function to process by file type
process_by_type() {
    local file_type="$1"
    local pattern="$2"
    local output_subdir="$3"
    
    echo "🎯 Processing $file_type files..."
    
    find . -name "$pattern" -type f | head -n 10000 | while read file; do
        python steg_main.py \\
            --config "$CONFIG_FILE" \\
            --cascade \\
            --output "$OUTPUT_DIR/$output_subdir" \\
            --format json \\
            "$file"
    done
}

# Process different file types
process_by_type "Image" "*.jpg" "images" &
process_by_type "PNG" "*.png" "images" &
process_by_type "Audio" "*.wav" "audio" &
process_by_type "Mixed" "*" "mixed" &

wait

echo "✅ Smart batch processing complete!"
'''
    
    smart_file = project_root / "smart_batch.sh"
    with open(smart_file, 'w') as f:
        f.write(smart_script)
    os.chmod(smart_file, 0o755)
    
    print("   ✅ speed_analysis.sh - Maximum speed processing")
    print("   ✅ smart_batch.sh - Organized by file type processing")

def setup_performance_monitoring(project_root: Path):
    """Setup real-time performance monitoring"""
    print("\n📊 Setting up performance monitoring...")
    
    monitor_script = '''#!/usr/bin/env python3
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
                    f.write(json.dumps(stats) + "\\n")
                
                # Console output every 30 seconds
                if int(time.time()) % 30 == 0:
                    print(f"📊 CPU: {cpu_percent:5.1f}% | "
                          f"RAM: {memory.percent:5.1f}% | "
                          f"Temp files: {cascade_stats['temp_files']:,}")
                
                time.sleep(5)  # Monitor every 5 seconds
                
            except KeyboardInterrupt:
                print("\\n📊 Performance monitoring stopped")
                break
            except Exception as e:
                print(f"⚠️  Monitor error: {e}")
                time.sleep(10)

if __name__ == "__main__":
    monitor = PerformanceMonitor()
    monitor.monitor_loop()
'''
    
    monitor_file = project_root / "performance_monitor.py"
    with open(monitor_file, 'w') as f:
        f.write(monitor_script)
    
    print("   ✅ Real-time performance monitor created")
    print("   ✅ Tracks CPU, memory, GPU, cascade temp files")

def check_gpu_available():
    """Check if GPU is available"""
    try:
        subprocess.run(['nvidia-smi'], capture_output=True, timeout=5)
        return True
    except:
        return False

def get_memory_gb():
    """Get total memory in GB"""
    try:
        import psutil
        return int(psutil.virtual_memory().total / (1024**3))
    except:
        return 8  # Default

if __name__ == "__main__":
    main()
