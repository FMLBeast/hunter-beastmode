#!/bin/bash
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
    
    python steg_main.py \
        --config "$CONFIG_FILE" \
        --cascade \
        --max-depth 20 \
        --max-files 50000 \
        --output "$OUTPUT_DIR/batch_$batch_id" \
        --format json \
        --compress \
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
find . -name "*.jpg" -o -name "*.png" -o -name "*.gif" -o -name "*.bmp" | \
    head -n 1000 | \
    xargs dirname | \
    sort -u | \
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
