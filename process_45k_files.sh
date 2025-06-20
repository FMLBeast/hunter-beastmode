#!/bin/bash
# Process 45K Files with Comprehensive Cascade Analysis
set -e

echo "🚀 Processing 45K files with comprehensive cascade analysis..."

CONFIG_FILE="config/vast_ai_comprehensive.json"
OUTPUT_BASE="comprehensive_results_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="logs/comprehensive_batch.log"

# Create directories
mkdir -p "$OUTPUT_BASE" logs

# Start monitoring
python cascade_monitor.py > logs/monitor_$(date +%Y%m%d_%H%M%S).log 2>&1 &
MONITOR_PID=$!

# Cleanup function
cleanup() {
    echo "🧹 Cleaning up..."
    kill $MONITOR_PID 2>/dev/null || true
}
trap cleanup EXIT

# Function to process file with comprehensive analysis
process_file_comprehensive() {
    local file="$1"
    local output_dir="$2"
    
    echo "🔍 Comprehensive analysis: $(basename "$file")"
    
    python comprehensive_analysis.py \
        "$file" \
        --output-dir "$output_dir/$(basename "$file" | sed 's/\.[^.]*$//')_analysis"
}

# Export function
export -f process_file_comprehensive

# Find image files and process them
echo "🔎 Finding candidate files..."

find . -type f \( \
    -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" -o \
    -name "*.gif" -o -name "*.bmp" -o -name "*.tiff" \
\) | head -n 45000 | \
    parallel -j 2 --progress process_file_comprehensive {} "$OUTPUT_BASE"

echo "🎉 45K file processing complete!"
echo "📊 Results in: $OUTPUT_BASE"

# Generate summary
python -c "
import json
from pathlib import Path
import time

results_dir = Path('$OUTPUT_BASE')
total_analyses = len(list(results_dir.glob('*_analysis')))
total_findings = 0

for analysis_dir in results_dir.glob('*_analysis'):
    report_file = analysis_dir / 'comprehensive_analysis_report.json'
    if report_file.exists():
        try:
            with open(report_file) as f:
                data = json.load(f)
                total_findings += data.get('summary', {}).get('high_confidence_findings', 0)
        except: pass

print(f'📊 BATCH PROCESSING SUMMARY:')
print(f'   Files analyzed: {total_analyses}')
print(f'   High confidence findings: {total_findings}')
print(f'   Results directory: $OUTPUT_BASE')
"
