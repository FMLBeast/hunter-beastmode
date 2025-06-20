#!/bin/bash
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
        python steg_main.py \
            --config "$CONFIG_FILE" \
            --cascade \
            --output "$OUTPUT_DIR/$output_subdir" \
            --format json \
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
