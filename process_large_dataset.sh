#!/bin/bash
# Process large datasets with the fixed system
echo "🚀 Processing large dataset with 61 CPU optimization..."

CONFIG_FILE="config/vast_ai_optimized.json"
OUTPUT_DIR="massive_analysis_$(date +%Y%m%d_%H%M%S)"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Function to process files in parallel
process_batch() {
    local file="$1"
    echo "🔍 Processing: $(basename "$file")"
    
    python3 steg_main.py \
        --config "$CONFIG_FILE" \
        --cascade \
        --verbose \
        --output "$OUTPUT_DIR" \
        "$file"
}

# Export function for parallel execution
export -f process_batch

# Find image files and process them in parallel
find . -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" -o -name "*.gif" -o -name "*.bmp" \) \
    | head -n 50000 \
    | parallel -j 20 process_batch

echo "🎉 Large dataset processing complete!"
echo "📊 Results in: $OUTPUT_DIR"
