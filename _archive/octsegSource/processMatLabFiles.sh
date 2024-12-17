#!/bin/bash

# Function to process MATLAB files
process_files() {
    for file in "$1"/*.m; do
        if [ -f "$file" ]; then
            echo "Processing $file"
            smop "$file"
        fi
    done
}

# Export the function to be used by find
export -f process_files

# Find directories and process MATLAB files in each
find . -type d -exec bash -c 'process_files "$0"' {} \;