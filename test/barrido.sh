#!/bin/bash

# -----------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------
# CHANGE THIS to the actual name of your C source file
SOURCE_FILE="test_gemm.c" 

# Directory to store results
RESULT_DIR="resultsBarridoSimple3Dim"

# -----------------------------------------------------------
# PRE-CHECKS
# -----------------------------------------------------------
if [ "$#" -eq 0 ]; then
    echo "Usage: $0 size1 size2 size3 ..."
    echo "Example: $0 2880 3220 4000"
    exit 1
fi

if [ ! -f "$SOURCE_FILE" ]; then
    echo "Error: Source file '$SOURCE_FILE' not found."
    echo "Please edit the SOURCE_FILE variable inside this script."
    exit 1
fi

mkdir -p "$RESULT_DIR"

echo "Starting Barrido execution..."
echo "Results will be saved in '$RESULT_DIR'"
echo "--------------------------------------------------"

# -----------------------------------------------------------
# BACKUP & CLEANUP STRATEGY
# -----------------------------------------------------------
# Create a backup of the original C file
cp "$SOURCE_FILE" "${SOURCE_FILE}.bak"

# Function to restore the file when the script ends or is interrupted
cleanup() {
    if [ -f "${SOURCE_FILE}.bak" ]; then
        mv "${SOURCE_FILE}.bak" "$SOURCE_FILE"
        echo ""
        echo "Original source file restored."
    fi
    # Remove temp file if it exists
    rm -f "${SOURCE_FILE}.tmp"
    echo "Done."
}

# Trap EXIT (normal finish) and INT (Ctrl+C)
trap cleanup EXIT INT

# -----------------------------------------------------------
# MAIN LOOP
# -----------------------------------------------------------
for SIZE in "$@"; do
    echo "Processing Matrix Size: $SIZE"

    # 1. MODIFY THE C FILE
    # We use a temp file instead of -i to avoid macOS/Linux syntax errors.
    # We also use [[:space:]] instead of \s for better compatibility.
    
    sed "s/p_begin[[:space:]]*=[[:space:]]*[0-9]*;/p_begin = $SIZE;/g" "$SOURCE_FILE" > "${SOURCE_FILE}.tmp" && mv "${SOURCE_FILE}.tmp" "$SOURCE_FILE"
    
    sed "s/p_end[[:space:]]*=[[:space:]]*[0-9]*;/p_end   = $SIZE;/g" "$SOURCE_FILE" > "${SOURCE_FILE}.tmp" && mv "${SOURCE_FILE}.tmp" "$SOURCE_FILE"

    # 2. COMPILE
    echo "  -> Compiling (make clean && make blis)..."
    make clean > /dev/null 2>&1
    make blis > /dev/null 2>&1

    if [ $? -ne 0 ]; then
        echo "  !! ERROR: Compilation failed for size $SIZE. Skipping."
        continue
    fi

    # 3. IDENTIFY EXECUTABLE
    # Find the most recently created *_blis.x file
    EXE=$(ls -t *_blis.x 2>/dev/null | head -n 1)

    if [ -z "$EXE" ]; then
        echo "  !! ERROR: No *_blis.x executable found after compilation."
        continue
    fi

    # 4. EXECUTE
    OUTPUT_FILE="${RESULT_DIR}/test_barrido_${SIZE}.m"
    echo "  -> Running ./$EXE"
    echo "  -> Saving to $OUTPUT_FILE"

    VECLIB_MAXIMUM_THREADS=1 ./"$EXE" > "$OUTPUT_FILE" 2>&1

    if [ $? -ne 0 ]; then
        echo "  !! WARNING: Execution exited with error code."
    fi
    echo "--------------------------------------------------"
done
