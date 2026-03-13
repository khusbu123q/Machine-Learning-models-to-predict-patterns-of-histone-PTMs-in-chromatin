#!/bin/bash

##Input directories

GENE_SPLIT_DIR="/mnt/data/khushbu/gene_split"
OUTPUT_DIR="/mnt/data/khushbu/nucpos_results"

NUCPOS_BIN="$HOME/NucPosSimulator_linux64/NucPosSimulator"
PARAMS="$HOME/NucPosSimulator_linux64/params.txt"


TOTAL_CORES=$(nproc)

# use ~10–15% cores on shared server
MAX_JOBS=$((TOTAL_CORES / 10))

# threads per job
export OMP_NUM_THREADS=2

#Output Directory

mkdir -p "$OUTPUT_DIR"

echo "-----------------------------------"
echo "NucPos Parallel Pipeline"
echo "Detected cores: $TOTAL_CORES"
echo "Parallel jobs: $MAX_JOBS"
echo "Threads per job: $OMP_NUM_THREADS"
echo "-----------------------------------"


# CHECK NUC POS BINARY


if [ ! -x "$NUCPOS_BIN" ]; then
    echo "Making NucPos executable..."
    chmod +x "$NUCPOS_BIN"
fi


# FUNCTION TO RUN ONE GENE


run_nucpos() {

    bedfile=$1

    sample=$(basename "$(dirname "$bedfile")")
    gene=$(basename "$bedfile" .bed)

    outdir="$OUTPUT_DIR/$sample/$gene"
    mkdir -p "$outdir"

    cd "$outdir" || exit

    echo "Starting $gene at $(date)"

    "$NUCPOS_BIN" "$bedfile" "$PARAMS" > nucpos.log 2>&1

    echo "Finished $gene at $(date)"
}

export -f run_nucpos
export OUTPUT_DIR
export NUCPOS_BIN
export PARAMS


# RUN PIPELINE


find "$GENE_SPLIT_DIR" -type f -name "*.bed" -size +0 | \
while IFS= read -r bedfile
do

    run_nucpos "$bedfile" &

    while [ "$(jobs -r | wc -l)" -ge "$MAX_JOBS" ]
    do
        sleep 2
    done

done

wait

echo "ALL NUCLEOSOME SIMULATIONS FINISHED"
