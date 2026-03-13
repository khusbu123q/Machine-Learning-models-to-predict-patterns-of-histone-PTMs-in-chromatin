#!/bin/bash

## ensure the script runs in bash
if [ -z "$BASH_VERSION" ]; then
    echo "This script requires bash. Please run it with bash."
    exit 1
fi

## input CSV file and BED file
input_file="/ibab/home/Downloads/MSC_Project/human_genes_GRCh38.csv"	##complete path to the human gene csv file
input_bed_directory="/ibab/home/Downloads/MSC_Project/h3k4me3_liver_files"	##complete path to PTM bed directory

## Output directory
output_dir="/ibab/home/Downloads/MSC_Project/gene_split"
mkdir -p "$output_dir"

## function to perform Bedtools intersect to subset main bed file gene wise
process_gene() {
    local gene_id="$1"
    local start_pos="$2"
    local end_pos="$3"
    local strand="$4"
    local chr_label="$5"

    ## Ensure all necessary values are non-empty
    if [[ -z "$gene_id" || -z "$start_pos" || -z "$end_pos" || -z "$strand" || -z "$chr_label" ]]; then
        echo "Skipping line due to missing values: $gene_id" >&2
        return
    fi

    ## Add "chr" prefix if not already present
    if [[ "$chr_label" != chr* ]]; then
        chr_no="chr${chr_label}"
    else
        chr_no="$chr_label"
    fi

    ## Convert strand values
    case "$strand" in
        -1) strand_symbol="-" ;;
        1 | +1) strand_symbol="+" ;;
        *)
            echo "Invalid strand value for gene ID $gene_id: $strand" >&2
            return
            ;;
    esac

    ## Prepare the query BED file
    query_bed="${gene_id}_query.bed"
    echo -e "${chr_no}\t${start_pos}\t${end_pos}\t${gene_id}\t0\t${strand_symbol}" > "$query_bed"

    ## Perform Bedtools intersect
    subset_bed="${output_dir}/${gene_id}_subset.bed"
    bedtools intersect -a "$input_bed" -b "$query_bed" -wa > "$subset_bed"

    ## Clean up the query BED file
    rm "$query_bed"
    echo "Processed: $gene_id"
}

export -f process_gene
export input_bed output_dir

## Run in parallel
tail -n +2 "$input_file" | awk -F',' '{print $1, $3, $4, $5, $9}' | parallel --colsep ' ' process_gene {1} {2} {3} {4} {5}

echo "Processing complete. Subset BED files are saved in $output_dir."
