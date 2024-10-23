#!/bin/bash

# Define input files and directories
READS_DIR="path/to/reads"  # Directory containing the raw reads
OUTPUT_DIR="path/to/output"  # Directory to store the output
FASTQC_DIR="path/to/fastqc"  # Directory for FastQC output
SPADES_DIR="path/to/spades_output"  # Directory for SPAdes output
VELVET_DIR="path/to/velvet_output"  # Directory for Velvet output
MAC_DIR="path/to/mac_output"  # Directory for MAC tool output

# User-specified installation paths for tools
SPADES_PATH="path/to/spades_installation"  # Path to SPAdes installation
VELVET_PATH="path/to/velvet_installation"  # Path to Velvet installation
MAC_TOOL_PATH="path/to/mac_installation"  # Path to MAC tool installation
FASTQC_PATH="path/to/fastqc_installation"  # Path to FastQC installation

# Step 1: Perform quality control on raw reads using FastQC
mkdir -p $FASTQC_DIR
echo "Running FastQC for quality control..."
$FASTQC_PATH/fastqc $READS_DIR/*.fastq -o $FASTQC_DIR

# Step 2: De novo assembly using SPAdes with key parameters
mkdir -p $SPADES_DIR
echo "Running SPAdes for de novo assembly..."
$SPADES_PATH/spades.py --careful -k 21,33,55,77,99,127 -1 $READS_DIR/reads_R1.fastq -2 $READS_DIR/reads_R2.fastq -o $SPADES_DIR

# Step 3: Check if target sequence length (196,608 bp) is achieved
# Extract contigs from SPAdes assembly
CONTIG_FILE="$SPADES_DIR/contigs.fasta"
TARGET_LENGTH=196608
ACTUAL_LENGTH=$(grep -v ">" $CONTIG_FILE | tr -d '\n' | wc -c)

if [ $ACTUAL_LENGTH -ge $TARGET_LENGTH ]; then
  echo "SPAdes assembly successful. Sequence length: $ACTUAL_LENGTH bp."
else
  echo "SPAdes assembly did not reach the target length. Proceeding with Velvet assembly..."

  # Step 4: Apply Velvet for further assembly if necessary
  mkdir -p $VELVET_DIR
  $VELVET_PATH/velveth $VELVET_DIR 31 -fastq -shortPaired $READS_DIR/reads_R1.fastq $READS_DIR/reads_R2.fastq
  $VELVET_PATH/velvetg $VELVET_DIR -exp_cov auto -cov_cutoff auto

  # Check if Velvet assembly achieves the target length
  CONTIG_FILE="$VELVET_DIR/contigs.fa"
  ACTUAL_LENGTH=$(grep -v ">" $CONTIG_FILE | tr -d '\n' | wc -c)

  if [ $ACTUAL_LENGTH -ge $TARGET_LENGTH ]; then
    echo "Velvet assembly successful. Sequence length: $ACTUAL_LENGTH bp."
  else
    echo "Velvet assembly also did not reach target length. Proceeding with MAC tool for further assembly..."

    # Step 5: Use MAC tool to merge contigs and reach 196,608 bp
    mkdir -p $MAC_DIR
    $MAC_TOOL_PATH/MAC2.0 --input $VELVET_DIR/contigs.fa --output $MAC_DIR/merged_sequence.fasta

    # Check final assembled sequence length
    CONTIG_FILE="$MAC_DIR/merged_sequence.fasta"
    ACTUAL_LENGTH=$(grep -v ">" $CONTIG_FILE | tr -d '\n' | wc -c)

    if [ $ACTUAL_LENGTH -ge $TARGET_LENGTH ]; then
      echo "MAC tool assembly successful. Sequence length: $ACTUAL_LENGTH bp."
    else
      echo "Failed to achieve the target sequence length. Final length: $ACTUAL_LENGTH bp."
    fi
  fi
fi

echo "Assembly process completed."
