#!/bin/bash

## Parameter settings
#chromosome='chr21'
#genome_version='hg19'
#fasta_file="E:\DeepLearning\enformer\data\hg19.fa"
#model_path='model'
#batch_size=8
#output_prefix='data/'

# Parse command-line arguments
chromosome=$1
genome_version=$2
fasta_file=$3
model_path=$4
batch_size=$5
output_prefix=$6

# Check if enough arguments are provided
if [ "$#" -ne 6 ]; then
    echo "Usage: $0 <chromosome> <genome_version> <fasta_file> <model_path> <batch_size> <output_prefix>"
    exit 1
fi

# Generate sequence.bed file
echo "Running genome_bed.py..."
python genome_bed.py $chromosome $genome_version

# Check if sequence.bed was generated successfully
if [ ! -f "sequence.bed" ]; then
  echo "Error: sequence.bed file was not generated."
  exit 1
fi

# Run model inference
echo "Running main.py..."
python main.py --bed_file sequence.bed --fasta_file $fasta_file --model_path $model_path --batch_size $batch_size --output_prefix $output_prefix

# Check if .bedGraph files were generated successfully
output_files=("${output_prefix}GM12878_CTCF.bedGraph" "${output_prefix}K562_YY1.bedGraph" "${output_prefix}K562_POLR2A.bedGraph" "${output_prefix}GM12878_RAD21.bedGraph")
for file in "${output_files[@]}"; do
  if [ ! -f "$file" ]; then
    echo "Error: $file was not generated."
    exit 1
  fi
done

# Perform peak calling on the four .bedGraph files
echo "Running MACS3 peak calling..."
macs3 bdgpeakcall -i ${output_prefix}GM12878_CTCF.bedGraph -o ${output_prefix}GM12878_CTCF.narrowPeak
macs3 bdgpeakcall -i ${output_prefix}K562_YY1.bedGraph -o ${output_prefix}K562_YY1.narrowPeak
macs3 bdgpeakcall -i ${output_prefix}K562_POLR2A.bedGraph -o ${output_prefix}K562_POLR2A.narrowPeak
macs3 bdgpeakcall -i ${output_prefix}GM12878_RAD21.bedGraph -o ${output_prefix}GM12878_RAD21.narrowPeak

echo "Pipeline completed successfully."
