import argparse


def get_chromosome_length(chromosome, genome_version='hg19'):
    # hg19 and hg38 chromosome lengths in base pairs (for reference, you can expand as needed)
    hg19_lengths = {
        'chr1': 249250621, 'chr2': 243199373, 'chr3': 198022430, 'chr4': 191154276,
        'chr5': 180915260, 'chr6': 171115067, 'chr7': 159138663, 'chr8': 146364022,
        'chr9': 141213431, 'chr10': 135534747, 'chr11': 135006516, 'chr12': 133851895,
        'chr13': 115169878, 'chr14': 107349540, 'chr15': 102531392, 'chr16': 90354753,
        'chr17': 81195210, 'chr18': 78077248, 'chr19': 59128983, 'chr20': 63025520,
        'chr21': 48129895, 'chr22': 51304566, 'chrX': 155270560, 'chrY': 59373566
    }

    hg38_lengths = {
        'chr1': 248956422, 'chr2': 242193529, 'chr3': 198295559, 'chr4': 190214555,
        'chr5': 181538259, 'chr6': 170805979, 'chr7': 159345973, 'chr8': 145138636,
        'chr9': 138394717, 'chr10': 133797422, 'chr11': 135086622, 'chr12': 133275309,
        'chr13': 114364328, 'chr14': 107043718, 'chr15': 101991189, 'chr16': 90338345,
        'chr17': 83257441, 'chr18': 80373285, 'chr19': 58617616, 'chr20': 64444167,
        'chr21': 46709983, 'chr22': 50818468, 'chrX': 156040895, 'chrY': 57227415
    }

    if genome_version == 'hg19':
        return hg19_lengths.get(chromosome, None)
    elif genome_version == 'hg38':
        return hg38_lengths.get(chromosome, None)
    else:
        raise ValueError("Unsupported genome version. Use 'hg19' or 'hg38'.")


def split_chromosome(chromosome, genome_version='hg19', step=114688):
    chromosome_length = get_chromosome_length(chromosome, genome_version)
    if chromosome_length is None:
        raise ValueError(f"Chromosome {chromosome} not found in genome version {genome_version}.")

    current_start = 0
    segments = []

    while current_start < chromosome_length:
        current_end = min(current_start + step, chromosome_length)
        segments.append((chromosome, current_start, current_end))
        current_start = current_end

    return segments


def write_to_bedfile(segments, filename="sequence.bed"):
    with open(filename, "w") as bed_file:
        for segment in segments:
            bed_file.write(f"{segment[0]}\t{segment[1]}\t{segment[2]}\n")


def main():
    parser = argparse.ArgumentParser(description="Split chromosome into segments and save to a BED file.")
    parser.add_argument("chromosome", type=str, help="Chromosome name (e.g., chr1, chr2, ...)")
    parser.add_argument("genome_version", type=str, choices=["hg19", "hg38"], help="Genome version (hg19 or hg38)")

    args = parser.parse_args()

    # Generate segments
    segments = split_chromosome(args.chromosome, args.genome_version)

    # Write to BED file
    write_to_bedfile(segments, filename="sequence.bed")


if __name__ == "__main__":
    main()
