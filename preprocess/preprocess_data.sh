#!/bin/bash

usage()
{
  echo "$(basename "$0") [-h] ChIAPet DNase TF Target Outdir"
  echo "-- Generate positve and negative loops."

  echo "where:"
  echo "-h            Display this help message"
  echo "ChIAPet       The path of ChIA-Pet file"
  echo "DNase         The path of DNase-seq file"
  echo "TF            The path of ChIP-seq file"
  echo "Target        The name of the target,like gm12878_rad21"
  echo "Outdir        The path of output directory"
}


if [ "$1" != "" ]; then
    case $1 in
        -h | --help )           usage
                                exit
                                ;;
    esac
fi

if [ $# -lt 5 ]; then
  usage
  exit
fi

ChIAPet=$1
DNase=$2
TF=$3
Target=$4
Outdir=$5

DIR=$(dirname "$0")

echo "Generate positive loops from ChIA-Pet"
cat $ChIAPet | awk '$1==$4 && ($3<$5 || $6<$2)' > ${Outdir}/${Target}_positive_loops.bedpe
positive_anchors=${Outdir}/${Target}_positive_anchors.bed
cat ${Outdir}/${Target}_positive_loops.bedpe \
    | awk 'BEGIN{FS=OFS="\t"}{printf("%s\t%s\t%s\n%s\t%s\t%s\n", $1, $2, $3, $4, $5, $6)}' \
    | sort -u > ${positive_anchors}

echo "Generate negative loops from DNase and TF"
echo "Generating random anchor pairs"
python preprocess/negative_anchors_generator.py $Target $Outdir
echo "Generating random TF peak pairs"
python preprocess/negative_pairs_generator.py $Target $TF tf ${Outdir}
echo "Generating random DNase pairs"
python preprocess/negative_pairs_generator.py $Target $DNase dnase ${Outdir}

echo "Delete overlap loops"
pairToPair -a ${Outdir}/${Target}_random_pairs_from_tf.bedpe -b $ChIAPet -type notboth  \
    | pairToPair -a stdin -b  ${Outdir}/${Target}_all_intra_negative_loops.bedpe   -type notboth \
    | pairToPair -a stdin -b  ${Outdir}/${Target}_exclusive_intra_negative_loops.bedpe  -type notboth \
    | uniq > ${Outdir}/${Target}_no_tf_negative_loops.bedpe
pairToPair -a ${Outdir}/${Target}_random_pairs_from_dnase.bedpe -b $ChIAPet -type notboth \
    | pairToPair -a stdin -b  ${Outdir}/${Target}_all_intra_negative_loops.bedpe -type notboth \
    | pairToPair -a stdin -b  ${Outdir}/${Target}_exclusive_intra_negative_loops.bedpe -type notboth \
    | uniq \
    | pairToPair -a stdin -b ${Outdir}/${Target}_no_tf_negative_loops.bedpe -type notboth \
    | uniq > ${Outdir}/${Target}_random_pairs_from_tf_and_dnase.bedpe

echo "Sample negative pairs"

python preprocess/sample_negative_pairs.py $Target ${Outdir}

echo "Generate train/test/valid loops hdf5"
python preprocess/generate_loops.py -m 1000 -e 500 -n ${Target}_loops \
-r data/hg19.fa -o ${Outdir} \
--positive_loops ${Outdir}/${Target}_positive_loops.bedpe \
--negative_loops ${Outdir}/${Target}_negative_loops.bedpe