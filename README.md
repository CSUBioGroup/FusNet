<img src="https://github.com/CSUBioGroup/FusNet/blob/main/Figure/FusNet_logo.png" width=250px>

Deciphering protein-mediated chromatin loops to explore disease mechanisms in 3D Genome architecture

## Contents
- [Installation](#Installation)
- [Long Sequence Generation (LSG)](#Long-Sequence-Generation)
- [Anchor Predictor](#Anchor-Predictor)
- [Loop Predictor](#Loop-Predictor)
- [Example usage](#Example-usage)

## Installation

**1. Install FusNet**
```
git clone https://github.com/bioinfomaticsCSU/FusNet.git
# Using FusNet as the root directory for program execution
cd FusNet
```

**2. Create an environment**
```
# create a new enviroment
conda env create -f environment.yml --name fusnet
# activate
conda activate fusnet
```

**3. Create directory**
```
mkdir logs
mkdir out_dir
```
## Long Sequence Generation
FusNet incorporates a Long Sequence Generation (LSG) module to efficiently process shorter sequencing reads.

**1. Pre-requries**

- SPAdes Genome Assembler (https://github.com/ablab/spades)
- Velvet Assembler (https://github.com/dzerbino/velvet)
- MAC merging assembly tool (https://github.com/bioinfomaticsCSU/MAC)

**2. Run the LSG script**

Please modify the LSG script (see LSG.sh) to allow specification of the directories for the required tools, as well as the input and output directories. The workflow of LSG is shown in the figure below:

<img src="https://github.com/CSUBioGroup/FusNet/blob/main/Figure/LSG.png" width=700px>

## Anchor-Predictor
First, download the model files `pytorch_model.bin` and `config.json`, and ensure they are organized in the following structure:
```
model
├── config.json
└── pytorch_model.bin
```
You can generate the anchors by either running the following .sh script:
```
bash anchor_prediction.sh chr1 hg19 /path/to/hg19.fa /path/to/model 8 anchor/
```
Or, you can follow these steps to generate the anchors:
#### Step 1: Generate the .bed file for the anchor of the desired chromosome.
```
python genome_bed.py chromosome=chr1 genome_version=hg19
```
The `sequence.bed` file will then be generated in the current folder.
#### Step 2: Generate bedGraph files through the model.
```
python profiling.py \
--bed_file sequence.bed \
--fasta_file path/to/hg19.fa \
--model_path path/to/model \
--batch_size 8 \
--output_prefix anchor/
```
#### Step 3: Generate anchor through MACS
```
macs3 bdgpeakcall -i anchor/GM12878_CTCF.bedGraph -o anchor/GM12878_CTCF.narrowPeak
macs3 bdgpeakcall -i anchor/K562_YY1.bedGraph -o anchor/K562_YY1.narrowPeak
macs3 bdgpeakcall -i anchor/K562_POLR2A.bedGraph -o anchor/K562_POLR2A.narrowPeak
macs3 bdgpeakcall -i anchor/GM12878_RAD21.bedGraph -o anchor/GM12878_RAD21.narrowPeak
```

## FusNet

**Sample data**

We provided 4 sample data in the data folder, each containing DNase, ChIP seq, and ChIA PET data for generating positive and negative sample loops.

**1. Data preprocessing**
```
bash preprocess/preprocess_data.sh \
data/gm12878_rad21/gm12878_rad21_interactions_hg19.bedpe \
data/gm12878_rad21/gm12878_DNase_hg19.narrowPeak \
data/gm12878_rad21/gm12878_rad21_TF_hg19.narrowPeak gm12878_rad21 out_dir
```
After the program runs successfully, the following files will be generated
```
gm12878_rad21_all_intra_negative_loops.bedpe
gm12878_rad21_exclusive_intra_negative_loops.bedpe
gm12878_rad21_loops_test.hdf5
gm12878_rad21_loops_train.hdf5
gm12878_rad21_loops_valid.hdf5
gm12878_rad21_no_tf_negative_loops.bedpe
gm12878_rad21_positive_anchors.bed
gm12878_rad21_random_pairs_from_dnase.bedpe
gm12878_rad21_random_pairs_from_tf.bedpe
gm12878_rad21_random_pairs_from_tf_and_dnase.bedpe
gm12878_rad21_negative_loops.bedpe
gm12878_rad21_positive_loops.bedpe
gm12878_rad21_loops_train.hdf5
gm12878_rad21_loops_test.hdf5
gm12878_rad21_loops_valid.hdf5
```

**2. Training sequence feature extractor**
```
python fusnet/train_feature_extractor.py out_dir/gm12878_rad21_loops \
gm12878_rad21_extractor out_dir
```
After the program runs successfully, the following files will be generated
```
gm12878_rad21_extractor.model.pt
gm12878_rad21_extractor.classifier.pt
```

**3. Feature extracting**
```
python fusnet/extract_feature.py \
                  out_dir/gm12878_rad21_extractor.model.pt \
                  out_dir/gm12878_rad21_loops \
                  gm12878_rad21_extracted \
                  out_dir;
```
After the program runs successfully, the following files will be generated
```
out_dir/gm12878_rad21_extracted_train_factor_outputs.hdf5   
out_dir/gm12878_rad21_extracted_valid_factor_outputs.hdf5
out_dir/gm12878_rad21_extracted_test_factor_outputs.hdf5
```

**4. Training fusion model classifiers**
```
python fusnet/train_fusion_model.py out_dir gm12878_rad21
```
After the program runs successfully, the following files will be generated
```
out_dir/gm12878_rad21_extracted_knn_predictor.pkl 
out_dir/gm12878_rad21_extracted_lgb_predictor.pkl 
out_dir/gm12878_rad21_extracted_xgb_predictor.pkl
out_dir/gm12878_rad21_extracted_lr_predictor.pkl 
```

**5. Loops predicting**
```
python fusnet/predict.py -m out_dir/gm12878_rad21_extractor.model.pt \
                --data_name gm12878_rad21 \
                --data_file out_dir/gm12878_rad21_extracted_test_factor_outputs.hdf5 \
                --output_pre out_dir/gm12878_rad21_extracted_test
```
After the program runs successfully, the following files will be generated
```
out_dir/gm12878_rad21_extracted_test_FusNet_probs.txt
```
