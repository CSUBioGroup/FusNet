import os
import sys
import argparse
from generate_data_tools import get_and_save_data, load_pairs
import read_reference
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def get_args():
    parser = argparse.ArgumentParser(description='Generate loops files for train/valid/test (hdf5 format).')
    parser.add_argument('-m', '--minimum_len', type=int, required=True, help='Minimum length of anchors.')
    parser.add_argument('-e', '--extend_len', type=int, required=False, help='Extension length of anchors.')
    parser.add_argument('-n', '--name', type=str, required=True, help='The name of the target.')
    parser.add_argument('-r', '--reference_genome', required=True,
                        help='The path of reference sequence data. e.g. hg19.fa')
    parser.add_argument('-o', '--out_dir', type=str, required=True, help='The output directory.')
    parser.add_argument('-p', '--all_predict', type=bool, required=False, default=True, help='Use all data for prediction.')
    parser.add_argument('--loop_files', nargs='*', default=[],
                        help='The positive loop files. e.g. gm12878_rad21_positive_loops.bedpe, allow input of multiple files')
    parser.add_argument('--negative_loops', nargs='*', required=False, default=[],
                        help='The negative loop files. e.g. gm12878_rad21_negative_loops.bedpe, allow input of multiple files')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    print('Minimum length of anchors:', args.minimum_len)
    print('The name of the target:', args.name)
    print('The reference genome:', args.reference_genome)
    print('Whether to use all data for prediction:', args.all_predict)

    name = args.name
    minimum_len = args.minimum_len
    read_reference.init_ref(args.reference_genome)

    if len(args.loop_files) <= 0 :
        print('Do not get files!!! Program exit.')
        sys.exit(0)


    train_pairs, train_labels, train_kmers, val_pairs, val_labels, val_kmers, \
        test_pairs, test_labels, test_kmers = load_pairs(args.loop_files, None, read_reference.hg19, all_for_prediction=True)

    data_pairs = [train_pairs, val_pairs, test_pairs]
    data_labels = [train_labels, val_labels, test_labels]
    data_kmers = [train_kmers, val_kmers, test_kmers]

    print('Use all data for prediction.')
    pairs = train_pairs + val_pairs + test_pairs
    labels = train_labels + val_labels + test_labels
    kmers = train_kmers + val_kmers + test_kmers

    fn = os.path.join(args.out_dir, '{}_all_predict.hdf5'.format(name))
    print(fn)
    get_and_save_data(pairs, labels, kmers, fn, minimum_len, extend_len=args.extend_len, all_predict=True)

