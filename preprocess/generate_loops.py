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
    parser.add_argument('-p', '--all_predict', type=bool, required=False, default=False, help='Use all data for prediction.')
    parser.add_argument('--positive_loops', nargs='*', default=[],
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
    if not args.all_predict:
        if len(args.positive_loops) <= 0 and len(args.negative_loops) <= 0:
            print('Do not get positive files and negative files!!! Program exit.')
            sys.exit(0)
    else:
        if len(args.positive_loops) <= 0:
            print('Do not get loop files!!! Program exit.')
            sys.exit(0)

    if not args.all_predict:
        dataset_names = ['train', 'valid', 'test']

        train_pairs, train_labels, train_kmers, val_pairs, val_labels, val_kmers, \
            test_pairs, test_labels, test_kmers = load_pairs(args.positive_loops, args.negative_loops, read_reference.hg19, all_for_prediction=False)
    else:
        train_pairs, train_labels, train_kmers, val_pairs, val_labels, val_kmers, \
            test_pairs, test_labels, test_kmers = load_pairs(args.positive_loops, None, read_reference.hg19, all_for_prediction=True)

    data_pairs = [train_pairs, val_pairs, test_pairs]
    data_labels = [train_labels, val_labels, test_labels]
    data_kmers = [train_kmers, val_kmers, test_kmers]

    if args.all_predict:
        print('Use all data for prediction.')
        pairs = train_pairs + val_pairs + test_pairs
        labels = train_labels + val_labels + test_labels
        kmers = train_kmers + val_kmers + test_kmers

        fn = os.path.join(args.out_dir, '{}_all_predict.hdf5'.format(name))
        print(fn)
        get_and_save_data(pairs, labels, kmers, fn, minimum_len, extend_len=args.extend_len)
    else:
        out_idxes = [0, 1, 2]
        print('Divide all data into training, testing, and validation sets.')
        print('Chrom 5, 14 are used for validation.')
        print('Chrom 4, 7, 8, 11 are used for testing.')
        print('The remaining chromosomes are used for training.')

        for idx in out_idxes:
            pairs = data_pairs[idx]
            labels = data_labels[idx]
            kmers = data_kmers[idx]
            dset = dataset_names[idx]
            fn = os.path.join(args.out_dir, "{}_{}.hdf5".format(name, dset))
            print(fn)
            get_and_save_data(pairs, labels, kmers, fn, minimum_len=1000, extend_len=args.extend_len)
