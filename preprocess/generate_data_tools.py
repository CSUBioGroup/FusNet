from sklearn.utils import shuffle
import numpy as np
import h5py
import pandas as pd
from functools import partial
from pair_features import encode_seq, _get_sequence

def validate_chrom_name(chrom):
    import re
    return re.match('^chr(\d+|X)$', chrom)

def extract_chrom_num(chrom):
    if chrom == 'chrX':
        chrom = 23
    else:
        chrom = int(chrom.replace('chr', ''))
    return chrom

def check_peaks(peaks, chrom, start, end):
    if chrom not in peaks:
        return False
    for p in peaks[chrom]:
        if min(end, p[2]) - max(start, p[1]) > 0:
            return True
    return False


def check_all_peaks(peaks_list, chrom, start, end):
    return [1 if check_peaks(peaks, chrom,start,end) else 0 for peaks in peaks_list]


def initialize_kmer_df(k):
    kmers = []
    for i in range(4 ** k):
        kmer = ''
        for j in range(k):
            kmer += 'ACGT'[i // (4 ** (k - j - 1)) % 4]
        kmers.append(kmer)
    df = pd.DataFrame({
        'kmer': kmers,
        'count': [0] * len(kmers)
    })

    return df


def count_kmers_in_df(df, sequence, k):
    kmer_counts = {}
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i + k]
        if kmer in kmer_counts:
            kmer_counts[kmer] += 1
        else:
            kmer_counts[kmer] = 1
    for i in range(len(df)):
        kmer = df.loc[i, 'kmer']
        count = kmer_counts.get(kmer, 0)
        df.loc[i, 'count'] += count


def _load_data(fn, hg19, label,
               train_pairs, train_labels, train_kmers, val_pairs, val_labels,val_kmers,
               test_pairs, test_labels, test_kmers, peaks_list, allow_inter=False, breakpoints={}):
    int_cols = [1, 2, 4, 5]
    chrom_cols = [0, 3]
    val_chroms = [5, 14]
    test_chroms = [4, 11, 7, 8]
    with open(fn) as f:
        for r in f:
            tokens = r.strip().split()
            if not validate_chrom_name(tokens[0]):
                continue

            for i in chrom_cols:
                tokens[i] = extract_chrom_num(tokens[i])

            for i in int_cols:
                tokens[i] = int(tokens[i])

            if tokens[0] >= len(hg19) or tokens[3] >= len(hg19):
                continue

            if not allow_inter and tokens[0] != tokens[3]:
                print('skipping different chrom ', tokens)
                continue
            elif allow_inter and tokens[0] != tokens[3]:
                if not (tokens[0] in breakpoints and tokens[3] in breakpoints):
                    continue
                temp_dl = 0.5 * (tokens[1] + tokens[2]) - breakpoints[tokens[0]]
                temp_dr = 0.5 * (tokens[4] + tokens[5]) - breakpoints[tokens[3]]
                if temp_dl * temp_dr > 0 or not (5000<=abs(temp_dl) + abs(temp_dr)<=2000000):
                    print('distance issues for different chromosome')
                    continue
                if tokens[0] > tokens[3]:
                    temp = tokens[3:6]
                    tokens[3:6] = tokens[:3]
                    tokens[:3] = temp

            if tokens[1] > tokens[4]:
                temp1,temp2,temp3 = tokens[3:6]
                tokens[3], tokens[4], tokens[5] = tokens[0], tokens[1], tokens[2]
                tokens[0], tokens[1], tokens[2] = temp1, temp2, temp3

            if (tokens[1] < 0 or tokens[4] < 0 or
                tokens[2] >= len(hg19[tokens[0]]) or
                tokens[5] > len(hg19[tokens[3]]) or
                (tokens[0] == tokens[3] and tokens[4] < tokens[2])):
                print('skipping', tokens)
                continue

            if (tokens[0] != tokens[3]) or (tokens[0] == tokens[3]
                      and 5000. <= 0.5 * (tokens[4] - tokens[1] + tokens[5] - tokens[2]) <= 2000000):
                if len(tokens) < 7:
                    tokens.append(label)
                else:
                    tokens[6] = int(float(tokens[6]))
                if peaks_list is not None:
                    temp_peaks = check_all_peaks(peaks_list, *tokens[:3]) + check_all_peaks(peaks_list, *tokens[3:6])
                    tokens += temp_peaks

                tokens = tuple(tokens)

                left_seq = _get_sequence(tokens[0], tokens[1], tokens[2], min_size=1000, crispred=None)
                right_seq = _get_sequence(tokens[3], tokens[4], tokens[5], min_size=1000, crispred=None)

                k = 3
                left_df = initialize_kmer_df(k)
                right_df = initialize_kmer_df(k)
                count_kmers_in_df(left_df, left_seq, k)
                count_kmers_in_df(right_df, right_seq, k)
                left_count_list = left_df['count'].tolist()
                right_count_list = right_df['count'].tolist()

                count_list = left_count_list + right_count_list

                if tokens[0] in val_chroms:
                    val_pairs.append(tokens)
                    val_labels.append(label)
                    val_kmers.append(count_list)
                elif tokens[0] in test_chroms:
                    test_pairs.append(tokens)
                    test_labels.append(label)
                    test_kmers.append(count_list)
                else:
                    train_pairs.append(tokens)
                    train_labels.append(label)
                    train_kmers.append(count_list)




def load_pairs(pos_files, neg_files, hg19, peaks_list=None, allow_inter=False, breakpoints={}, all_for_prediction=False):
    train_pairs = []
    train_kmers = []
    train_labels = []
    val_pairs = []
    val_kmers = []
    val_labels = []
    test_pairs = []
    test_kmers = []
    test_labels = []

    for fn in pos_files:
        if not all_for_prediction:
            print('Loading positive files...')
        else:
            print('Loading loop files...')
        _load_data(fn, hg19, 1,
                   train_pairs, train_labels,train_kmers,
                   val_pairs, val_labels, val_kmers,
                   test_pairs, test_labels, test_kmers, peaks_list, allow_inter, breakpoints)
    if not all_for_prediction:
        for fn in neg_files:
            print('Loading negative files...')
            _load_data(fn, hg19, 0,
                       train_pairs, train_labels, train_kmers,
                       val_pairs, val_labels, val_kmers,
                       test_pairs, test_labels, test_kmers, peaks_list, allow_inter, breakpoints)
    # 打乱顺序，但pairs和labels还是对应的
    train_pairs, train_labels, train_kmers = shuffle(train_pairs, train_labels, train_kmers)
    val_pairs, val_labels, val_kmers = shuffle(val_pairs, val_labels, val_kmers)
    test_pairs, test_labels, test_kmers = shuffle(test_pairs, test_labels, test_kmers)
    return train_pairs, train_labels, train_kmers, val_pairs, val_labels, val_kmers, test_pairs, test_labels, test_kmers


def __get_mat(p, left, min_size, ext_size, crispred=None):
    if left:
        chrom, start, end = (0, 1, 2)
    else:
        chrom, start, end = (3, 4, 5)
    curr_chrom = p[chrom]
    
    if ext_size is not None:
        min_size = p[end]-p[start] + 2*ext_size
    temp = encode_seq(curr_chrom, p[start], p[end], min_size=min_size, crispred=crispred)
    if temp is None:
        raise ValueError('Nong value for matrix')
    return temp
    

def get_one_side_data_parallel(pairs, pool, left=True, out=None, verbose=False,
                               min_size=1000, ext_size=None, crispred=None):
    edges = [0]
    data = pool.map(partial(__get_mat, left=left, min_size=min_size, ext_size=ext_size, crispred=crispred), pairs)
    for d in data:
        edges.append(d.shape[0] + edges[-1])

    return np.concatenate(data, axis=0), edges


def get_one_side_data(pairs, left=True, out=None, verbose=False, min_size=1000, ext_size=None, crispred=None):

    if out is not None:

        data_name = "left_data" if left else "right_data"
        data_store = out.create_dataset(data_name, (50000, 4, 1000), dtype='uint8', maxshape=(None, 4, 1000),
                                        chunks=True, compression='gzip')

    if left:
        chrom, start, end = (0, 1, 2)
    else:
        chrom, start, end = (3, 4, 5)
    edges = [0]
    data = []
    last_cut = 0

    for p in pairs:
        curr_chrom = p[chrom]
        if type(curr_chrom) == int:
            if curr_chrom == 23:
                curr_chrom = 'chrX'
            else:
                curr_chrom = 'chr%d' % curr_chrom
        if ext_size is not None:

            min_size = p[end]-p[start] + 2*ext_size
        temp = encode_seq(p[chrom], p[start], p[end], min_size=min_size, crispred=crispred)
        if temp is None:
            raise ValueError('Nong value for matrix')
        new_cut = edges[-1] + temp.shape[0]
        data.append(temp)
        edges.append(new_cut)

        if out is not None and new_cut - last_cut > 50000:
            data_store.resize((edges[-1], 4, 1000))
            data_store[last_cut:edges[-1]] = np.concatenate(data, axis=0)
            data = []
            last_cut = edges[-1]
            if verbose:
                print(last_cut, len(edges))

    if out is not None:
        data_store.resize((edges[-1], 4, 1000))
        data_store[last_cut:edges[-1]] = np.concatenate(data, axis=0)
        edge_name = 'left_edges' if left else 'right_edges'
        out.create_dataset(edge_name, data=edges, dtype='long', chunks=True, compression='gzip')
    else:

        return np.concatenate(data, axis=0), edges


def get_and_save_data(pairs, labels, kmers, filename, minimum_len, all_predict=False, extend_len=None, crispred=None):
    print('Extension length of anchors: ', extend_len)
    with h5py.File(filename, 'w') as out:
        pair_dtype = ','.join('uint8,u8,u8,uint8,u8,u8,u8'.split(',') + ['uint8'] * (len(pairs[0]) - 7))
        kmer_dtype = 'u8'
        if not all_predict:
            out.create_dataset('labels', data=np.array(labels, dtype='uint8'), chunks=True, compression='gzip')
        out.create_dataset('pairs', data=np.array(pairs, dtype=pair_dtype), chunks=True,compression='gzip')
        out.create_dataset('kmers', data=np.array(kmers, dtype=kmer_dtype), chunks=True, compression='gzip')
        print('Generating left anchor features...')
        get_one_side_data(pairs, left=True, out=out, verbose=True,
                          min_size=minimum_len, ext_size=extend_len, crispred=crispred)
        print('Generating right anchor features...')
        get_one_side_data(pairs, left=False, out=out, verbose=True,
                          min_size=minimum_len, ext_size=extend_len, crispred=crispred)
