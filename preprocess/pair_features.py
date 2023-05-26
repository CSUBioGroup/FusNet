# coding: utf-8

import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICE']='3'
import read_reference


CHANNELS_FIRST = "channels_first"
CHANNELS_LAST = "channels_last"

#将碱基序列转换成四维向量
def get_seq_matrix(seq, seq_len: int, data_format: str, one_d: bool, rc=False):
    channels = 4
    # seq长度*4列
    mat = np.zeros((seq_len, channels), dtype="float32")

    for i, a in enumerate(seq):
        idx = i
        if idx >= seq_len:
            break
        a = a.lower()

        if a == 'a':
            mat[idx, 0] = 1
        elif a == 'g':
            mat[idx, 1] = 1
        elif a == 'c':
            mat[idx, 2] = 1
        elif a == 't':
            mat[idx, 3] = 1
        else:
            mat[idx, 0:4] = 0

    if rc:

        mat = mat[::-1, ::-1]

    if not one_d:

        mat = mat.reshape((1, seq_len, channels))

    if data_format == CHANNELS_FIRST:
        axes_order = [len(mat.shape)-1,] + [i for i in range(len(mat.shape)-1)]

        mat = mat.transpose(axes_order)

    return mat



def _get_sequence(chrom, start, end, min_size=1000, crispred=None):

    if crispred is not None:

        seq = ''
        curr_start = start
        for cc, cs, ce in crispred:
            if chrom == cc and min(end, ce) > max(cs, curr_start):
                if curr_start > cs:
                    seq += read_reference.hg19[chrom][curr_start:cs]
                curr_start = ce
        if curr_start < end:
            seq += read_reference.hg19[chrom][curr_start:end]

    else:
        seq = read_reference.hg19[chrom][start:end]

    if len(seq) < min_size:
        diff = min_size - (end - start)
        ext_left = diff // 2
        if start - ext_left < 0:
            ext_left = start
        elif diff - ext_left + end > len(read_reference.hg19[chrom]):
            ext_left = diff - (len(read_reference.hg19[chrom]) - end)
        curr_start = start - ext_left
        curr_end = end + diff - ext_left

        if curr_start < start:
            seq = read_reference.hg19[chrom][curr_start:start] + seq
        if curr_end > end:
            seq = seq + read_reference.hg19[chrom][end:curr_end]
    if start < 0 or end > len(read_reference.hg19[chrom]):
        return None
    return seq


def encode_seq(chrom, start, end, min_size=1000, crispred=None):
    seq = _get_sequence(chrom, start, end, min_size, crispred)
    if seq is None:
        return None
    mat = get_seq_matrix(seq, len(seq), 'channels_first', one_d=True, rc=False)
    parts = []
    for i in range(0, len(seq), 500):
        if i + 1000 >= len(seq):
            break
        parts.append(mat[:, i:i + 1000])
    parts.append(mat[:, -1000:])
    parts = np.array(parts, dtype='float32')
    return parts









