import os
import numpy as np
import argparse
from pair_generation import sample_from_neg_pairs

def load_pairs_and_distances(files, min_length=1000, min_dist=5000, max_dist=2000000, max_length=None):
    scores = {}
    t_dists = {}

    for inter_file in files:
        with open(inter_file, 'r') as f:
            for line in f:
                tokens = line.strip().split()
                if len(tokens) < 7:
                    tokens.append(0)
                for i in [1, 2, 4, 5, 6]:
                    tokens[i] = int(float(tokens[i]))

                if tokens[0] == tokens[3] and tokens[1] > tokens[4]:
                    tokens[1], tokens[4] = tokens[4], tokens[1]
                    tokens[2], tokens[5] = tokens[5], tokens[2]

                if max_length is not None and (tokens[2] - tokens[1] > max_length or tokens[5] - tokens[4] > max_length):
                    continue

                if min_length > 0:
                    for i, j in zip([1, 4], [2, 5]):
                        if tokens[j] - tokens[i] < min_length:
                            diff = min_length - (tokens[j] - tokens[i])
                            half_diff = diff // 2
                            tokens[i] -= half_diff
                            tokens[j] += diff - half_diff

                curr_dist = 0.5 * (tokens[4] + tokens[5] - tokens[1] - tokens[2])
                if min_dist <= curr_dist <= max_dist:
                    pair = tuple(tokens[:6])
                    scores[pair] = tokens[6]
                    t_dists.setdefault(tokens[0], []).append(curr_dist)

    return scores, t_dists


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sampling 5x negative samples.")
    parser.add_argument("name", help="The name of the target. e.g. gm12878_rad21.")
    parser.add_argument("datadir", help="The directory where the input and output are in.")
    args = parser.parse_args()

    num_bins = 50
    dist_range = (np.log10(5000), np.log10(2000000))

    # 提取正样本
    pos_pairs, pos_dists = load_pairs_and_distances(
        [os.path.join(args.datadir, f"{args.name}_positive_loops.bedpe")])

    # 从3个来源中采样负样本
    neg_pairs, neg_dists = load_pairs_and_distances(
        [os.path.join(args.datadir, f"{args.name}_all_intra_negative_loops.bedpe"),
         os.path.join(args.datadir, f"{args.name}_no_tf_negative_loops.bedpe")],
        min_length=1000)
    other_neg_pairs, other_neg_dists = load_pairs_and_distances(
        [os.path.join(args.datadir, f"{args.name}_random_pairs_from_tf_and_dnase.bedpe")],
        min_length=1000)

    # 每个染色体容器中大约有5个阴性样品对应1个阳性样品
    selected_neg_pairs = sample_from_neg_pairs(pos_dists, neg_pairs, 5, other_neg_pairs, num_bins, dist_range)
    sselected_neg_dists = [0.5*(p[5]+p[4]-p[2]-p[1]) for p in selected_neg_pairs]
    with open(os.path.join(args.datadir, f"{args.name}_negative_loops.bedpe"), 'w') as out:
        for p in selected_neg_pairs:
            out.write("\t".join(map(str, p)) + "\n")


