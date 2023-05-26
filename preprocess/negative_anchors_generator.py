import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
import numpy as np
import argparse
from pair_generation import load_data, get_clusters, get_neg_pairs, get_cluster_sizes, print_total_pairs
from pair_generation import save_neg_pairs

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Generate negative loops.")
    parser.add_argument("name", help="The name of the target. Like gm12878_rad21.")
    parser.add_argument("outdir", help="The output directory.")
    args = parser.parse_args()

    anchor_file = "{}/{}_positive_anchors.bed".format(args.outdir, args.name)
    loop_file = "{}/{}_positive_loops.bedpe".format(args.outdir, args.name)
    folds = []
    exclusive_intra = [False, True]

    for type in exclusive_intra:
        anchors, interactions, dists = load_data(anchor_file, loop_file)
        clusters = get_clusters(anchors)
        bin_stats = np.histogram(np.log10(dists), bins=10, range=(np.log10(5000), np.log10(2000000)))
        histogram_counts, histogram_edges = bin_stats
        cluster_sizes = get_cluster_sizes(clusters)
        print_total_pairs(cluster_sizes)
        all_pairs = get_neg_pairs(interactions, clusters, bin_stats, allow_intra=type, only_intra=type, fold=None)
        selected_pairs = []
        for k in all_pairs:
            selected_pairs += list(k)
        print(len(selected_pairs))
        intra_type = "exclusive_intra" if type else "all_intra"
        name = "{}_{}".format(args.name, intra_type)
        save_neg_pairs("{}/{}_negative_loops.bedpe".format(args.outdir, name), selected_pairs)
