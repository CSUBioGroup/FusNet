import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"
import argparse
import h5py
import numpy as np
import torch
from models import SeqFeatureModel
import train
import time


def get_args():
    parser = argparse.ArgumentParser('Extract seq feature and save to hdf5 file.')
    parser.add_argument('extractor', help='The path of extractor model.')
    parser.add_argument('loop_hdf5', help='The loop file in hdf5 format.')
    parser.add_argument('output_name', help='The name of the output file.')
    parser.add_argument('out_dir', help='The output directory.')
    parser.add_argument('-p', '--all_predict', type=bool, required=False, default=False, help='Use all data for prediction.')
    return parser.parse_args()


def load_extractor_model(extractor_model):
    extractor = SeqFeatureModel(4, use_weightsum=True, leaky=True, use_sigmoid=False)
    extractor.load_state_dict(torch.load(extractor_model))
    extractor.cuda()
    extractor.eval()
    return extractor

def extract_seq_feature_to_hdf5(extractor, loop_hdf5, output_name, out_dir, all_for_prediction):
    start_time = time.time()


    def generate_factor_output(output_name1, file_path):
        print('Generating:', output_name1, 'factor_outputs.hdf5')
        (train_data, train_left_data, train_right_data, train_left_edges, train_right_edges,
         train_labels, train_dists, train_kmers) = train.load_loop_hdf5(file_path)

        save_feature = h5py.File(os.path.join(out_dir, output_name1 + 'factor_outputs.hdf5'), 'w')
        left_anchor_feature = save_feature.create_dataset('left_out', (len(train_labels), extractor.num_filters[-1] * 2),
                                                          dtype='float32', chunks=True, compression='gzip')
        right_anchor_feature = save_feature.create_dataset('right_out', (len(train_labels), extractor.num_filters[-1] * 2),
                                                           dtype='float32', chunks=True, compression='gzip')
        pairs = train_data['pairs'][:]
        pair_dtype = ','.join('uint8,u8,u8,uint8,u8,u8,u8'.split(',') + ['uint8'] * (len(pairs[0]) - 7))

        save_feature.create_dataset('dists', data=train_dists, dtype='float32',chunks=True, compression='gzip')
        save_feature.create_dataset('pairs', data=np.array(pairs, dtype=pair_dtype), chunks=True, compression='gzip')
        save_feature.create_dataset('kmers', data=np.array(train_kmers, dtype='float32'), chunks=True, compression='gzip')
        save_feature.create_dataset('labels', data=train_labels, dtype='uint8')
        i = 0
        last_print = 0
        with torch.no_grad():
            while i < len(train_left_edges) - 1:
                end, left_out, right_out, _, _, _ = train.extract_seq_feature(train_left_data, train_left_edges, train_right_data,
                                                                              train_right_edges, train_dists, train_labels, train_kmers, i,
                                                                              factor_model=extractor, max_size=2000)
                left_anchor_feature[i:end] = left_out.data.cpu().numpy()
                right_anchor_feature[i:end] = right_out.data.cpu().numpy()
                if end - last_print > 5000:
                    last_print = end
                    elapsed_time = time.time() - start_time
                    minutes = int(elapsed_time // 60)
                    seconds = int(elapsed_time % 60)
                    print('Extracted: %d / %d Spend: %d minutes %d seconds' % (end, len(train_labels), minutes, seconds))
                i = end
        save_feature.close()

    if not all_for_prediction:
        for i in ['train', 'valid', 'test']:
            file_path = loop_hdf5 + '_' + i + '.hdf5'
            output_name1 = output_name + '_' + i + '_'
            generate_factor_output(output_name1, file_path)
    else:
        file_path = loop_hdf5 + '.hdf5'
        output_name1 = output_name + '_'
        generate_factor_output(output_name1, file_path)


if __name__ == '__main__':
    args = get_args()
    extractor = load_extractor_model(args.extractor)
    extract_seq_feature_to_hdf5(extractor, args.loop_hdf5, args.output_name, args.out_dir, all_for_prediction=args.all_predict)


