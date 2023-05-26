import h5py
import numpy as np
import sklearn.metrics as metrics
import time
from six.moves import cPickle
import logging
import sys
from collections import Counter
from models import *
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def get_batch_data(left_seq, left_indices, right_seq, right_indices, distances, labels, kmers, start_idx, max_size=3000):
    end_idx = start_idx + 1
    curr_idx = start_idx + 1
    while curr_idx < len(left_indices) - 1:
        if max(left_indices[curr_idx] - left_indices[start_idx], right_indices[curr_idx] - right_indices[start_idx]) > max_size:
            break
        curr_idx += 1
    end_idx = max(end_idx, curr_idx - 1)
    left_seq_batch = left_seq[left_indices[start_idx]:left_indices[end_idx]]
    left_indices_batch = left_indices[start_idx:(end_idx + 1)]
    right_seq_batch = right_seq[right_indices[start_idx]:right_indices[end_idx]]
    right_indices_batch = right_indices[start_idx:(end_idx + 1)]
    labels_batch = labels[start_idx:end_idx]
    distances_batch = distances[start_idx:end_idx]
    kmers_batch = kmers[start_idx:end_idx]
    return end_idx, left_seq_batch, left_indices_batch, right_seq_batch, right_indices_batch, distances_batch, labels_batch, kmers_batch



def find_max_idx(curr_out, start, end):
    curr_max = torch.max(curr_out[start:end, :], dim=0)
    curr_idxes = curr_max[1].data.cpu().numpy()

    counts = Counter(curr_idxes)
    items = sorted(counts.items())
    largest_idx = 0
    max_count = 0
    if len(items) > 4:
        for i in range(len(items) - 4):
            curr_count = sum([x[1] for x in items[i:i+5]])
            if curr_count > max_count:
                max_count = curr_count
                largest_idx = i
    selected_idx = start + largest_idx
    return selected_idx


def extract_one_anchor_feature(seq_data, indices, extractor):
    seq = torch.autograd.Variable(torch.from_numpy(seq_data).float()).cuda()
    seq_reverse_complement = torch.autograd.Variable(torch.from_numpy(np.array(seq_data[:,::-1,::-1])).float()).cuda()
    indices = np.array(indices) - indices[0]
    combined = []
    curr_outputs = torch.cat((extractor(seq), extractor(seq_reverse_complement)), dim=1)
    for i in range(len(indices) - 1):
        combined.append(torch.max(curr_outputs[indices[i]:indices[i + 1], :], dim=0, keepdim=True)[0])
    out = torch.cat([x for x in combined], dim=0)
    return out


def extract_seq_feature(left_anchor, left_indices, right_nchor, right_indices, dists, labels,
                        kmers, start, factor_model, max_size=300, limit_to_one=False, legacy=False):
    (end, batch_left_anchor, batch_left_indices, batch_right_anchor,
     batch_right_indices, batch_dists, batch_labels, batch_kmers) = get_batch_data(left_anchor, left_indices, right_nchor, right_indices, dists,
                                                                                   labels, kmers, start, max_size=max_size)
    left_anchor_feature = extract_one_anchor_feature(batch_left_anchor, batch_left_indices, factor_model)
    right_anchor_feature = extract_one_anchor_feature(batch_right_anchor, batch_right_indices, factor_model)
    if legacy:
        batch_labels = torch.autograd.Variable(torch.from_numpy(batch_labels).long()).cuda()
    else:
        batch_labels = torch.autograd.Variable(torch.from_numpy(batch_labels).float()).cuda()
    batch_dists = torch.autograd.Variable(torch.from_numpy(np.array(batch_dists, dtype='float32'))).cuda()
    batch_kmers = torch.autograd.Variable(torch.from_numpy(np.array(batch_kmers, dtype='float32'))).cuda()
    return end, left_anchor_feature, right_anchor_feature, batch_dists, batch_labels, batch_kmers


def clf_predict(clf, left_anchor_feature, right_anchor_feature, batch_dists, batch_kmers):
    if len(batch_dists.size()) == 1:
        batch_dists = batch_dists.view(-1, 1)
    all_feature = torch.cat((left_anchor_feature, right_anchor_feature, batch_dists, batch_kmers), dim=1)
    probs = clf(all_feature)
    return probs


def write_log(valid_labels, all_probs, val_err, val_samples):
    c_auprc = [metrics.average_precision_score(valid_labels, all_probs), ]
    c_roc = [metrics.roc_auc_score(valid_labels, all_probs), ]

    logging.info("  validation loss:\t\t{:.6f}".format(val_err / val_samples))
    logging.info("  auPRCs: {}".format("\t".join(map(str, c_auprc))))
    logging.info("  auROC: {}".format("\t".join(map(str, c_roc))))
    all_preds = np.zeros(all_probs.shape[0])
    all_preds[all_probs > 0.5] = 1
    logging.info("  f1: {}".format(str(metrics.f1_score(valid_labels, all_preds))))
    logging.info("  precision: {}".format(str(metrics.precision_score(valid_labels, all_preds))))
    logging.info("  recall: {}".format(str(metrics.recall_score(valid_labels, all_preds))))
    logging.info("  accuracy: {}".format(str(metrics.accuracy_score(valid_labels, all_preds))))
    logging.info("  ratio: {}".format(np.sum(valid_labels) / len(valid_labels)))
    precision, recall, _ = metrics.precision_recall_curve(valid_labels, all_probs, pos_label=1)
    fpr, tpr, _ = metrics.roc_curve(valid_labels, all_probs, pos_label=1)


def predict_probs(extractor, clf, loss_fn, valid_left_anchor, valid_left_indices, valid_right_anchor, valid_right_indices,
                  valid_dists, valid_labels, valid_kmers, max_size=300):
    extractor.eval()
    clf.eval()
    val_err = 0.
    val_samples = 0
    all_probs = []
    edge = 0
    with torch.no_grad():
        while edge < len(valid_left_indices) - 1:
            end, left_anchor_feature, right_anchor_feature, batch_dists, \
                batch_labels, batch_kmers = extract_seq_feature(valid_left_anchor, valid_left_indices,valid_right_anchor, valid_right_indices,
                                                                valid_dists, valid_labels, valid_kmers, edge,extractor, max_size=max_size, legacy=True)
            batch_probs = clf_predict(clf, left_anchor_feature, right_anchor_feature, batch_dists, batch_kmers)
            loss = loss_fn(batch_probs, batch_labels)
            if int(torch.__version__.split('.')[1]) > 2:
                val_probs = F.softmax(batch_probs, dim=1).data.cpu().numpy()
            else:
                val_probs = F.softmax(batch_probs).data.cpu().numpy()
            val_probs = val_probs[:,1]
            all_probs.append(val_probs)
            val_err += loss.data.item() * (end - edge)
            val_samples += end - edge
            edge = end
    all_probs = np.concatenate(all_probs)
    write_log(valid_labels, all_probs, val_err, val_samples)
    return val_err / val_samples

def load_loop_hdf5(path):
    data = h5py.File(path, 'r')
    left_anchor = data['left_data']
    right_anchor = data['right_data']
    left_indices = data['left_edges'][:]
    right_indices = data['right_edges'][:]
    labels = data['labels'][:]
    pairs = data['pairs'][:]
    kmers = data['kmers'][:]
    dists = [[np.log10(abs(p[5] / 5000 - p[2] / 5000 + p[4] / 5000 - p[1] / 5000) * 0.5) / np.log10(2000001 / 5000),] + list(p)[7:] for p in pairs]
    return data,left_anchor,right_anchor,left_indices,right_indices,labels,dists,kmers


def train_extractor(extractor, classifier, target_name, model_name, model_dir, epochs=40, init_lr=0.0001,
          interval=5000, verbose=0, eps=1e-8, legacy=True):
    extractor.cuda()
    classifier.cuda()
    print('***********************************Loading data***********************************')
    (train_data, train_left_data, train_right_data,train_left_edges, train_right_edges,
     train_labels, train_dists, train_kmers) = load_loop_hdf5("%s_train.hdf5" % target_name)
    (val_data, valid_left_data, valid_right_data,valid_left_edges, valid_right_edges,
     valid_labels, valid_dists, valid_kmers) = load_loop_hdf5("%s_valid.hdf5" % target_name)

    rootLogger = logging.getLogger()
    for handler in rootLogger.handlers:
        rootLogger.removeHandler(handler)
    rootLogger.setLevel(logging.INFO)
    logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    fileHandler = logging.FileHandler('logs/' + model_name + ".log")
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)
    logging.info('learning rate: %f, eps: %f' % (init_lr, eps))

    weights = torch.FloatTensor([1, 1]).cuda()
    logging.info(str(weights))
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(list(classifier.parameters()) + list(extractor.parameters()),lr=init_lr, eps=eps,weight_decay=init_lr * 0.1)
    print('***********************************Model initial performance***********************************')
    best_val_loss = predict_probs(extractor, classifier, loss_fn,valid_left_data, valid_left_edges,
                                  valid_right_data, valid_right_edges,valid_dists, valid_labels, valid_kmers)
    print(best_val_loss)
    last_update = 0

    for epoch in range(0, epochs):
        start_time = time.time()
        i = 0
        train_loss = 0.
        processed_samples_num = 0
        extractor.train()
        classifier.train()
        last_print = 0
        curr_loss = 0.
        curr_pos = 0
        print('***********************************Training***********************************')
        while i < len(train_labels):
            end, left_out, right_out, curr_dists, \
            curr_labels, curr_kmers = extract_seq_feature(train_left_data, train_left_edges,train_right_data, train_right_edges,
                                                          train_dists, train_labels,train_kmers, i, extractor,legacy=legacy)
            if verbose > 0:
                logging.info(str(curr_dists.size()))
            curr_outputs = clf_predict(classifier, left_out, right_out, curr_dists, curr_kmers)

            loss = loss_fn(curr_outputs, curr_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            processed_samples_num += end - i
            curr_loss += loss.data.item() * (end - i)
            train_loss += loss.data.item() * (end - i)
            curr_pos += torch.sum(curr_labels).data.item()
            i = end
            if processed_samples_num < 1000 or processed_samples_num - last_print > interval:
                logging.info("%d  %f  %f  %f  %f", i, time.time() - start_time, train_loss / processed_samples_num,
                             curr_loss / (processed_samples_num - last_print), curr_pos*1.0 / (processed_samples_num - last_print))
                curr_pos = 0
                curr_loss = 0
                last_print = processed_samples_num

        logging.info("Epoch {} of {} took {:.3f}s".format(epoch + 1, epochs, time.time() - start_time))
        logging.info("Train loss: %f", train_loss / processed_samples_num)

        val_err = predict_probs(extractor, classifier, loss_fn, valid_left_data, valid_left_edges, valid_right_data,
                          valid_right_edges, valid_dists, valid_labels, valid_kmers)

        if val_err < best_val_loss or epoch == 0:
            best_val_loss = val_err
            last_update = epoch
            logging.info("current best val: %f", best_val_loss)
            torch.save(extractor.state_dict(), "{}/{}.model.pt".format(model_dir, model_name), pickle_protocol=cPickle.HIGHEST_PROTOCOL)
            torch.save(classifier.state_dict(), "{}/{}.classifier.pt".format(model_dir, model_name), pickle_protocol=cPickle.HIGHEST_PROTOCOL)
        if epoch - last_update >= 10:
            break
    train_data.close()
    val_data.close()
    fileHandler.close()
    consoleHandler.close()
    rootLogger.removeHandler(fileHandler)
    rootLogger.removeHandler(consoleHandler)


