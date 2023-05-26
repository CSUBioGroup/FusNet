import os
import argparse
import h5py
import time
import xgboost as xgb
import lightgbm as lgb
import joblib
import numpy as np

from sklearn.metrics import precision_score, recall_score, f1_score, auc, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression



def load_factor_outputs(fn):
    f = h5py.File(fn, 'r')
    left_out = f['left_out'][:]
    right_out = f['right_out'][:]
    dists = f['dists'][:]
    kmers = f['kmers'][:]
    labels = f['labels'][:]
    if 'pairs' in f:
        pairs = f['pairs'][:]
    else:
        pairs = None
    data = np.concatenate((left_out, right_out, dists, kmers), axis=1)
    return data, labels, pairs


def train_knn(X_train, y_train):
    knn = KNeighborsClassifier(
        n_neighbors=10,
        weights='distance',
        algorithm='ball_tree',
        leaf_size=30,
        p=2,
        metric='minkowski'
    )
    start = time.time()
    print('training KNN...')
    knn.fit(X_train, y_train)
    end = time.time()
    print("KNN train time:", end - start, "s")

    return knn


def train_xgb(X_train, y_train, X_val, y_val, n_estimators=1000, threads=20, max_depth=6, verbose_eval=True):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    evallist = [(dtrain, 'train'), (dval, 'eval')]
    evals_result = {}
    params = {'max_depth': max_depth, 'objective': 'binary:logistic',
              'eta': 0.1, 'nthread': threads, 'eval_metric': ['aucpr', 'map', 'logloss']}
    print('training XGB...')
    bst = xgb.train(params, dtrain, n_estimators, evallist, early_stopping_rounds=40,
                    verbose_eval=verbose_eval, evals_result=evals_result)

    return bst


def train_lgb(X_train, y_train, X_val, y_val):
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    }


    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val)


    print('training LGB...')
    model = lgb.train(params, train_data, num_boost_round=100, valid_sets=val_data, early_stopping_rounds=10)

    return model


def print_performance(model_name, val_label, pred_prob):
    print('Performance of ', model_name, ':')
    acc = accuracy_score(val_label, pred_prob.round())
    print('Accuracy:', acc)
    precision = precision_score(val_label, pred_prob.round())
    print('Precision:', precision)
    recall = recall_score(val_label, pred_prob.round())
    print('Recall:', recall)
    f1 = f1_score(val_label, pred_prob.round())
    print('F1-Score:', f1)


def train_lr(stacking_X_train, stacking_y_train):
    lr = LogisticRegression(
        penalty='l2',            # Regularization type: L1 ('l1') or L2 ('l2')
        C=1.0,                   # Inverse of regularization strength; smaller values specify stronger regularization
        solver='lbfgs',          # Algorithm to use for optimization
        max_iter=100,            # Maximum number of iterations for the solver to converge
        random_state=42,         # Seed for random number generation
        n_jobs=None              # Number of CPU cores to use for parallel processing (-1 for all cores)
    )
    print('Training Logistic Regression...')
    lr.fit(stacking_X_train, stacking_y_train)
    return lr


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train predictors using extended datasets.")
    parser.add_argument('data_dir', help='The directory of the data location with train, valid, and test.')
    parser.add_argument('dataset_name',
                        help='The name (prefix) of the dataset before _[train|valid|test]_factor_outputs.hdf5.' +
                             ' For example: gm12878_ctcf_train_factor_outputs.hdf5 -> gm12878_ctcf')

    args = parser.parse_args()
    name = args.dataset_name

    train_data, train_labels, _ = load_factor_outputs(os.path.join(args.data_dir, f'{name}_train_factor_outputs.hdf5'))
    val_data, val_labels, _ = load_factor_outputs(os.path.join(args.data_dir, f'{name}_valid_factor_outputs.hdf5'))
    test_data, test_labels, test_pairs = load_factor_outputs(
        os.path.join(args.data_dir, f'{name}_test_factor_outputs.hdf5'))

    print('KNN training...')
    knn_predictor = train_knn(train_data, train_labels)
    joblib.dump(knn_predictor, os.path.join(args.data_dir, f"{name}_knn_predictor.pkl"))
    print('XGB training...')
    xgb_predictor = train_xgb(train_data, train_labels, val_data, val_labels)
    joblib.dump(xgb_predictor, os.path.join(args.data_dir, f"{name}_xgb_predictor.pkl"))
    print('LGB traing...')
    lgb_predictor = train_lgb(train_data, train_labels, val_data, val_labels)
    joblib.dump(lgb_predictor, os.path.join(args.data_dir, f"{name}_lgb_predictor.pkl"))

    print('Predicting KNN...')
    knn_val_pred = knn_predictor.predict_proba(val_data)[:, 1]
    print('Predicting XGB...')
    xgb_val_pred = xgb_predictor.predict(xgb.DMatrix(val_data))
    print('Predicting LGB...')
    lgb_val_pred = lgb_predictor.predict(val_data)
    print_performance('KNN predictor', val_labels, knn_val_pred)
    print('*****************************************************')
    print_performance('XGB predictor', val_labels, xgb_val_pred)
    print('*****************************************************')
    print_performance('LGB predictor', val_labels, lgb_val_pred)
    print('*****************************************************')

    # 训练模型融合层
    stacking_X = np.vstack([knn_val_pred, xgb_val_pred, lgb_val_pred]).T
    stacking_y = val_labels
    stacking_X_train, stacking_X_test, stacking_y_train, stacking_y_test = train_test_split(stacking_X, stacking_y,
                                                                                            test_size=0.3,
                                                                                            random_state=42)


    print('Training Logistic Regression...')
    lr_predictor = train_lr(stacking_X_train, stacking_y_train)
    joblib.dump(lr_predictor, os.path.join(args.data_dir, f"{name}_lr_predictor.pkl"))


    print('Predicting Logistic Regression...')
    lr_val_pred = lr_predictor.predict_proba(stacking_X_test)[:, 1]
    print_performance('Logistic Regression predictor', stacking_y_test, lr_val_pred)

