import numpy as np
from time import time
import pickle
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegressionCV as LogReg
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from collections import Counter
from datetime import datetime
import subprocess


output_path = 'output/models/'
TRAIN_SPLIT = 0.9


def read_pickle_dicts(filename):
    full_dict = {}
    with open(filename, 'rb') as f:
        while True:
            try:
                full_dict.update(pickle.load(f))
            except EOFError:
                break
    return full_dict


def get_data(data_x, data_y, apply_pca=False, M=300, prev_perm=[], prev_x=[]):
    with open(data_x, 'rb') as fx:
        X = pickle.load(fx)
    with open(data_y, 'rb') as fy:
        Y = pickle.load(fy)

    N = X.shape[0]
    print('#datapoints:', N)

    if prev_perm:
        perm_i = prev_perm
    else:
        perm_i = np.random.choice(range(N), N, replace=False)

    X = X[perm_i]
    Y = Y[perm_i]

    SPLIT = int(TRAIN_SPLIT * len(Y))

    TRAIN_X = X[:SPLIT]
    TRAIN_Y = Y[:SPLIT]

    TEST_X = X[SPLIT:]
    TEST_Y = Y[SPLIT:]

    if prev_x:
        TRAIN_X = np.concatenate((TRAIN_X, prev_x[0]), axis=1)
        TEST_X = np.concatenate((TEST_X, prev_x[1]), axis=1)
        print(TRAIN_X.shape)

    if apply_pca:
        TRAIN_X, TEST_X = get_pca(TRAIN_X, TEST_X, M=M)

    return TRAIN_X, TEST_X, TRAIN_Y, TEST_Y, perm_i


def get_pca(TRAIN_X, TEST_X, M=4):
    print('Starting pca...')

    pca_ = PCA(n_components=M)
    pca_.fit(TRAIN_X)
    TRAIN_Z = pca_.transform(TRAIN_X)
    rec_x = pca_.inverse_transform(TRAIN_Z)
    rec_loss = ((TRAIN_X - rec_x) ** 2).mean()
    print(rec_loss)

    TEST_Z = pca_.transform(TEST_X)
    return TRAIN_Z, TEST_Z


def create_model(model_type, TRAIN_Y, kernel='rbf', probs=True, balance=True, balance_param=0.5, model_path=''):
    npi_counts = Counter(TRAIN_Y)
    if balance:
        npi_balance = {x: len(TRAIN_Y) / c ** balance_param for x, c in npi_counts.items()}
        print(npi_balance)
    else:
        npi_balance = {x: 1 for x in npi_counts.keys()}

    if model_path:
        model = joblib.load(model_path)
        print('Pretrained model loaded...')
    else:
        model = {
            'svm': SVC(kernel=kernel, decision_function_shape='ovo', class_weight=npi_balance, verbose=0, probability=probs),
            'logreg': LogReg(),
        }[model_type]

    return model


def classify(x, y, x_test, y_test, model, perm_i, sample_size=-1, save_model=True):
    if sample_size > 0:
        sample_i = np.random.choice(range(len(y)), sample_size, replace=False)
        x = x[sample_i]
        y = y[sample_i]

    print('Starting fitting model...')
    t0 = time()

    #model.fit(x, y)

    print('Fitting done in', time() - t0)
    y_pred = model.predict(x_test)
    print('Accuracy:', accuracy_score(y_test, y_pred))

    date = datetime.now().strftime("%d-%m|%H:%M")

    if save_model:
        with open(output_path + 'preds-' + date + '.pickle', 'wb') as file:
            pickle.dump(y_pred, file)
        joblib.dump(model, output_path + 'lr-' + date + '.pickle')
        print('Saved at:', output_path + 'lr-' + date + '.pickle')
        date = datetime.now().strftime("%d-%m|%H:%M")
        with open(output_path+str(TRAIN_SPLIT)+'~perm' + date, 'wb') as f:
            pickle.dump(perm_i, f)


def run_experiments(sample_size, model_type, kernel='rbf', save_model=True):
    print('Start experiment:', model_type, sample_size, datetime.now().strftime("%d-%m|%H:%M"))
    TRAIN_X, TEST_X, TRAIN_Y, TEST_Y, perm_i = get_data('data/scope/hx_l1_avg.pickle', 'data/scope/250000-labels-lc.pickle')
    model = create_model(model_type, TRAIN_Y, kernel=kernel, probs=False, model_path='output/models/lr-06-07|12:33.pickle')
    classify(TRAIN_X, TRAIN_Y, TEST_X, TEST_Y, model, perm_i, sample_size=sample_size, save_model=save_model)


def pickle_embs(states=['cx', 'hx'], layers=['l0', 'l1']):
    for state in states:
        for layer in layers:
            arrays = []
            data_path = 'data/scope/'
            print('Reading pickle data', state, layer)
            for i in range(50000,300000,50000):
                with open('%s%s-%d_%s.pickle' % (data_path, state, i, layer), 'rb') as f:
                    arrays.append(pickle.load(f))
            print('Writing pickle data')
            with open('%s%s_%s_avg.pickle' % (data_path, state, layer), 'wb') as f:
                pickle.dump(np.concatenate(arrays), f)
    print('Done!')


run_experiments(1, 'logreg', save_model=False)
# subprocess.call(["systemctl", "suspend"])
