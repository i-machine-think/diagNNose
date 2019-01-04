import numpy as np
from time import time
import pickle
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegressionCV as LogReg
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from collections import Counter
from datetime import datetime
import subprocess


output_path = './data/output/trained_classifiers/'
TRAIN_SPLIT = 0.9


# Read multiple pickle dumps to one dictionary
def read_pickle_dicts(filename):
    full_dict = {}
    with open(filename, 'rb') as f:
        while True:
            try:
                full_dict.update(pickle.load(f))
            except EOFError:
                break
    return full_dict


def get_data(data_x, data_y, prev_perm=[], prev_x=[]):
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

    return TRAIN_X, TEST_X, TRAIN_Y, TEST_Y, perm_i


def create_model(model_type, TRAIN_Y=None, kernel='rbf', probs=True, balance=False, balance_param=0.5, model_path=''):
    npi_counts = Counter(TRAIN_Y)
    if balance:
        assert TRAIN_Y is not None, 'Provide label data for class balancing'
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


def classify(x, y, x_test, y_test, model, sample_size=-1, save_model=True):
    if sample_size > 0:
        sample_i = np.random.choice(range(len(y)), sample_size, replace=False)
        x = x[sample_i]
        y = y[sample_i]

    print('Starting fitting model...')
    t0 = time()

    model.fit(x, y)

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


def run_experiments(xpathname, ypathname, sample_size=-1, model_type='logreg', kernel='rbf', save_model=True, kill=False):
    print('Start experiment:', model_type, sample_size, datetime.now().strftime("%d-%m|%H:%M"))

    TRAIN_X, TEST_X, TRAIN_Y, TEST_Y, perm_i = get_data(xpathname, ypathname)

    model = create_model('logreg')

    classify(TRAIN_X, TRAIN_Y, TEST_X, TEST_Y, model, sample_size=sample_size, save_model=save_model)

    if kill:
        subprocess.call(["systemctl", "suspend"])
