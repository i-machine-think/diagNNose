# TODO: REFACTOR!!

import torch
import pickle
import numpy as np
from time import time


def extract(model, data_config, cutoff=50000, bsz=50000):
    with open(data_config['parsed_data'], 'rb') as f:
        parsed_data = pickle.load(f)
    print('data loaded')

    # TODO: Add zero init option
    with open(data_config['init_embs'], 'rb') as f:
        hidden_avgs = pickle.load(f)

    activation_names = ['hx', 'cx']
    hx_l0_list, cx_l0_list = np.zeros((bsz, 650)), np.zeros((bsz, 650))
    hx_l1_list, cx_l1_list = np.zeros((bsz, 650)), np.zeros((bsz, 650))

    l0_lists = [hx_l0_list, cx_l0_list]

    l1_lists = [hx_l1_list, cx_l1_list]

    labels = []

    t0 = time()
    n = 0
    tot_n = 0
    n_sens = 0
    stop = False
    for i, data in parsed_data.items():
        h0_l0 = torch.Tensor(hidden_avgs['hx_l0'])
        c0_l0 = torch.Tensor(hidden_avgs['cx_l0'])
        h0_l1 = torch.Tensor(hidden_avgs['hx_l1'])
        c0_l1 = torch.Tensor(hidden_avgs['cx_l1'])
        if n_sens % 10 == 0:
            print(n_sens, n, time() - t0)

        for t, (word_t, label) in enumerate(zip(data.sen, data.labels)):
            if n % bsz == 0 and n > 0:
                for idx, l in enumerate(l0_lists):
                    filename = '{}/{}-{}_l0.pickle'.format(output_path_prefix, activation_names[idx], n)

                    with open(filename, 'wb') as f_out:
                        pickle.dump(np.array(l), f_out)

                for idx, l in enumerate(l1_lists):
                    filename = '{}/{}-{}_l1.pickle'.format(output_path_prefix, activation_names[idx], n)

                    with open(filename, 'wb') as f_out:
                        pickle.dump(np.array(l), f_out)

                with open('{}/{}-labels.pickle'.format(output_path_prefix, n), 'wb') as f_out:
                    pickle.dump(np.array(labels), f_out)
                l0_lists = [np.zeros((bsz, 650)), np.zeros((bsz, 650))]
                l1_lists = [np.zeros((bsz, 650)), np.zeros((bsz, 650))]

            # TODO: make more generic
            out, layer0, layer1 = model(word_t, h0_l0, c0_l0, h0_l1, c0_l1)

            for idx, l in enumerate(l0_lists):
                l[n%bsz] = (layer0[idx].detach().numpy())

            for idx, l in enumerate(l1_lists):
                l[n%bsz] = (layer1[idx].detach().numpy())

            h0_l0, c0_l0 = layer0[:2]
            h0_l1, c0_l1 = layer1[:2]

            labels.append(label)
            n += 1
            tot_n += 1
            if tot_n == cutoff:
                stop = True
                break
        if stop:
            break
        n_sens += 1

    for idx, l in enumerate(l0_lists):
        filename = '{}/{}-{}_l0.pickle'.format(output_path_prefix, activation_names[idx], n)

        with open(filename, 'wb') as f_out:
            pickle.dump(np.array(l), f_out)

    for idx, l in enumerate(l1_lists):
        filename = '{}/{}-{}_l1.pickle'.format(output_path_prefix, activation_names[idx], n)

        with open(filename, 'wb') as f_out:
            pickle.dump(np.array(l), f_out)

    with open('{}/{}-labels.pickle'.format(output_path_prefix, n), 'wb') as f_out:
        pickle.dump(np.array(labels), f_out)

    print(time() - t0)
