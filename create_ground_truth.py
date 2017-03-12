import numpy as np


def create_ground_truth_signal(L, fs, low, high):
    s = np.zeros(L, dtype=bool)
    low = np.array(low) * fs
    high = np.array(high) * fs
    for start, end in zip(low, high):
        for i in range(int(start), int(end)):
            s[i] = True

    return s

if __name__ == '__main__':
    fs = 48000

    # Validation set
    if 0:
        if 1: # 25 seconds signal
            out_file = r'resources\ground_truth_val_25s_signal.npy'
            with open(r'resources\validation_25s.npy', 'rb') as f:
                s = np.load(f) # Reading signal. ndarray
            with open(r'resources\ground_truth_val_25s.npy', 'rb') as f:
                dict = np.load(f).item()
        else: # 75 seconds signal
            out_file = r'resources\ground_truth_val_75s_signal.npy'
            with open(r'resources\validation_75s.npy', 'rb') as f:
                s = np.load(f) # Reading signal. ndarray
            with open(r'resources\ground_truth_val_75s.npy', 'rb') as f:
                dict = np.load(f).item()

        low = dict['low']
        high = dict['high']

        gt = create_ground_truth_signal(len(s), fs, low, high)
        np.save(out_file, gt)

    # Test set
    if 0:
        out_file = r'resources\ground_truth_test_signal.npy'
        with open(r'resources\test_signal.npy', 'rb') as f:
            s = np.load(f) # Reading signal. ndarray
        with open(r'resources\ground_truth_test.npy', 'rb') as f:
            dict = np.load(f).item()

        low_regular = dict['low'][dict['idx_regular']]
        high_regular = dict['high'][dict['idx_regular']]
        low_all = dict['low'][dict['idx_all']]
        high_all = dict['high'][dict['idx_all']]

        gt_regular = create_ground_truth_signal(len(s), fs, low_regular, high_regular)
        gt_all = create_ground_truth_signal(len(s), fs, low_all, high_all)
        dict_out = {'regular': gt_regular, 'all': gt_all}
        np.save(out_file, dict_out)

