import numpy as np
import matplotlib.pyplot as plt
import peakutils

default_threshold = 0.25
default_min_dist = 100000
fs = 48000

def plot_cost(J):
    plt.figure()
    plt.plot(J); plt.ylim([0, 0.5])

def get_peaks(x, threshold, min_dist):
    i = peakutils.indexes(x, thres=threshold, min_dist=min_dist)
    i = [[a, 0] for a in i]
    return i # returns a list

def plot_calls(J):
    plt.figure()
    plt.plot(J)

    # Find peaks (representing manatee call) in the cost signal
    peaks = get_peaks(J, default_threshold, default_min_dist)
    peaks = peaks[:-1] # Ignoring last entry. Library might have some bug
    plt.plot(*zip(*peaks), marker='o', color='r', ls='')

    plt.show()

def get_scores(peaks, low, high):
    low = np.array(low) * fs
    high = np.array(high) * fs

    tp = 0; fp = 0
    num_true = low.shape[0]
    num_det = len(peaks)
    num_matches = 0
    for peak in peaks:
        match_found = False
        for i in range(num_true):
            if peak[0] >= low[i] and peak[0] <= high[i]:
                match_found = True
                break
        if match_found == True:
            num_matches += 1
            tp += 1
        else:
            fp += 1
    fn = num_true - num_matches
    return tp, fp, fn

def get_accuracy(J, low, high):
    # Find peaks (representing manatee call) in the cost signal
    peaks = get_peaks(J, default_threshold, default_min_dist)
    peaks = peaks[:-1] # Ignoring last entry. Library might have some bug

    tp, fp, fn = get_scores(peaks, low, high)
    accuracy = 100 * tp/(tp + fn)

    print('Scores: tp: {0:d} fp: {1:d} fn: {2:d}'.format(tp, fp, fn))
    print('Accuracy: {0:.2f}%'.format(accuracy))
    return accuracy






