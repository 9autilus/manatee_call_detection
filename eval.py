import numpy as np
import matplotlib.pyplot as plt
import peakutils
from sklearn import metrics

fs = 48000  # Fixed by the problem statement

# parameters to detect peaks
default_threshold = 0.25
# min_dist value obtained from checking the minimum distance
# between manatee calls in test set. Min is 0.57 seconds
default_min_dist = int(0.5 * fs)

# Plots a cost
def plot_cost(J):
    plt.figure()
    plt.plot(J); plt.ylim([0, 0.5])

# Returns a list of peak locations in signal x
def get_peaks(x, threshold, min_dist):
    i = peakutils.indexes(x, thres=threshold, min_dist=min_dist)
    i = [[a, 0] for a in i]
    return i # returns a list

# Plots detected peaks (with a red dot) overlayed on signal J
def plot_calls(J):
    plt.figure()
    plt.plot(J)

    # Find peaks (representing manatee call) in the cost signal
    peaks = get_peaks(J, default_threshold, default_min_dist)
    peaks = peaks[:-1] # Ignoring last entry. Library might have some bug
    plt.plot(*zip(*peaks), marker='o', color='r', ls='')

    plt.show()

# Returns TP, FP and FN on a list of peaks and
# square wave info packed inside low/high
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

# Returns Accuracy of a cost plot
# Finds the peaks and computes accuracy
def get_accuracy(J, low, high):
    # Find peaks (representing manatee call) in the cost signal
    peaks = get_peaks(J, default_threshold, default_min_dist)
    peaks = peaks[:-1] # Ignoring last entry. Library might have some bug

    tp, fp, fn = get_scores(peaks, low, high)
    accuracy = 100 * tp/(tp + fn)

    print('Scores: tp: {0:d} fp: {1:d} fn: {2:d}'.format(tp, fp, fn))
    print('Accuracy: {0:.2f}%'.format(accuracy))
    return accuracy

def get_pr_curve(J, low, high):
    print('Computing Precision Recall curve...')
    num_points = 5
    thresholds = np.linspace(0.1, np.max(J), num_points)

    # num_points = 1; thresholds = [0.25]

    precision = np.empty(num_points)
    recall = np.empty(num_points)

    for i, threshold in enumerate(thresholds):
        # Find peaks (representing manatee call) in the cost signal
        peaks = get_peaks(J, threshold, default_min_dist)
        peaks = peaks[:-1]  # Ignoring last entry. Library might have some bug

        tp, fp, fn = get_scores(peaks, low, high)
        precision[i] = tp/(tp + fp)
        recall[i] = tp/(tp + fn)

    avg_precision = 0
    for i in range(1, len(thresholds)):
        avg_precision += precision[i] * (recall[i] - recall[i-1])

    print("Average Precision: {0:.2f}".format(avg_precision))

    plt.figure()
    plt.plot(recall, precision)
    plt.show()

def get_roc_curve(J, gt):
    fpr, tpr, thresholds = metrics.roc_curve(gt, J)
    roc_auc = metrics.auc(fpr, tpr)

    plt.figure()
    lw = 2 #linewidth
    plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.show()