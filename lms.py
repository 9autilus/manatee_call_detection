import os
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import eval
import time




def lms(x, m, lr):
    N = x.shape[0]
    e = np.zeros(N)
    J = np.zeros(N)
    x_pred = np.zeros(N)
    w_n = np.zeros(m)

    for i in range(m, N):
        x_n = x[i - m:i]
        x_pred[i] = np.dot(w_n, x_n)
        e[i] = x[i] - x_pred[i]
        J[i] = e[i] ** 2
        w_n = w_n + lr * e[i] * x_n

    # Reversing w so that first entries represent coefficients
    # that are closest in time domain from current samples
    # print(np.flip(w_n, axis=0))

    return w_n, x_pred, J

def smooth(x):
    box_pts = 100
    box = np.ones(box_pts)/box_pts
    x_smooth = np.convolve(x, box, mode='same')
    return x_smooth

def detect_manatee(x, w_call, w_noise):
    m = w_call.shape[0]
    N = x.shape[0]
    e = np.zeros(N)
    J_call = np.zeros(N)
    J_noise = np.zeros(N)
    x_pred_call = np.zeros(N)
    x_pred_noise = np.zeros(N)

    for i in range(m, N):
        x_pred_call[i] = np.dot(w_call, x[i - m:i])
        x_pred_noise[i] = np.dot(w_noise, x[i - m:i])

    J_call = (x_pred_call - x) ** 2
    J_noise = (x_pred_noise - x) ** 2
    J_call = smooth(J_call)
    J_noise = smooth(J_noise)
    return J_call, J_noise

def train_filter(filter_order):
    # Load train signal
    with open(r'resources\train_signal.npy', 'rb') as f:
        train_signal = np.load(f)
    with open(r'resources\noise_signal.npy', 'rb') as f:
        noise_signal = np.load(f)

    w_train, x_train, J_train = lms(train_signal, filter_order, 0.01)
    # plt.plot(x_train)

    w_noise, x_noise, J_noise = lms(noise_signal, filter_order, 0.01)
    # plt.plot(x_noise)

    return w_train, w_noise

def run_test_set(x, gt, w_train, w_noise):
    print('Running test...')
    auc = -1
    dict_roc = {}
    J_call, J_noise = detect_manatee(x, w_train, w_noise)
    # eval.plot_cost(J_call)
    # eval.plot_cost(J_noise)
    J_diff = J_noise - J_call

    if 0:
        eval.plot_calls(J_diff)

    if 1:
        dict_roc = eval.get_roc_curve(J_diff, gt)

    test_result = {'roc': dict_roc}
    return test_result

if __name__ == '__main__':
    start = time.time()

    # Configurable parameters
    # filter_orders = [1,2,4,6,8,10,13,15,17, 20, 25, 30, 35, 50]
    filter_orders = [1, 2, 5, 10]
    plot_auc_vs_filter_size = False
    plot_roc_vs_filter_size = False
    val_type = '25s'

    # Load validation data
    print('Loading validation data...')
    if val_type == '75s':
        with open(r'resources\validation_75s.npy', 'rb') as f:
            x_val = np.load(f)
        with open(r'resources\ground_truth_val_75s.npy', 'rb') as f:
            gt_val = np.load(f)
    else:
        with open(r'resources\validation_25s.npy', 'rb') as f:
            x_val = np.load(f)
        with open(r'resources\ground_truth_val_25s_signal.npy', 'rb') as f:
            gt_val = np.load(f)

    # Read test data
    print('Loading Test data...')
    with open(r'resources\test_signal.npy', 'rb') as f:
        x_test = np.load(f)
    with open(r'resources\ground_truth_test_signal.npy', 'rb') as f:
        dict_signal = np.load(f).item()
        gt_test = dict_signal['regular']

    if plot_auc_vs_filter_size:
        auc = np.zeros([2, len(filter_orders)])

    if plot_roc_vs_filter_size:
        fpr_list = [[]] * len(filter_orders)
        tpr_list = [[]] * len(filter_orders)
        auc = np.zeros(len(filter_orders))

    '''
    main working code
    '''
    for i, filter_order in enumerate(filter_orders):
        w_train = w_noise = None
        weights_file = r'resources\lms_weights_w'+str(filter_order)+'.npy'
        if os.path.exists(weights_file):
            print('Filter order: {0:d} Getting stored weights...'.format(filter_order))
            with open(weights_file, 'rb') as f:
                dict = np.load(f).item()
            w_train = dict['w_train']
            w_noise = dict['w_noise']
        else:
            print('Filter order: {0:d} Computing weights...'.format(filter_order))
            w_train, w_noise = train_filter(filter_order)
            weights = {'w_train': w_train, 'w_noise': w_noise}
            np.save(weights_file, weights)

        # Testing while plotting AUC
        if plot_roc_vs_filter_size:
            if 1:
                result = run_test_set(x_val, gt_val, w_train, w_noise)
            else:
                result = run_test_set(x_test, gt_test, w_train, w_noise)
            auc[i] = result['roc']['auc']
            fpr_list[i] = result['roc']['fpr']
            tpr_list[i] = result['roc']['tpr']

        if plot_auc_vs_filter_size:
            result = run_test_set(x_val, gt_val, w_train, w_noise)
            auc[0, i] = result['roc']['auc']
            result = run_test_set(x_test, gt_test, w_train, w_noise)
            auc[1, i] = result['roc']['auc']

    if plot_roc_vs_filter_size:
        plt.figure()
        for i, filter_order in enumerate(filter_orders):
            lw = 1 #linewidth
            plt.plot(fpr_list[i], tpr_list[i],lw=lw, label='Filter Order:%d AUC:%.2f)' % (filter_order, auc[i]))
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC curve')
            plt.legend(loc="lower right")
        plt.show()

    # AUC vs filter size plot
    if plot_auc_vs_filter_size:
        plt.figure()
        lw = 2 #linewidth
        plt.plot(filter_orders, auc[0], color='blue',lw=lw, label='Validation Set')
        plt.plot(filter_orders, auc[1], color='red',lw=lw, label='Test Set')
        plt.xlim([1, max(filter_orders)])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Filter Order')
        plt.ylabel('AUC')
        plt.title('AUC vs Filter Order')
        plt.legend(loc="lower right")
        plt.show()



    end = time.time()
    print('Time taken: ', end - start)