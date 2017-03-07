import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

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


if __name__ == '__main__':
    audio = loadmat(r'D:\work\Course\EEL6935 Deep Learning\Assignments\Project1\manatee_signals.mat')
    train_signal = audio['train_signal'][:, 0]
    noise_signal = audio['noise_signal'][:, 0]
    test_signal = audio['test_signal'][:, 0]

    # plt.plot(train_signal)

    filter_orders = [2, 6, 15, 30, 50, 80, 100]
    filter_order = 15
    w_train, x_train, J_train = lms(train_signal, filter_order, 0.01)
    # plt.plot(x_train)

    w_noise, x_noise, J_noise = lms(noise_signal, filter_order, 0.01)
    # plt.plot(x_noise)

    J_call, J_noise = detect_manatee(test_signal, w_train, w_noise)

    # plt.figure()
    # plt.plot(J_call); plt.ylim([0, 0.5])
    # plt.figure()
    # plt.plot(J_noise); plt.ylim([0, 0.5])
    plt.figure()
    plt.plot(J_noise - J_call)

    plt.show()


def train():
    # All units in seconds
    low = [2.0, 4.274, 6.562, 8.837, 11.192, 13.495, 15.785, 18.076, 20.508, 22.834]
    high = [2.274, 4.563, 6.837, 9.192, 11.495, 13.785, 16.076, 18.508, 20.834, 23.073]
    mean_length = 0.3074
    std_dev = 0.0509

'''
    if plot_type == 0:
        x_valid = x_win[m + 1:window_size]
        input_power = dot(x_valid, x_valid)/(window_size - m)
        mse = mean(J(m + 1:window_size))
        nmse = mse / input_power
        print('{0:f} '.format(nmse))
    elif plot_type == 1:
        figure, hold
        on
        title('Original vs Predicted signal')
        plot(x_win)
        plot(x_pred)
        hold
        off
        legend('Original', 'Predicted')
        xlabel('Samples -->')
        ylabel('Signal Strength -->')
        % soundsc(x_pred)
        % axis([1, window_size, -10, 10])
    elif plot_type == 2:
        plot(J)
        disp(J(1: 20)')
        disp(max(J))
        % axis([1, window_size, 0, 10])

    if (plot_type == 0):
        fprintf('\n')
'''