import os
import numpy as np
from scipy.io import loadmat
from scipy.io.wavfile import write

def create_val_npy(train_signal, noise_signal, type):
    if type == '75s':
        #Creating a longer signal by appending
        train_signal = np.concatenate([train_signal, train_signal, train_signal])

    len_train = train_signal.shape[0]
    len_noise = noise_signal.shape[0]

    num_noise = int(np.ceil(len_train/len_noise))

    # Create a longer noise sequence to match with the length of train signal
    longer_noise = np.empty(len_train)
    idx = 0
    for i in range(num_noise-1):
        longer_noise[idx:idx+len_noise] = noise_signal
        idx += len_noise
    remaining = num_noise * len_noise - len_train
    longer_noise[-remaining:] = noise_signal[:remaining]

    # Combine train signal with noise signal
    result = longer_noise + train_signal
    # Rescale signal back to same level
    scale = float(np.maximum(np.abs(np.min(result)), np.abs(np.max(result))))
    result = result/scale # Final signal should be in [-1, +1] range

    # write results to disk
    write(os.path.join('resources','validation' + '_' + type + '.wav'), 48000, result)
    np.save(os.path.join('resources','validation' + '_' + type + '.npy'), result)

'''
Read the .mat file and create train_signal.npy and validation signals
'''
if __name__ == '__main__':
    audio = loadmat(r'resources\manatee_signals.mat')
    train_signal = audio['train_signal'][:, 0]
    noise_signal = audio['noise_signal'][:, 0]

    if 0:
        create_val_npy(train_signal, noise_signal, '25s')

    if 0:
        create_val_npy(train_signal, noise_signal, '75s')

    if 0:
        write(r'resources\train_signal.wav', 48000, train_signal)
        np.save(r'resources\train_signal.npy', train_signal)

    if 0:
        write(r'resources\noise_signal.wav', 48000, noise_signal)
        np.save(r'resources\noise_signal.npy', noise_signal)