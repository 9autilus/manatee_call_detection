import numpy as np
from scipy.io import loadmat
from scipy.io.wavfile import write

audio = loadmat(r'D:\work\Course\EEL6935 Deep Learning\Assignments\Project1\manatee_signals.mat')
train_signal = audio['train_signal'][:, 0]
noise_signal = audio['noise_signal'][:, 0]

len_train = train_signal.shape[0]
len_noise = noise_signal.shape[0]

num_noise = int(np.ceil(len_train/len_noise))
longer_noise = np.empty(len_train)

idx = 0
for i in range(num_noise-1):
    longer_noise[idx:idx+len_noise] = noise_signal
    idx += len_noise

remaining = num_noise * len_noise - len_train
longer_noise[-remaining:] = noise_signal[:remaining]

result = longer_noise + train_signal
result = result * 1/0.8 # Rescale signal back to same level



write('validation.wav', 48000, result)
np.save('validation.npy', result)
