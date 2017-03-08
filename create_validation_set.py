import os
import numpy as np
from scipy.io import loadmat
from scipy.io.wavfile import write

audio = loadmat(r'D:\work\Course\EEL6935 Deep Learning\Assignments\Project1\repo\resources\manatee_signals.mat')
train_signal = audio['train_signal'][:, 0]
noise_signal = audio['noise_signal'][:, 0]

#Creating a longer signal by appending
# train_signal = np.concatenate([train_signal, train_signal, train_signal])

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
write(os.path.join('resources','validation.wav'), 48000, result)
np.save(os.path.join('resources','validation.npy'), result)
