# EXP.NO.5-Simulation-of-Signal-Sampling-Using-Various-Types
5.Simulation of Signal Sampling Using Various Types such as
    i) Ideal Sampling
    ii) Natural Sampling
    iii) Flat Top Sampling

# AIM
To study and analyze the sampling of natural, ideal and flat top sampling.

# SOFTWARE REQUIRED
Google Colab

# ALGORITHMS
step1:Generate a continuous signal using a sine wave.
step2:Apply uniform sampling by selecting fixed-interval samples.
step3:Apply random sampling by selecting random indices.
step4:Apply Platop sampling using probability-based selection.
step5:Plot the original signal and sampled points.
step6:reconstruct the signal using resampling.

# PROGRAM
``` python
#impulse sampling
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample
fs = 600
t = np.arange(0, 1, 1/fs)
f = 8
signal = np.sin(2 * np.pi * f * t)
plt.figure(figsize=(10, 4))
plt.plot(t, signal, label='Continuous Signal')
plt.title('Continuous Signal (fs = 100 Hz)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.show()
t_sampled = np.arange(0, 1, 1/fs)
signal_sampled = np.sin(2 * np.pi * f * t_sampled)
plt.figure(figsize=(10, 4))
plt.plot(t, signal, label='Continuous Signal', alpha=0.7)
plt.stem(t_sampled, signal_sampled, linefmt='r-', markerfmt='ro', basefmt='r-', label='Sampled
Signal (fs = 100 Hz)')
plt.title('Sampling of Continuous Signal (fs = 100 Hz)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.show()
reconstructed_signal = resample(signal_sampled, len(t))
plt.figure(figsize=(10, 4))
plt.plot(t, signal, label='Continuous Signal', alpha=0.7)
plt.plot(t, reconstructed_signal, 'r--', label='Reconstructed Signal (fs = 100 Hz)')
plt.title('Reconstruction of Sampled Signal (fs = 100 Hz)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.show()
#natural sampling
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
fs = 1000
T = 1
t = np.arange(0, T, 1/fs)
fm = 10
message_signal = np.sin(2 * np.pi * fm * t)
pulse_rate = 50
pulse_train = np.zeros_like(t)
pulse_width = int(fs / pulse_rate / 2)
for i in range(0, len(t), int(fs / pulse_rate)):
pulse_train[i:i+pulse_width] = 1
nat_ signal = message_signal * pulse_train
sampled_signal = nat_signal[pulse_train == 1]
sample_times = t[pulse_train == 1]
reconstructed_signal = np.zeros_like(t)
for i, time in enumerate(sample_times):
index = np.argmin(np.abs(t - time))
reconstructed_signal[index:index+pulse_width] = sampled_signal[i]
def lowpass_filter(signal, cutoff, fs, order=5): nyquist = 0.5 * fs normal_cutoff = cutoff / nyquist b, a
= butter(order, normal_cutoff, btype='low', analog=False) return lfilter(b, a, signal)
reconstructed_signal = lowpass_filter(reconstructed_signal,10, fs)
plt.figure(figsize=(14, 10))
Original Message Signal
plt.subplot(3, 1, 1)
plt.plot(t, message_signal, label='Original Message Signal')
plt.legend()
plt.grid(True)
Pulse Train
plt.subplot(4, 1, 2)
plt.plot(t, pulse_train, label='Pulse Train')
plt.legend()
plt.grid(True)
Natural Sampling
plt.subplot(4, 1, 3)
plt.plot(t, nat_signal, label='Natural Sampling')
plt.legend()
plt.grid(True)
Reconstructed Signal
plt.subplot(4, 1, 4)
plt.plot(t, reconstructed_signal, label='Reconstructed Message Signal', color='green')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
flattop sampling
import numpy as np import matplotlib.pyplot as plt from scipy.signal import resample
def platop_sampling(probabilities, platop=0.9): """ Platop Sampling: A modified nucleus sampling
approach. :param probabilities: List or numpy array of probabilities for each token. :param platop:
The cumulative probability threshold for nucleus sampling. :return: Index of the sampled token. """
sorted_indices = np.argsort(probabilities)[::-1] # Sort indices by probability (descending order)
sorted_probs = probabilities[sorted_indices] # Sort probabilities accordingly
cumulative_probs = np.cumsum(sorted_probs) # Compute cumulative probabilities
cutoff_index = np.searchsorted(cumulative_probs, platop) + 1 # Find the cutoff index
# Restrict to the nucleus of tokens
nucleus_indices = sorted_indices[:cutoff_index]
nucleus_probs = sorted_probs[:cutoff_index]
nucleus_probs /= nucleus_probs.sum() # Normalize probabilities
# Sample from the nucleus
sampled_index = np.random.choice(nucleus_indices, p=nucleus_probs)
return sampled_index
fs = 100 # Sampling frequency t = np.arange(0, 1, 1/fs) # Time vector f = 5 # Frequency of the sine
wave signal = np.sin(2 * np.pi * f * t) # Generate sine wave
Plot continuous signal
plt.figure(figsize=(10, 4))
plt.plot(t, signal, label='Continuous Signal')
plt.title('Continuous Signal (fs = 100 Hz)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.show()
Sampling using Platop Sampling
probs = np.abs(signal) / np.sum(np.abs(signal)) # Normalize probabilities
t_sampled_indices = [platop_sampling(probs) for _ in range(len(t)//2)] # Select indices
signal_sampled = signal[t_sampled_indices] # Sampled signal values
t_sampled = t[t_sampled_indices] # Corresponding time values
Plot sampled signal
plt.figure(figsize=(10, 4))
plt.plot(t, signal, label='Continuous Signal', alpha=0.7)
plt.stem(t_sampled, signal_sampled, linefmt='r-', markerfmt='ro', basefmt='r-', label='Platop
Sampled Signal')
plt.title('Platop Sampling of Continuous Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.show()
Reconstruction
reconstructed_signal = resample(signal_sampled, len(t))
plt.figure(figsize=(10, 4))
plt.plot(t, signal, label='Original Signal', alpha=0.7)
plt.plot(t, reconstructed_signal, 'r--', label='Reconstructed Signal')
plt.title('Reconstruction of Platop Sampled Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.show()
```

# OUTPUT
![image](https://github.com/user-attachments/assets/9a0d7586-2766-42cd-b2f3-ea476a8951ea)
![image](https://github.com/user-attachments/assets/3feaa27f-1ce4-415a-9b8a-deb3a715cd40)
![image](https://github.com/user-attachments/assets/acb0870f-76de-4e64-b254-d475755d295c)
![image](https://github.com/user-attachments/assets/c8e32afe-7266-4f57-bfd1-3f1e6e98276a)
![image](https://github.com/user-attachments/assets/7688cdd4-88e8-40f5-9980-3bb14ec28c79)





![download (2)](https://github.com/user-attachments/assets/368e3b43-7a2b-47c0-bed7-bab2494b94d9)

 
# RESULT / CONCLUSIONS
Thus the sampling of natural, ideal and flattop sampling techniques were analyzed.

