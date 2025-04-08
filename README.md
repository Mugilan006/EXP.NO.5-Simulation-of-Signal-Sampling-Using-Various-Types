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
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample, butter, lfilter, filtfilt

# ------------------------------------------------
# IMPULSE SAMPLING
# ------------------------------------------------
fs_impulse = 100
t_impulse = np.arange(0, 1, 1/fs_impulse)
f = 8
signal = np.sin(2 * np.pi * f * t_impulse)

plt.figure(figsize=(10, 4))
plt.plot(t_impulse, signal, label='Continuous Signal')
plt.title('Continuous Signal (fs = 100 Hz)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.show()

# Impulse Sampling
t_sampled = np.arange(0, 1, 1/fs_impulse)
signal_sampled = np.sin(2 * np.pi * f * t_sampled)

plt.figure(figsize=(10, 4))
plt.plot(t_impulse, signal, label='Continuous Signal', alpha=0.7)
plt.stem(t_sampled, signal_sampled, linefmt='r-', markerfmt='ro', basefmt='r-', label='Sampled Signal')
plt.title('Impulse Sampling (fs = 100 Hz)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.show()

# Reconstruction from Impulse Sampled Signal using resample
reconstructed_signal = resample(signal_sampled, len(t_impulse))

plt.figure(figsize=(10, 4))
plt.plot(t_impulse, signal, label='Original Signal', alpha=0.7)
plt.plot(t_impulse, reconstructed_signal, 'r--', label='Reconstructed Signal')
plt.title('Reconstruction from Impulse Sampled Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.show()

# ------------------------------------------------
# NATURAL SAMPLING
# ------------------------------------------------
fs_nat = 1000
T = 1
t_nat = np.arange(0, T, 1/fs_nat)
fm = 10
message_signal = np.sin(2 * np.pi * fm * t_nat)
pulse_rate = 50
pulse_train = np.zeros_like(t_nat)
pulse_width = int(fs_nat / pulse_rate / 2)

# Create Pulse Train
for i in range(0, len(t_nat), int(fs_nat / pulse_rate)):
    pulse_train[i:i+pulse_width] = 1

nat_signal = message_signal * pulse_train
sampled_signal_nat = nat_signal[pulse_train == 1]
sample_times_nat = t_nat[pulse_train == 1]

# Zero-Order Hold Reconstruction
reconstructed_signal_nat = np.zeros_like(t_nat)
for i, time in enumerate(sample_times_nat):
    index = np.argmin(np.abs(t_nat - time))
    reconstructed_signal_nat[index:index+pulse_width] = sampled_signal_nat[i]

# Lowpass Filter for Natural Sampling
def lowpass_filter(signal, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, signal)

reconstructed_signal_nat = lowpass_filter(reconstructed_signal_nat, 10, fs_nat)

# Plotting Natural Sampling
plt.figure(figsize=(14, 10))

plt.subplot(4, 1, 1)
plt.plot(t_nat, message_signal, label='Original Message Signal')
plt.legend()
plt.grid(True)

plt.subplot(4, 1, 2)
plt.plot(t_nat, pulse_train, label='Pulse Train')
plt.legend()
plt.grid(True)

plt.subplot(4, 1, 3)
plt.plot(t_nat, nat_signal, label='Natural Sampled Signal')
plt.legend()
plt.grid(True)

plt.subplot(4, 1, 4)
plt.plot(t_nat, reconstructed_signal_nat, color='green', label='Reconstructed Signal')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# ------------------------------------------------
# FLAT-TOP (PLATOP) SAMPLING
# ------------------------------------------------
fs_flat = 1000
t_flat = np.arange(0, 1, 1/fs_flat)
f_signal = 5
x_t = np.sin(2 * np.pi * f_signal * t_flat)

fs_sample = 50
T_sample = 1/fs_sample
tau = 0.01  # Pulse width

t_sample = np.arange(0, 1, T_sample)
x_sample = np.sin(2 * np.pi * f_signal * t_sample)

x_flat_top = np.zeros_like(t_flat)

for i in range(len(t_sample)):
    idx = (t_flat >= t_sample[i]) & (t_flat < t_sample[i] + tau)
    x_flat_top[idx] = x_sample[i]

# Low-pass Filter for Reconstruction
def low_pass_filter(signal, cutoff, fs):
    nyquist = fs / 2
    b, a = butter(5, cutoff / nyquist, btype='low')
    return filtfilt(b, a, signal)

x_reconstructed = low_pass_filter(x_flat_top, f_signal * 2, fs_flat)

# Plotting Flat-Top Sampling
plt.figure(figsize=(12, 6))

plt.subplot(3, 1, 1)
plt.plot(t_flat, x_t, 'g', label="Continuous Signal")
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(t_flat, x_flat_top, 'r', label="Flat-Top Sampled Signal")
plt.stem(t_sample, x_sample, 'r', markerfmt="ro", basefmt=" ")
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(t_flat, x_reconstructed, 'b', label="Reconstructed Signal")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

```

# OUTPUT
## Impulse

![image](https://github.com/user-attachments/assets/8fab100e-0d50-403d-bf38-c0e595eabb54)
![image](https://github.com/user-attachments/assets/d44d866f-7160-4f1b-8c9a-30419b257621)
![image](https://github.com/user-attachments/assets/acb0870f-76de-4e64-b254-d475755d295c)
## Natural 
![image](https://github.com/user-attachments/assets/c8e32afe-7266-4f57-bfd1-3f1e6e98276a)
## Flattop
![image](https://github.com/user-attachments/assets/28c4c86d-98ba-48e5-b907-a8e5143ea172)

 
# RESULT / CONCLUSIONS
Thus the sampling of natural, ideal and flattop sampling techniques were analyzed.

