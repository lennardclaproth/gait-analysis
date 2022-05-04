import os
import glob
import pandas as pd
import numpy as np
import numpy.linalg
import matplotlib.pyplot as plt
import scipy.fftpack

cwd = os.getcwd()
path = cwd + r'/Gait/data/'
files = glob.glob(path+'*1_unk_30hz*')
acc = pd.read_csv(files[0], sep=';').drop('index', axis=1)
acc_norm = acc.copy()

sample_freq = 30

for col in acc_norm.columns:
    acc_norm[col] = acc_norm[col] - acc_norm[col].mean()

# vT[0]  - the primary direction of motion (vector norm is 1.0)
# u[:,0] - the coordinate on the 'primary direction of motion'
#          axis for each point in the data (vector norm is 1.0)
# s[0]   - a scale factor that when applied to u and vT gives back
#          the actual magnitudes of the acceleration over the data
#          (so it is a sort of average absolute acceleration over
#          the history).
u, s, vT = numpy.linalg.svd(acc_norm)
acc_norm_1d = u[:, 0]
fft = scipy.fft(acc_norm_1d)
fft_power = abs(fft)
for i in range(len(fft)):
    fft_power[i] *= 2**(-i/float(len(acc_norm_1d)))

# Find the peak of the power spectrum, and the frequency that corresponds
# to that power.
f0_i = numpy.argmax(fft_power[1:]) + 1  # skip the DC component of the signal
freqs = scipy.fftpack.fftfreq(len(acc_norm_1d), 1.0/sample_freq)
f0 = abs(freqs[f0_i])

# Plots
fig, ax = plt.subplots(figsize=(80, 25))
ticks = np.arange(0, len(acc_norm_1d), (f0*sample_freq))
ax.set_xticks(ticks)
ax.xaxis.grid(which='major', linewidth=3)
# ax.plot(acc_norm_1d)

ax.plot(acc.loc[:, 'y'], linewidth=3.0)
# plt.plot(acc_norm)
# plt.plot(fft)
# # plt.plot(fft_power)
# plt.plot(acc.loc[:, 'y'])

plt.show()
