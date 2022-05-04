import os
import numpy as np
import pandas as pd
import peakutils
from peakutils.plot import plot as pplot
from matplotlib import pyplot as plt
from pandas.plotting import lag_plot, autocorrelation_plot
import scipy as sp

cwd = os.getcwd()
path = cwd + '/sensormotion/'
files = []
i_x = 0
i_y = 1
i_z = 2
com_force_arr = []
vert_acc_arr = []
scaled_values = []

# r=root, d=directories, f = files
for r, d, f in os.walk(cwd):
    for file in f:
        if '.csv' in file:
            files.append(os.path.join(r, file))

# Create combined signal from x,y,z signals
# for f in files:
#     force = pd.read_csv(f, sep=';')
#     force = force.drop('index', axis=1)
#     force = force.values
#     for r in force:
#         com_force = np.arcsin(r[i_z]/(np.sqrt(np.square(r[i_x]) + np.square(r[i_y]) + np.square(r[i_z]))))
#         com_force_arr.append(com_force)
# weights = np.ones_like(com_force_arr)/float(len(com_force_arr))
# plt.hist(com_force_arr, weights=weights)
# plt.title("histogram")
# plt.show()

# Extract vertical acceleration signal
for f in files:
    force = pd.read_csv(f, sep=';')
    force = force.drop('index', axis=1)
    force = force.values
    for r in force:
        vert_acc_arr.append(r[i_y])

# Calculate mean value of vertical acceleration values
vert_acc_mean = np.mean(vert_acc_arr)

# Create new array with scaled values, for every observed value - mean values
for r in vert_acc_arr:
    r = r - vert_acc_mean
    scaled_values.append(r)

starting_point = 0
for i, value in enumerate(scaled_values[:-1]):
    if value < 0:
        if scaled_values[i+1] > 0:
            starting_point = value
            break

scaled_values_series = pd.Series(scaled_values)
# corr = scaled_values_series.autocorr(lag=1000)
# print(corr)
pd.plotting.autocorrelation_plot(scaled_values_series)
# pd.plotting.lag_plot(scaled_values_series)
# sp.stats.ttest_ind()
# plt.plot(scaled_values)
# plt.show()

#  ---- VOORBEELD CODE
centers = (30.5, 72.3)
x = np.linspace(0, 120, 121)
y = (peakutils.gaussian(x, 5, centers[0], 3) +
    peakutils.gaussian(x, 7, centers[1], 10) +
    np.random.rand(x.size))
plt.figure(figsize=(10,6))
plt.plot(x, y)
plt.title("Data with noise")

indexes = peakutils.indexes(y, thres=0.5, min_dist=30)
print(indexes)
print(x[indexes], y[indexes])
plt.figure(figsize=(10,6))
pplot(x, y, indexes)
plt.title('First estimate')
plt.show()
# ----- EINDE VOORBEELD CODE

indexes = peakutils.indexes(scaled_values, thres=0.02/max(scaled_values))
scaled_values = np.asarray(scaled_values)
plt.plot(scaled_values)
plt.plot(scaled_values[indexes], marker="o", ms=3)
# pplot(scaled_values, indexes)
plt.show()
