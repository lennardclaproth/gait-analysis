import os
import numpy as np
import pandas as pd
import math
import scipy
import glob
from scipy import signal
from matplotlib import pyplot as plt


def auto_correlate(data):
    res = np.correlate(data, data, mode='full')
    return res[res.size//2:]


avg_cyc_len = []
ticker = 0
cwd = os.getcwd()
path = cwd + r'/Gait/data/'
i_x = 0
i_y = 1
i_z = 2

sampling_rate = 30  # samples per second in CSV data
time_per_sample = 1 / sampling_rate * 1000
start_point = 0

files = glob.glob(path+'*.csv*')

# First loop: calculate average cycles for all data sets
for dataset in files:
    acc_3d = pd.read_csv(dataset, sep=';').drop('index', axis=1)
    acc_3d = acc_3d.values
    y_acc = []
    for r in acc_3d:
        y_acc.append(r[i_y])  # i_y selects vertical acc. signal

    # --- Create normalized array, for every observed value minus mean values
    normalized = []
    for r in y_acc:
        r = r - np.mean(y_acc)
        normalized.append(r)

    # Build low-pass filter for vertical acceleration signal
    b, a = scipy.signal.butter(4, (2*10)/sampling_rate, btype="low", analog=False)

    # Filter the vertical acceleration signal
    normalized_filtered = scipy.signal.filtfilt(b, a, normalized)

    # --- Rewrite total samples to delete start values containing too much irrelevant noise
    normalized = normalized[150:len(normalized)-150]
    normalized_filtered = normalized_filtered[150:len(normalized_filtered)-150]

    # --- Calculate start point value of cycle
    for i, value in enumerate(normalized_filtered[:-1]):
        if value < 0:
            if normalized_filtered[i+1] > 0:
                start_point = i
                break

    # --- Rewrite total samples to start from gait cycle start_point
    normalized = normalized[start_point:]
    normalized_filtered = normalized_filtered[start_point:]

    # Create auto-correlation from function defined at top
    max_lags = round(((len(normalized_filtered)-1)*0.05))  # Use 5% of the auto-correlation data
    lags_array = []
    for i in range(max_lags):
        lags_array.append(i)

    # Auto-correlate normalized values and find peaks
    auto_correlation = auto_correlate(normalized)
    auto_correlation[:] = [x / 1000 for x in auto_correlation]
    auto_correlation_peaks = scipy.signal.find_peaks(auto_correlation)[0]

    # Auto-correlate normalized and filtered values
    auto_correlation_filtered = auto_correlate(normalized_filtered)
    auto_correlation_filtered[:] = [x / 1000 for x in auto_correlation_filtered]
    auto_correlation_filtered = auto_correlation_filtered[:max_lags]  # Shorten filtered auto-corr to 5%

    # Find peaks of filtered auto-correlation data
    auto_correlation_filtered_peaks = scipy.signal.find_peaks(auto_correlation_filtered, distance=20)[0]
    auto_correlation_filtered_peaks = auto_correlation_filtered_peaks[:max_lags]
    auto_correlation_filtered_peaks[0] = 0  # Set first value to 0 to include first data point/peak (corr=1.0)

    # --- Calculate average distance from auto-correlation peaks
    distance = []
    sum_dist = 0
    avg_dist = 0
    for i in range(len(auto_correlation_filtered_peaks)):
        if i+1 == len(auto_correlation_filtered_peaks):
            break
        distance.append(auto_correlation_filtered_peaks[i+1] - auto_correlation_filtered_peaks[i])
        sum_dist += auto_correlation_filtered_peaks[i+1] - auto_correlation_filtered_peaks[i]
    avg_dist = int(np.ceil(sum_dist/len(distance)))

    # -------- Create array with data points per cycle as input for the neural network
    normalized_array = np.asarray(normalized)  # create numpy array from list
    na_2d = np.reshape(normalized_array, (-1, 1))  # reshape 1D array to 2D array, with 1 column
    cycles = math.floor(len(normalized)/avg_dist)  # calculate amount of gait cycles
    cycles_list = list(range(cycles-1))  # create list from cycles for the loop below
    n = 0
    m = avg_dist
    final_arr = na_2d[n:m]  # init array which will get data appended to in the loop below

    n = n + avg_dist
    m = m + avg_dist
    for x in cycles_list:  # For every gait cycle, append all data points that belong to that cycle to a column
        append_arr = na_2d[n:m]
        final_arr = np.append(final_arr, append_arr, axis=1)
        n = n + avg_dist
        m = m + avg_dist

    print(dataset)
    print('array shape:')
    print(final_arr.shape)
    avg_cyc_len.append(final_arr.shape[0])

    average_cycle_length = sum(avg_cyc_len) / len(avg_cyc_len)
    average_cycle_length = math.floor(average_cycle_length)
    print('Average cycle length for all data sets: ' + str(average_cycle_length))

merged_data_array = np.zeros(shape=(average_cycle_length, 0))

# -------------------------------------------------------------------------------------------------
# Second loop: use the average cycles to create an array with all data sets for neural network input
# -------------------------------------------------------------------------------------------------
for dataset in files:
    acc_3d = pd.read_csv(dataset, sep=';').drop('index', axis=1)
    acc_3d = acc_3d.values
    y_acc = []
    for r in acc_3d:
        y_acc.append(r[i_y])  # i_y selects vertical acc. signal

    # --- Create normalized array, for every observed value minus mean values
    normalized = []
    for r in y_acc:
        r = r - np.mean(y_acc)
        normalized.append(r)

    # Build low-pass filter for vertical acceleration signal
    b, a = scipy.signal.butter(4, (2*10)/sampling_rate, btype="low", analog=False)

    # Filter the vertical acceleration signal
    normalized_filtered = scipy.signal.filtfilt(b, a, normalized)

    # --- Rewrite total samples to delete start values containing too much irrelevant noise
    normalized = normalized[150:len(normalized)-150]
    normalized_filtered = normalized_filtered[150:len(normalized_filtered)-150]

    # --- Calculate start point value of cycle
    for i, value in enumerate(normalized_filtered[:-1]):
        if value < 0:
            if normalized_filtered[i+1] > 0:
                start_point = i
                break

    # --- Rewrite total samples to start from gait cycle start_point
    normalized = normalized[start_point:]
    normalized_filtered = normalized_filtered[start_point:]

    # Create auto-correlation from function defined at top
    max_lags = round(((len(normalized_filtered)-1)*0.05))  # Use 5% of the auto-correlation data
    lags_array = []
    for i in range(max_lags):
        lags_array.append(i)

    # Auto-correlate normalized values and find peaks
    auto_correlation = auto_correlate(normalized)
    auto_correlation[:] = [x / 1000 for x in auto_correlation]
    auto_correlation_peaks = scipy.signal.find_peaks(auto_correlation)[0]

    # Auto-correlate normalized and filtered values
    auto_correlation_filtered = auto_correlate(normalized_filtered)
    auto_correlation_filtered[:] = [x / 1000 for x in auto_correlation_filtered]
    auto_correlation_filtered = auto_correlation_filtered[:max_lags]  # Shorten filtered auto-corr to 5%

    # Find peaks of filtered auto-correlation data
    auto_correlation_filtered_peaks = scipy.signal.find_peaks(auto_correlation_filtered, distance=20)[0]
    auto_correlation_filtered_peaks = auto_correlation_filtered_peaks[:max_lags]
    auto_correlation_filtered_peaks[0] = 0  # Set first value to 0 to include first data point/peak (corr=1.0)

    # --- Calculate average distance from auto-correlation peaks
    distance = []
    sum_dist = 0
    avg_dist = 0
    for i in range(len(auto_correlation_filtered_peaks)):
        if i+1 == len(auto_correlation_filtered_peaks):
            break
        distance.append(auto_correlation_filtered_peaks[i+1] - auto_correlation_filtered_peaks[i])
        sum_dist += auto_correlation_filtered_peaks[i+1] - auto_correlation_filtered_peaks[i]
    avg_dist = int(np.ceil(sum_dist/len(distance)))

    # -------- Create plots with overlaid filtered signals (in red)
    time = np.arange(0, normalized.__len__()) * time_per_sample  # time in ms for x-axis
    ticks = np.arange(0, len(normalized_filtered)*time_per_sample, avg_dist*time_per_sample)
    fig, ax = plt.subplots(nrows=3, ncols=1, sharex=False, figsize=(17, 12))

    # ax[0].set_title('Vertical acceleration signal 30hz, normalized and unfiltered')
    # ax[0].grid(color='xkcd:grey', linestyle='--', linewidth=0.5)
    # ax[0].set_xlabel('Time in milliseconds')
    # ax[0].axhline(linewidth=1.0, color='k')
    # ax[0].plot(time, normalized, linewidth=0.3, color='g')
    #
    # ax[1].set_title('Auto-correlation, from normalized and low-pass filtered signal 30hz')
    # ax[1].grid(color='xkcd:grey', linestyle='--', linewidth=0.5)
    # ax[1].set_xlim([0, max_lags])
    # ax[1].axhline(linewidth=1.0, color='k')
    # ax[1].set_xlabel('Lags (30 lags = 1000 milliseconds)')
    # ax[1].set_ylabel('Correlation')
    # ax[1].plot(lags_array, auto_correlation_filtered[:max_lags], linewidth=0.8, color='g')
    # ax[1].scatter(auto_correlation_filtered_peaks, auto_correlation_filtered[auto_correlation_filtered_peaks])
    #
    # ax[2].set_title('Vertical acceleration signal, normalized and low-pass filtered (filtered in red)')
    # ax[2].grid(color='xkcd:grey', linestyle='--', linewidth=0.5)
    # ax[2].set_xticks(ticks)
    # for tick in ax[2].get_xticklabels():
    #     tick.set_rotation(45)
    # ax[2].axhline(linewidth=1.0, color='k')
    # ax[2].set_xlabel('Time in milliseconds')
    # # ax[2].set_xlim([0, 41700])
    # ax[2].plot(time, normalized, linewidth=0.5, color='k')
    # ax[2].plot(time, normalized_filtered, linewidth=0.6, color='r')

    # plt.show()

    # -------- Create array with data points per cycle as input for the neural network
    normalized_array = np.asarray(normalized)  # create numpy array from list
    na_2d = np.reshape(normalized_array, (-1, 1))  # reshape 1D array to 2D array, with 1 column
    cycles = math.floor(len(normalized)/avg_dist)  # calculate amount of gait cycles
    cycles_list = list(range(cycles-1))  # create list from cycles for the loop below
    n = 0
    m = avg_dist
    final_arr = na_2d[n:m]  # init array which will get data appended to in the loop below

    n = n + avg_dist
    m = m + avg_dist
    for x in cycles_list:  # For every gait cycle, append all data points that belong to that cycle to a column
        append_arr = na_2d[n:m]
        final_arr = np.append(final_arr, append_arr, axis=1)
        n = n + avg_dist
        m = m + avg_dist

    print(dataset)
    print('array shape:')
    print(final_arr.shape)
    avg_cyc_len.append(final_arr.shape[0])
    print('initial array length:')
    print(len(final_arr))

    if len(final_arr) > average_cycle_length:
        n = len(final_arr) - 32
        final_arr = final_arr[:-n, :]
        df = pd.DataFrame(final_arr)
        print('new array length:')
        print(len(final_arr))
    elif len(final_arr) < average_cycle_length:
        arr_to_append = np.zeros(shape=(32-len(final_arr), final_arr.shape[1]))
        final_arr = np.concatenate((final_arr, arr_to_append), axis=0)
        print('new array length:')
        print(len(final_arr))
    print(pd.DataFrame(final_arr))

    if ticker < 1:
        merged = np.concatenate((merged_data_array, final_arr), axis=1)
        ticker += 1
    else:
        merged = np.concatenate((merged, final_arr), axis=1)
    print('--------------------')


merged_df = pd.DataFrame(merged)
print(merged_df)

# TODO: CREATE LABEL ARRAY FOR FINAL COMBINED DATA ARRAY
# TODO: TEST BOTH DATA ARRAY AND LABEL ARRAY FOR NEURAL NETWORK INPUT
