import os
import numpy as np
import pandas as pd
import math
import scipy
import time
from collections import Counter
from sklearn.model_selection import train_test_split
from keras.models import Sequential
import keras.optimizers
from keras.utils import to_categorical
import glob
from scipy import signal
from matplotlib import pyplot as plt

start_time = time.time()


def auto_correlate(data):
    res = np.correlate(data, data, mode='full')
    return res[res.size//2:]


def print2(array):
    print(pd.DataFrame(array).to_string())


def norm_data(data):
    normalized = []
    start_point = 0
    for row in data:
        row = row-np.mean(data)
        normalized.append(row)
    normalized = normalized[150:len(normalized)-150]
    for i, value in enumerate(normalized[:-1]):
        if value < 0:
            if normalized[i+1] > 0:
                start_point = i
                break
    normalized = normalized[start_point:]
    return normalized


def filt_data(data):
    sampling_rate = 30
    b, a = scipy.signal.butter(4, (2*10)/sampling_rate, btype="low", analog=False)
    data_filt = scipy.signal.filtfilt(b, a, data)
    return data_filt


def calc_avg_cyc_len(data):
    max_lags = round(((len(data)-1)*0.5))
    auto_correlation = auto_correlate(data)
    auto_correlation[:] = [x/1000 for x in auto_correlation]
    auto_correlation = auto_correlation[:max_lags]
    auto_correlation_peaks = scipy.signal.find_peaks(auto_correlation, distance=20)[0]
    auto_correlation_peaks = auto_correlation_peaks[:max_lags]
    auto_correlation_peaks[0] = 0
    distance = []
    sum_dist = 0
    for i in range(len(auto_correlation_peaks)):
        if i+1 == len(auto_correlation_peaks):
            break
        distance.append(auto_correlation_peaks[i+1] - auto_correlation_peaks[i])
        sum_dist += auto_correlation_peaks[i+1] - auto_correlation_peaks[i]
    avg_dist = int(np.ceil(sum_dist/len(distance)))
    return avg_dist


def datapoints_per_cyc(data, cyc_len):
    data = np.asarray(data)
    data_2d = np.reshape(data, (-1, 1))
    tot_num_cyc = math.floor(len(data)/cyc_len)
    cyc_list = list(range(tot_num_cyc-1))
    n = 0
    m = cyc_len
    fin_data = data_2d[n:m]
    for x in cyc_list:
        append_data = data_2d[n:m]
        fin_data = np.append(fin_data, append_data, axis=1)
        n = n + cyc_len
        m = m + cyc_len
    return fin_data


cwd = os.getcwd()
path = cwd + r'/Gait/data/'
test_path = cwd + r'/Gait/test_data/'
files = glob.glob(path+'*.csv*')
test_files = glob.glob(test_path+'*.csv')
avg_cyc_len_arr = []
class_names = []
sampling_rate = 30  # samples per second in CSV data
time_per_sample = 1 / sampling_rate * 1000
plot = False
plot2 = False
x = 0
y = 1
z = 2

for dataset in files:
    class_names.append(dataset)

# -----------------------------------------------------------------------------
#  The purpose of this loop is only to calculate the avg cycle length of all datasets!
# -----------------------------------------------------------------------------
print('Calculating average cycle length...')
for dataset in files:
    arr_3d = pd.read_csv(dataset, sep=';').drop('index', axis=1).values
    normalized = []
    for row in arr_3d:
        normalized.append(row[1])

    normalized = norm_data(normalized)
    filtered = filt_data(normalized)
    avg_dist = calc_avg_cyc_len(filtered)
    avg_cyc_len_arr.append(avg_dist)
    max_lags = round(((len(filtered)-1)*0.05))  # Use 5% of the auto-correlation data
    auto_correlation_filtered = auto_correlate(filtered)
    auto_correlation_filtered[:] = [x / 1000 for x in auto_correlation_filtered]
    auto_correlation_filtered = auto_correlation_filtered[:max_lags]  # Shorten filtered auto-corr to 5%
    auto_correlation_filtered_peaks = scipy.signal.find_peaks(auto_correlation_filtered, distance=20)[0]
    auto_correlation_filtered_peaks = auto_correlation_filtered_peaks[:max_lags]
    auto_correlation_filtered_peaks[0] = 0  # Set first value to 0 to include first data point/peak (corr=1.0)
    lags_array = []
    for i in range(max_lags):
        lags_array.append(i)
    seconds = np.arange(0, normalized.__len__()) * time_per_sample  # time in ms for x-axis
    ticks = np.arange(0, len(filtered)*time_per_sample, avg_dist*time_per_sample)
    if plot:
        fig, ax = plt.subplots(nrows=3, ncols=1, sharex=False, figsize=(17, 12))
        ax[0].set_title('Vertical acceleration signal 30hz, normalized and unfiltered')
        ax[0].grid(color='xkcd:grey', linestyle='--', linewidth=0.5)
        ax[0].set_xlabel('Time in milliseconds')
        ax[0].axhline(linewidth=1.0, color='k')
        ax[0].plot(seconds, normalized, linewidth=0.3, color='g')
        ax[1].set_title('Auto-correlation, from normalized and low-pass filtered signal 30hz')
        ax[1].grid(color='xkcd:grey', linestyle='--', linewidth=0.5)
        ax[1].set_xlim([0, max_lags])
        ax[1].axhline(linewidth=1.0, color='k')
        ax[1].set_xlabel('Lags (30 lags = 1000 milliseconds)')
        ax[1].set_ylabel('Correlation')
        ax[1].plot(lags_array, auto_correlation_filtered[:max_lags], linewidth=0.8, color='g')
        ax[1].scatter(auto_correlation_filtered_peaks, auto_correlation_filtered[auto_correlation_filtered_peaks])
        ax[2].set_title('Vertical acceleration signal, normalized and low-pass filtered (filtered in red)')
        ax[2].grid(color='xkcd:grey', linestyle='--', linewidth=1)
        ax[2].set_xticks(ticks)
        for tick in ax[2].get_xticklabels():
            tick.set_rotation(45)
        ax[2].axhline(linewidth=1.0, color='k')
        ax[2].set_xlabel('Time in milliseconds')
        ax[2].set_xlim([0, 50000])
        ax[2].plot(seconds, normalized, linewidth=0.5, color='k')
        # ax[2].plot(seconds, filtered, linewidth=0.6, color='r')
        fig.suptitle(str(os.path.basename(os.path.normpath(dataset)[:-4])), x=0.1, fontsize=30)
        plt.show()

sum_cyc = 0
for item in avg_cyc_len_arr:
    sum_cyc += item
avg_cyc_len = int(sum_cyc/len(avg_cyc_len_arr))

print('Average cycle length of all datasets: ' + str(avg_cyc_len))
print('-----')

# -----------------------------------------------------------------------------
# The next part is for preparing the datasets to be input in the neural network
# -----------------------------------------------------------------------------
i = 0
ticker = 0
lbl_ticker = 0
cycle_list_ticker = 0
range_sum = 0
dataset_person = {}
test_dataset_person = {}
person_id = 0
merged_data_array = np.zeros(shape=(avg_cyc_len*2, 0))
test_data_array = np.zeros(shape=(avg_cyc_len, 0))
lbl_arr_init = np.zeros(shape=(2, 0))

for dataset in files:
    in_dataset_person = False
    for id in dataset_person:
        if dataset_person[id] in os.path.basename(os.path.normpath(dataset)[:-4]):
            person_id = id
            in_dataset_person = True
    if not in_dataset_person:
        person_id = len(dataset_person)
        dataset_person[person_id] = os.path.basename(os.path.normpath(dataset)[:-4])
    print(dataset_person[person_id])
    per_test = os.path.basename(os.path.normpath(dataset)[:-4])
    print('Current person: ' + str(dataset_person[person_id]))
    print('Current dataset: ' + str(os.path.normpath(dataset)[:-4]))
    print('This person has id: ' + str(person_id))
    arr_3d = pd.read_csv(dataset, sep=';').drop('index', axis=1).values
    normalized = []
    normalized_z = []
    start_point = 0
    for row in arr_3d:
        normalized.append(row[1])
        normalized_z.append(row[z])
    normalized = norm_data(normalized)
    normalized_z = norm_data(normalized_z)
    final_arr = datapoints_per_cyc(normalized, avg_cyc_len_arr[i])

    normalized_array = np.asarray(normalized)  # create array from list
    normalized_2d = np.reshape(normalized_array, (-1, 1))  # create 2d array from array
    new_normalized_2d = normalized_2d  # clone 2d array for further use
    normalized_2d_avg_len_init = np.zeros(shape=(avg_cyc_len, 0))  # make init array for other arrays to append to
    normalized_2d_avg_len_to_append = normalized_2d[:avg_cyc_len]  # start initial to-append-array with first 32
    normalized_2d_final = normalized_2d_avg_len_init
    while len(new_normalized_2d[avg_cyc_len:]) > (avg_cyc_len - 1):  # stop loop when last iteration is < avg_cyc_len
        # create array to append, with a length of avg_cyc_len (usually 32). First iteration is the first 32 data points
        normalized_2d_avg_len_to_append = new_normalized_2d[:avg_cyc_len]
        # Delete 32 from array to start over next iteration (until no cycle is left)
        new_normalized_2d = new_normalized_2d[avg_cyc_len:]
        # find start point after deleting 32 data points
        start_point = np.where(new_normalized_2d == min(new_normalized_2d[:round(avg_cyc_len*1.1)]))
        try:
            start_point = start_point[0][0]
        except:
            start_point = start_point[0]

        new_normalized_2d = new_normalized_2d[start_point:]  # let new array start from start point
        # Append the array that we prepared to the final array every time until done
        normalized_2d_final = np.append(normalized_2d_final, normalized_2d_avg_len_to_append, axis=1)
    print('Amount of Y cycles in this dataset: ' + str(len(normalized_2d_final[1])))

    normalized_z_array = np.asarray(normalized_z)  # create array from list
    normalized_z_2d = np.reshape(normalized_z_array, (-1, 1))  # create 2d array from array
    new_normalized_z_2d = normalized_z_2d  # clone 2d array for further use
    normalized_2d_z_avg_len_init = np.zeros(shape=(avg_cyc_len, 0))  # make init array for other arrays to append to
    normalized_2d_z_avg_len_to_append = normalized_z_2d[:avg_cyc_len]  # start initial to-append-array with first 32
    normalized_2d_z_final = normalized_2d_z_avg_len_init
    while len(new_normalized_z_2d[avg_cyc_len:]) > (avg_cyc_len - 1):  # stop loop when last iteration is < avg_cyc_len
        # create array to append, with a length of avg_cyc_len (usually 32). First iteration is the first 32 data points
        normalized_2d_z_avg_len_to_append = new_normalized_z_2d[:avg_cyc_len]
        # Delete 32 from array to start over next iteration (until no cycle is left)
        new_normalized_z_2d = new_normalized_z_2d[avg_cyc_len:]
        # find start point after deleting 32 data points
        start_point = np.where(new_normalized_z_2d == min(new_normalized_z_2d[:round(avg_cyc_len*1.1)]))
        try:
            start_point = start_point[0][0]
        except:
            start_point = start_point[0]

        new_normalized_z_2d = new_normalized_z_2d[start_point:]  # let new array start from start point
        # Append the array that we prepared to the final array every time until done
        normalized_2d_z_final = np.append(normalized_2d_z_final, normalized_2d_z_avg_len_to_append, axis=1)
    print('Amount of Z cycles in this dataset: ' + str(len(normalized_2d_z_final[1])))

    if len(normalized_2d_final[2]) > len(normalized_2d_z_final[2]):
        normalized_2d_final = np.delete(normalized_2d_final, np.s_[(len(normalized_2d_z_final[1])):(len(normalized_2d_final[1]))], axis=1)
    elif len(normalized_2d_final[2]) < len(normalized_2d_z_final[2]):
        normalized_2d_z_final = np.delete(normalized_2d_z_final, np.s_[(len(normalized_2d_final[1])):(len(normalized_2d_z_final[1]))], axis=1)
    normalized_2d_final_combined = np.append(normalized_2d_final, normalized_2d_z_final, axis=0)

    print('Amount of combined Y and Z cycles in this dataset: ' + str(len(normalized_2d_final_combined[1])))

    # Create a (2,x) shape 2d array that marks the person_id with all of their cycles
    cycles = len(normalized_2d_final_combined[1])  # amount of cycles in dataset

    if cycle_list_ticker < 1:  # initialize the cycle list
        cycle_list = [x for x in range(0, cycles)]
        range_sum = range_sum + cycles
        print('total range_sum' + str(range_sum))
        cycle_list_ticker += 1
    else:
        cycle_list = [x for x in range(range_sum, range_sum + cycles)]  # adds the range of cycles so it does not reset
        range_sum = range_sum + cycles
        print('total range_sum' + str(range_sum))

    person_id_list = [person_id] * len(normalized_2d_final_combined[1])  # create a person_id list with equal length to amount of cycles
    cycle_array = np.array(cycle_list)  # convert list to array
    cycle_array = np.reshape(cycle_array, (-1, len(normalized_2d_final_combined[1])))  # reshape to 1 row with cycle amount columns
    cycle_array = np.vstack([cycle_array, person_id_list])  # concatenate cycles with person_id array
    if lbl_ticker < 1:  # init 2D lbl array with first dataset
        lbl = np.concatenate((lbl_arr_init, cycle_array), axis=1)
        lbl_ticker += 1
    else:  # append new datasets to initialized list every time
        lbl = np.concatenate((lbl, cycle_array), axis=1)
    # print('lbl array length: ' + str(lbl.shape[1]))

    if ticker < 1:
        features = np.concatenate((merged_data_array, normalized_2d_final_combined), axis=1)
        ticker += 1
    else:
        features = np.concatenate((features, normalized_2d_final_combined), axis=1)

    print('shape features: ' + str(np.shape(features)))
    print('shape labels: ' + str(np.shape(lbl)))
    print('person id shape: ' + str(np.shape(person_id_list)))

    seconds = np.arange(0, (normalized.__len__())) * time_per_sample  # time in ms for x-axis
    seconds_z = np.arange(0, (normalized_z.__len__())) * time_per_sample
    seconds_comb = np.arange(0, (len(normalized)+len(normalized_z))) * time_per_sample
    ticks = np.arange(0, len(normalized)*time_per_sample, avg_cyc_len*time_per_sample)
    ticks_z = np.arange(0, len(normalized_z)*time_per_sample, avg_cyc_len*time_per_sample)
    ticks_comb = np.arange(0, (len(normalized_z)+len(normalized))*time_per_sample, avg_cyc_len*time_per_sample)
    if plot2:
        fig, ax = plt.subplots(nrows=5, ncols=1, sharex=True, figsize=(20, 15))
        ax[0].set_title('Vertical acceleration signal 30hz, normalized and unfiltered')
        ax[0].grid(color='xkcd:grey', linestyle='--', linewidth=1.5)
        ax[0].set_xlabel('Time in milliseconds')
        ax[0].axhline(linewidth=1.0, color='k')
        ax[0].plot(seconds[:len(normalized)], normalized, linewidth=1, color='k')

        ax[1].set_title('Z acceleration signal 30hz, normalized and unfiltered')
        ax[1].set_xlabel('Time in milliseconds')
        ax[1].axhline(linewidth=1.0, color='k')
        ax[1].plot(seconds_z[:len(normalized_z)], normalized_z, linewidth=1, color='g')
        ax[1].grid(color='xkcd:grey', linestyle='--', linewidth=1)
        ax[1].set_xticks(ticks_z)
        for tick in ax[1].get_xticklabels():
            tick.set_rotation(45)

        ax[2].set_title('Adjusted cycles in vertical acc. signal')
        ax[2].grid(color='xkcd:grey', linestyle='--', linewidth=1.5)
        ax[2].set_xticks(ticks)
        for tick in ax[1].get_xticklabels():
            tick.set_rotation(45)
        ax[2].axhline(linewidth=1.0, color='k')
        ax[2].set_xlabel('Time in milliseconds')
        ax[2].plot(seconds[:len(normalized_2d_final.transpose().flatten())], normalized_2d_final.transpose().flatten(), linewidth=1, color='k')

        ax[3].set_title('Adjusted cycles in Z acc. signal')
        ax[3].grid(color='xkcd:grey', linestyle='--', linewidth=1.5)
        ax[3].set_xticks(ticks_z)
        ax[3].axhline(linewidth=1.0, color='k')
        ax[3].set_xlabel('Time in milliseconds')
        for tick in ax[3].get_xticklabels():
            tick.set_rotation(45)
        ax[3].plot(seconds_z[:len(normalized_2d_z_final.transpose().flatten())], normalized_2d_z_final.transpose().flatten(), linewidth=1, color='g')

        ax[4].set_title('Y + Z adjusted signals combined')
        ax[4].grid(color='xkcd:grey', linestyle='--', linewidth=1.5)
        ax[4].set_xticks(ticks_comb)
        ax[4].axhline(linewidth=1.0, color='k')
        ax[4].set_xlabel('Time in milliseconds')
        for tick in ax[4].get_xticklabels():
            tick.set_rotation(45)
        ax[4].plot(seconds_comb[:len(normalized_2d_final_combined.transpose().flatten())],
                   normalized_2d_final_combined.transpose().flatten(), linewidth=1, color='r')

        ax[4].set_xlim([0, 30000])
        fig.suptitle(str(os.path.basename(os.path.normpath(dataset)[:-4])), x=0.1, fontsize=40)
        plt.show()

    i += 1
    person_id += 1
    print('-----')

# Split data into test and training data
train_features = np.transpose(features)  # permute the dimensions of lbl (swap columns and rows)
train_lbl = np.delete(lbl, 0, axis=0)  # delete from lbl array: item index 0, axis=0 equals row, axis=1 equals column
train_lbl = np.transpose(train_lbl)  # permute the dimensions of lbl (swap columns and rows)
train_lbl = to_categorical(train_lbl)  # One-hot encode lbl data

# Neural network
epochs = 150
batch_size = 5
loss = 'categorical_crossentropy'  # mean_squared_error / categorical_crossentropy
test_size = 0.20  # percentage of data that will be used for testing
sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
optimizer = sgd  # adam / rmsprop / adagrad / adadelta / adamax / nadam /
x_train, x_test, y_train, y_test = train_test_split(train_features, train_lbl, test_size=test_size)
model = Sequential()
model.add(keras.layers.Dense(64, activation='relu', input_shape=(features.shape[0],)))
model.add(keras.layers.Dropout(0.20))
model.add(keras.layers.Dense(48, activation='relu'))
model.add(keras.layers.Dropout(0.20))
model.add(keras.layers.Dense(24, activation='relu'))
model.add(keras.layers.Dense(train_lbl.shape[1], activation='softmax'))
model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
print(model.summary())
model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
# model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
scores = model.evaluate(x_test, y_test)
print("-----------------------------------")
print("%s: %.2f%%" % ('Accuracy', scores[1]*100))
print("-----------------------------------")


persons = pd.DataFrame(data=dataset_person, index=[0])
persons = persons.transpose()
persons.columns = ['person']
test_person_id = 0

for test_dataset in test_files:
    test_dataset_person[test_person_id] = os.path.basename(os.path.normpath(test_dataset)[:-4])
    print('Current dataset: ' + str(test_dataset_person[test_person_id]))
    arr_3d = pd.read_csv(test_dataset, sep=';').drop('index', axis=1).values
    normalized = []
    normalized_z = []
    start_point = 0
    for row in arr_3d:
        normalized.append(row[1])
        normalized_z.append(row[z])
    normalized = norm_data(normalized)
    normalized_z = norm_data(normalized_z)

    # Only if cyc_len for dataset is higher than avg_cyc_len, execute code below:
    normalized_array = np.asarray(normalized)  # create array from list
    normalized_2d = np.reshape(normalized_array, (-1, 1))  # create 2d array from array
    new_normalized_2d = normalized_2d  # clone 2d array for further use
    normalized_2d_avg_len_init = np.zeros(shape=(avg_cyc_len, 0))  # make init array for other arrays to append to
    normalized_2d_avg_len_to_append = normalized_2d[:avg_cyc_len]  # start initial to-append-array with first 32
    normalized_2d_final = normalized_2d_avg_len_init
    while len(new_normalized_2d[avg_cyc_len:]) > (avg_cyc_len - 1):  # stop loop when last iteration is < avg_cyc_len
        # create array to append, with a length of avg_cyc_len (usually 32). First iteration is the first 32 data points
        normalized_2d_avg_len_to_append = new_normalized_2d[:avg_cyc_len]
        # Delete 32 from array to start over next iteration (until no cycle is left)
        new_normalized_2d = new_normalized_2d[avg_cyc_len:]
        # find start point after deleting 32 data points
        start_point = np.where(new_normalized_2d == min(new_normalized_2d[:round(avg_cyc_len*1.1)]))
        try:
            start_point = start_point[0][0]
        except:
            start_point = start_point[0]

        new_normalized_2d = new_normalized_2d[start_point:]  # let new array start from start point
        # Append the array that we prepared to the final array every time until done
        normalized_2d_final = np.append(normalized_2d_final, normalized_2d_avg_len_to_append, axis=1)

    # Repeat above code for Z data
    normalized_z_array = np.asarray(normalized_z)  # create array from list
    normalized_z_2d = np.reshape(normalized_z_array, (-1, 1))  # create 2d array from array
    new_normalized_z_2d = normalized_z_2d  # clone 2d array for further use
    normalized_2d_z_avg_len_init = np.zeros(shape=(avg_cyc_len, 0))  # make init array for other arrays to append to
    normalized_2d_z_avg_len_to_append = normalized_z_2d[:avg_cyc_len]  # start initial to-append-array with first 32
    normalized_2d_z_final = normalized_2d_z_avg_len_init
    while len(new_normalized_z_2d[avg_cyc_len:]) > (avg_cyc_len - 1):  # stop loop when last iteration is < avg_cyc_len
        # create array to append, with a length of avg_cyc_len (usually 32). First iteration is the first 32 data points
        normalized_2d_z_avg_len_to_append = new_normalized_z_2d[:avg_cyc_len]
        # Delete 32 from array to start over next iteration (until no cycle is left)
        new_normalized_z_2d = new_normalized_z_2d[avg_cyc_len:]
        # find start point after deleting 32 data points
        start_point = np.where(new_normalized_z_2d == min(new_normalized_z_2d[:round(avg_cyc_len*1.1)]))
        try:
            start_point = start_point[0][0]
        except:
            start_point = start_point[0]

        new_normalized_z_2d = new_normalized_z_2d[start_point:]  # let new array start from start point
        # Append the array that we prepared to the final array every time until done
        normalized_2d_z_final = np.append(normalized_2d_z_final, normalized_2d_z_avg_len_to_append, axis=1)

    print('Amount of Y cycles in this dataset: ' + str(len(normalized_2d_final[1])))
    print('Amount of Z cycles in this dataset: ' + str(len(normalized_2d_z_final[1])))

    # Resize Y data to match shape with Z data
    if len(normalized_2d_final[2]) > len(normalized_2d_z_final[2]):
        normalized_2d_final = np.delete(normalized_2d_final, np.s_[(len(normalized_2d_z_final[1])):(len(normalized_2d_final[1]))], axis=1)
    elif len(normalized_2d_final[2]) < len(normalized_2d_z_final[2]):
        normalized_2d_z_final = np.delete(normalized_2d_z_final, np.s_[(len(normalized_2d_final[1])):(len(normalized_2d_z_final[1]))], axis=1)
    normalized_2d_final_combined = np.append(normalized_2d_final, normalized_2d_z_final, axis=0)

    print('Amount of combined Y and Z cycles in this dataset: ' + str(len(normalized_2d_final_combined[1])))

    test_len = calc_avg_cyc_len(normalized)
    sum = 0
    normalized_2d_final_combined = np.transpose(normalized_2d_final_combined)
    predict = model.predict_classes(normalized_2d_final_combined)  # OLD: final_arr
    predict_counter = Counter(predict)
    for count in predict_counter.values():
        sum = sum + count
    predict_counter = pd.DataFrame(predict_counter, index=[0])
    predict_counter = predict_counter.transpose()
    predict_counter.columns = ['Cycles']

    for index in predict_counter.index.values:
        predict_counter.rename(index={index: persons.loc[index, 'person']}, inplace=True)

    predict_cycle_list = predict_counter.values.tolist()
    predict_cycle_list = [item for sublist in predict_cycle_list for item in sublist]

    for i, row in predict_counter.iterrows():
        predict_counter.at[i, 'Percentage'] = round((row[0] / sum) * 100, 1)

    predict_counter['Percentage'] = predict_counter['Percentage'].astype(str) + '%'

    print('Predicted cycles per person for this dataset ' + '(' + str(test_dataset_person[test_person_id]) + '):')
    print(predict_counter.to_string())
    print('-----------------------------------')
    test_person_id += 1


end_time = time.time()
print("Script duration: " + str(round(end_time - start_time)) + " seconds")
