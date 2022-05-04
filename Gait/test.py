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
    result = np.correlate(data, data, mode='full')
    return result[result.size//2:]

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
                start_point = 1
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

def datapoints_per_cyc(dataset, avg_cyc_len, cyc_len):
    normalized = np.asarray(dataset)
    normalized_2d = np.reshape(normalized, (-1,1))
    normalized_2d_clone = normalized_2d
    normalized_2d_avg_len_init = np.zeros(shape=(avg_cyc_len, 0))
    normalized_2d_avg_len_to_append = normalized_2d[:avg_cyc_len]
    normalized_2d_final = normalized_2d_avg_len_init
    while len(normalized_2d_clone[avg_cyc_len:]) > (avg_cyc_len - 1):
        normalized_2d_avg_len_to_append = normalized_2d_clone[:avg_cyc_len]
        normalized_2d_clone = normalized_2d_clone[avg_cyc_len:]
        start_point = np.where(normalized_2d_clone == min(normalized_2d_clone[:round(avg_cyc_len*1.1)]))
        try:
            start_point = start_point[0][0]
        except:
            start_point = start_point[0]

        normalized_2d_clone = normalized_2d_clone[start_point:]
        normalized_2d_final = np.append(normalized_2d_final, normalized_2d_avg_len_to_append, axis=1)    
    return normalized_2d_final

def datapoints_per_cyc2(dataset, avg_cyc_len, cyc_len):
    normalized = np.asarray(dataset)
    normalized_2d = np.reshape(normalized, (-1,1))
    normalized_2d_clone = normalized_2d
    normalized_2d_avg_len_init = np.zeros(shape=(avg_cyc_len, 0))
    normalized_2d_avg_len_to_append = normalized_2d[:avg_cyc_len]
    normalized_2d_final = normalized_2d_avg_len_init
    while len(normalized_2d_clone[cyc_len:]) > (cyc_len - 1):
        if cyc_len >= avg_cyc_len:
            normalized_2d_avg_len_to_append = normalized_2d_clone[:avg_cyc_len]
        else:
            arr_concat = np.zeros(shape=(avg_cyc_len - cyc_len, 1))
            # print(np.shape(arr_concat))
            # normalized_2d_clone = normalized_2d_clone[cyc_len:]
            # print(np.shape(normalized_2d_clone))
            normalized_2d_avg_len_to_append = np.concatenate((normalized_2d_clone[:cyc_len], arr_concat), axis=0)
            # print('changing shape from: ' + str(np.shape(normalized_2d_clone)) + ' to: ' + str(np.shape(normalized_2d_avg_len_to_append)))
            # print(pd.DataFrame(normalized_2d_avg_len_to_append))
        normalized_2d_clone = normalized_2d_clone[cyc_len:]
        start_point = np.where(normalized_2d_clone == min(normalized_2d_clone[:round(cyc_len*1.1)]))
        try:
            start_point = start_point[0][0]
        except:
            start_point = start_point[0]

        normalized_2d_clone = normalized_2d_clone[start_point:]
        normalized_2d_final = np.append(normalized_2d_final, normalized_2d_avg_len_to_append, axis=1)    
    return normalized_2d_final
        

def combine_y_z(data_y, data_z):
    if len(data_y[2]) > len(data_z[2]):
        data_y = np.delete(data_y, np.s_[(len(data_z[1])):(len(data_y[1]))], axis=1)
    elif len(data_y[2]) < len(data_z[2]):
        data_z = np.delete(data_z, np.s_[(len(data_y[1])):(len(data_z[1]))], axis=1)
    data_y_z = np.append(data_y, data_z, axis=0)
    return data_y_z

def prep_labels(merged_data, labels, person_id):
    num_cyc = len(merged_data[1])
    person_id_list = [person_id] * len(merged_data[1])
    cycle_list = [x for x in range(0, num_cyc)]
    cycle_array = np.array(cycle_list)
    cycle_array = np.reshape(cycle_array, (-1, num_cyc))
    cycle_array = np.vstack([cycle_array, person_id_list])
    labels = np.concatenate((labels, cycle_array), axis=1)
    return labels

def prep_features(merged_data, features):
    features = np.concatenate((features, merged_data), axis=1)
    return features

def train_model(features, labels, avg_cyc_len):
    train_features = np.transpose(features)
    train_lbl = np.delete(labels, 0, axis=0)
    train_lbl = np.transpose(train_lbl)
    train_lbl = to_categorical(train_lbl)
    epochs = 150
    batch_size = 5
    loss = 'categorical_crossentropy'
    test_size = 0.15
    sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    optimizer = sgd
    x_train, x_test, y_train, y_test = train_test_split(train_features, train_lbl, test_size=test_size)
    model = Sequential()
    model.add(keras.layers.Dense(avg_cyc_len*2, activation='relu', input_shape=(features.shape[0],)))
    model.add(keras.layers.Dropout(0.20))
    model.add(keras.layers.Dense((avg_cyc_len*2-10), activation='relu'))
    model.add(keras.layers.Dropout(0.20))
    model.add(keras.layers.Dense((avg_cyc_len*2-35), activation='relu'))
    model.add(keras.layers.Dense(train_lbl.shape[1], activation='softmax'))
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    print(model.summary())
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
    scores = model.evaluate(x_test, y_test)
    print("-----------------------------------")
    print("%s: %.2f%%" % ('Accuracy', scores[1]*100))
    print("-----------------------------------")
    return model

def predict(model, data, persons):
    sum = 0
    data = np.transpose(data)
    predict = model.predict_classes(data)
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

    if int(max(predict_counter['Percentage'])) < 60:
        print('##################')
        print('Validation failed.')
        print('##################')

    predict_counter['Percentage'] = predict_counter['Percentage'].astype(str) + '%'
    return predict_counter



cwd = os.getcwd()
path = cwd + r'/Gait/data/'
test_path = cwd + r'/Gait/test_data/'
files = glob.glob(path+'*.csv*')
test_files = glob.glob(test_path+'*.csv')
avg_cyc_len_arr = []
sampling_rate = 30
time_per_sample = 1 / sampling_rate * 1000

print('Calculating average cycle length for all datasets...')
#---------------------------------------------------------------------------------------#
# This for loop is responsible for calculating the avg cycle length of all the datasets #
#---------------------------------------------------------------------------------------#
for dataset in files:
    arr_3d = pd.read_csv(dataset, sep=';').drop('index', axis=1).values
    normalized = []
    for row in arr_3d:
        normalized.append(row[1])
    normalized = norm_data(normalized)
    filtered = filt_data(normalized)
    avg_dist = calc_avg_cyc_len(filtered)
    avg_cyc_len_arr.append(avg_dist)

sum_cyc = 0
for item in avg_cyc_len_arr:
    sum_cyc += item
avg_cyc_len = int(sum_cyc/len(avg_cyc_len_arr))
#---------------------------------------------------------------------------------------#
print('Average cycle length per dataset: ' + str(avg_cyc_len_arr))
print('Average cycle length of all datasets: ' + str(avg_cyc_len))
print('Smallest cycle length of all datasets: ' + str(min(avg_cyc_len_arr)))
print('-----')

# avg_cyc_len = min(avg_cyc_len_arr)
range_sum = 0
dataset_person = {}
test_dataset_person = {}
person_id = 0
features = np.zeros(shape=(avg_cyc_len*2, 0))
labels = np.zeros(shape=(2, 0))
y = 1
z = 2
#-------------------------------------------------------------------------------------#
# This for loop is responsible for preparing the data for training the neural network #
#-------------------------------------------------------------------------------------#
for dataset in files:
    in_dataset_person = False
    for id in dataset_person:
        name_no_number = []
        for char in dataset_person[id]:
            if char.isalpha():  # Checks if character in string is a letter
                name_no_number.append(char)
        name_no_number = "".join(name_no_number)  # Creates string out of list of letters
        if name_no_number in os.path.basename(os.path.normpath(dataset)[:-4]):
            person_id = id
            in_dataset_person = True
    if not in_dataset_person:
        person_id = len(dataset_person)
        dataset_person[person_id] = os.path.basename(os.path.normpath(dataset)[:-4])
    print('Current person: ' + str(dataset_person[person_id]))
    print('Current dataset: ' + str(os.path.basename(os.path.normpath(dataset)[:-4])))
    print('This person has id: ' + str(person_id))
    arr_3d = pd.read_csv(dataset, sep=';').drop('index', axis=1).values
    normalized_y = []
    normalized_z = []
    for row in arr_3d:
        normalized_y.append(row[y])
        normalized_z.append(row[z])
    normalized_y = norm_data(normalized_y)
    normalized_z = norm_data(normalized_z)
    filtered_y = filt_data(normalized_y)
    cycles_y = datapoints_per_cyc2(normalized_y, avg_cyc_len, calc_avg_cyc_len(filtered_y))
    cycles_z = datapoints_per_cyc2(normalized_z, avg_cyc_len, calc_avg_cyc_len(filtered_y))
    combined_y_z = combine_y_z(cycles_y, cycles_z)
    labels = prep_labels(combined_y_z, labels, person_id)
    features = prep_features(combined_y_z, features)
    print('Amount of Y cycles in this dataset: ' + str(len(cycles_y[1])))
    print('Amount of Z cycles in this dataset: ' + str(len(cycles_z[1])))
    print('Amount of combined Y and Z cycles in this dataset: ' + str(len(combined_y_z[1])))
    print('Shape of combined data: ' + str(np.shape(combined_y_z)))
    print('Shape of labels: ' + str(np.shape(labels)))
    print('Shape of features: ' + str(np.shape(features)))
    print('-----')
#-------------------------------------------------------------------------------------#
model = train_model(features, labels, avg_cyc_len)
class_names = pd.DataFrame(data=dataset_person, index=[0])
class_names = class_names.transpose()
class_names.columns = ['person']
#---------------------------------------------------------------------------------------------------------------#
# This for loop is responsible for validating the persons in the dataset by feeding new datasets into the model #
#---------------------------------------------------------------------------------------------------------------#
for dataset in test_files:
    print('Current dataset: ' + str(os.path.basename(os.path.normpath(dataset)[:-4])))
    arr_3d = pd.read_csv(dataset, sep=';').drop('index', axis=1).values
    normalized_y = []
    normalized_z = []
    for row in arr_3d:
        normalized_y.append(row[y])
        normalized_z.append(row[z])
    normalized_y = norm_data(normalized_y)
    normalized_z = norm_data(normalized_z)
    filtered_y = filt_data(normalized_y)
    cycles_y = datapoints_per_cyc2(normalized_y, avg_cyc_len, calc_avg_cyc_len(filtered_y))
    cycles_z = datapoints_per_cyc2(normalized_z, avg_cyc_len, calc_avg_cyc_len(filtered_y))
    combined_y_z = combine_y_z(cycles_y, cycles_z)
    predictions = predict(model, combined_y_z, class_names)
    print('Amount of Y cycles in this dataset: ' + str(len(cycles_y[1])))
    print('Amount of Z cycles in this dataset: ' + str(len(cycles_z[1])))
    print('Amount of combined Y and Z cycles in this dataset: ' + str(len(combined_y_z[1])))
    print('Predicted cycles per person for this dataset ' + '(' + str(os.path.basename(os.path.normpath(dataset)[:-4]) + '):'))
    print(predictions.to_string())
    print('-----------------------------------')
#-----------------------------------------------------------------------------------------------------------#
end_time = time.time()
print("Script duration: " + str(round(end_time - start_time)) + " seconds")
