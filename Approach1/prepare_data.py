__author__ = 'Zeynab'

import numpy as np
import pandas
from sklearn.preprocessing import MinMaxScaler
from biosppy.signals.tools import smoother
import csv


from tools.r_detection import detect_r_points

min_range = -50
max_range = 50
total_length = 60000


#sequenced features
def load_sample(path, name, real_label, sampling_rate, dimension1, dimension2, train=True):
    x_signal, y_signal = read_sample(path, name)
    length = y_signal.shape[0]
    rpeaks = detect_r_points(y_signal, length, sampling_rate)

    sample_x = np.empty([0, dimension1, dimension2])
    sample_y = np.empty([0, 1])
    if real_label == 'Normal':
        label = 0
    else:
        label = 1
    if (real_label == 'Normal') | (train is False):
        for index in range(0, len(rpeaks)-dimension1, dimension1):
            x = np.empty([dimension1, dimension2])
            for i in range(dimension1):
                x[i] = y_signal[rpeaks[i + index]:rpeaks[i + index] + dimension2,0]
            y = np.array(label)
            sample_x = np.append(sample_x, np.array([x]), axis=0)
            sample_y = np.append(sample_y, np.array(label))
    else:
        problems = analyze_rpoints(y_signal, rpeaks, sampling_rate)
        for p in problems:
            problem = int(p)
            for index in range(problem, problem + dimension1):
                if (index > 0) & (index + dimension1 < rpeaks.shape[0]):
                    x = np.empty([dimension1, dimension2])
                    for i in range(dimension1):
                        x[i] = y_signal[rpeaks[i + index]:rpeaks[i + index] + dimension2,0]
                    sample_x = np.append(sample_x, np.array([x]), axis=0)
                    sample_y = np.append(sample_y, np.array(label))
    sample_y = np.reshape(sample_y, [sample_y.shape[0],1])
    return sample_x, sample_y


def read_sample(path, name):
    try:
        #print('1')
        #signal = np.loadtxt(path+name+'.txt', skiprows=5)
        dataset = pandas.read_csv(path + name + '.txt', delimiter='\t', skiprows=4)
    except:
        print('Error in reading file!')
    #print('2')
    dataset = pandas.read_csv(path + name + '.txt', delimiter='\t', skiprows=4)

    x_signal = dataset.values[:, 0]
    y = dataset.values[:, 1]
    y_signal = normalize_data(pandas.DataFrame(y), max_range, min_range)
    smoothed_signal,params = smoother(y_signal[:,0])
    dataset =dataset.as_matrix()
    #dataset[:,0] = smoothed_signal
    return x_signal, y_signal


#Normalize data in specified range
def normalize_data(dataset, max=max_range, min=min_range):
    scaler = MinMaxScaler(feature_range=(min, max))
    data = scaler.fit_transform(dataset)
    #move to fit baseline to zero
    index = np.where(dataset == 0)
    value = data[index[0][0]]
    data = data-value
    return data


def load_file(path, file):
    paths = []
    names = []
    sampling_rates = []
    labels = []
    with open(path + file) as csvfile:
        readCSV = csv.reader(csvfile)
        next(readCSV)
        for row in readCSV:
            path = row[0]
            name = row[1]
            sampling_rate = row[2]
            label = row[3]
            paths.append(path)
            names.append(name)
            sampling_rates.append(int(sampling_rate))
            labels.append(label)
    print(file + 'is read!')
    return paths,names,sampling_rates,labels


def load_base_features(x_signal, y_signal, sampling_rate, real_label, rpeaks):
    base_features = np.empty([0, 1])
    length = y_signal.shape[0]

    total_seconds = length/sampling_rate
    total_beats = rpeaks.shape[0]
    bps_total = total_beats/total_seconds
    bpm_total = bps_total * 60

    base_features = simple_concatenate(base_features, np.array([bpm_total]))

    rr_interval = np.empty([0,1])
    for j in range(0, rpeaks.shape[0]-1):
        rr_interval = simple_concatenate(rr_interval, np.array([(rpeaks[j + 1] - rpeaks[j])/sampling_rate]))
    rr_interval = np.reshape(rr_interval, [1,rr_interval.shape[0]])
    rr_average = np.average(rr_interval)
    rr_variance = np.var(rr_interval)

    #handy_features = simple_concatenate(handy_features, rr_interval[0,: num_beats])
    base_features = simple_concatenate(base_features, np.array([rr_average]))
    base_features = simple_concatenate(base_features, np.array([rr_variance]))


    r_values = y_signal[rpeaks[:,0]]
    r_values_average = np.average(abs(r_values))
    r_values_var = np.var(abs(r_values))
    r_values = np.reshape(r_values, [1, r_values.shape[0]])

    #handy_features = simple_concatenate(handy_features, r_values[0,: num_beats])
    base_features = simple_concatenate(base_features, np.array([r_values_average]))
    base_features = simple_concatenate(base_features, np.array([r_values_var]))
    print('base features     ->  %d' %base_features.shape[0])
    base_features = np.reshape(base_features, [1, base_features.shape[0]])
    return base_features


def load_window_features(x_signal, y_signal, sampling_rate, real_label, rpeaks, start_index, num_beats):
    window_features = np.empty([0, 1])
    length = y_signal.shape[0]
    rr_interval = np.empty([0, 1])
    for j in range(start_index, start_index + num_beats):
        rr_interval = simple_concatenate(rr_interval, np.array([(rpeaks[j + 1] - rpeaks[j])/sampling_rate]))
    r_values = y_signal[rpeaks[start_index:start_index + num_beats, 0]]

    window_features = simple_concatenate(window_features, rr_interval)
    window_features = simple_concatenate(window_features, r_values)

    print('Window features  -> %d' %window_features.shape[0])
    window_features = np.reshape(window_features, [1, window_features.shape[0]])
    return window_features


def analyze_rpoints(y_signal, rpeaks, sampling_rate):
    print('in analyse  %d'%rpeaks.shape[0])
    problems = np.empty([0, 1])
    #r-r interval
    rr_interval = np.empty([0, 1])
    for j in range(0, rpeaks.shape[0]-1):
        #rr_interval = simple_concatenate(rr_interval,np.array((rpeaks[j + 1] - rpeaks[j])/sampling_rate))
        rr_interval = np.append(rr_interval, np.array((rpeaks[j + 1] - rpeaks[j])/sampling_rate))
        #rr_interval.append((rpeaks[j + 1] - rpeaks[j])/sampling_rate)
    rr_average = np.average(rr_interval)
    rr_variance = np.var(rr_interval)
    #normal_mu, normal_std = norm.fit(normal_rmse)
    #rr_difference = rr_interval - rr_variance -> to calculate r-r avg from a clean and standard part not all of signal
    rr_problems = np.empty([0, 1])
    for j, r in enumerate(rr_interval):
        distance = abs(r - rr_average)
        if distance >= rr_average/2:
            type = 'R-R problem'
            #rr_problems = simple_concatenate(rr_problems, np.array(j))
            #rr_problems = simple_concatenate(rr_problems, np.array(j + 1))
            rr_problems = np.append(rr_problems, np.array([j]))
            rr_problems = np.append(rr_problems, np.array([j+1]))

    #set threshold for r-r interval variance
    #r peaks amplitude
    r_values = y_signal[rpeaks]
    r_values_average = np.average(abs(r_values))
    r_values_var = np.var(abs(r_values))
    r_values_problems = np.empty([0, 1])
    for j, r in enumerate(r_values):
        distance = abs(abs(r)-abs(r_values_average))
        if distance >= r_values_average/2:
            type = 'R value problem'
            #r_values_problems = simple_concatenate(r_values_problems, np.array(j))
            r_values_problems = np.append(r_values_problems, np.array([j]))
    #problems = rr_problems.append(r_values_problems)

    #problems = simple_concatenate(problems, rr_problems)
    #problems = simple_concatenate(problems, r_values_problems)
    problems = np.append(problems, rr_problems)
    problems = np.append(problems, r_values_problems)

    print(problems)
    return problems


def split_samples(file, path, fraction=0.15):
    train = []
    test = []
    type1 = []
    type2 = []
    paths, names, sampling_rates, labels = load_file(path, file)
    for i, l in enumerate(labels):
        if l == 'Normal':
            type1.append(i)
        else:
            type2.append(i)
    total = list(range(0, len(labels)))
    test_type1 = np.random.choice(type1, size=int(len(type1) * fraction), replace=False)
    test_type2 = np.random.choice(type2, size=int(len(type2) * fraction), replace=False)
    test = np.append(test_type1, list(set(test_type2) - set(test_type1)))
    train = list(set(total) - set(test))
    return train, test


def simple_concatenate(a,b):
    if a.shape[0] == 0:
        a = b
    else:
        a = np.append(a, [b], axis=0)
    return a
