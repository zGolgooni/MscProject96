__author__ = 'Zeynab'

import numpy as np
import pandas
from sklearn.preprocessing import MinMaxScaler
from biosppy.signals.tools import smoother


min_range = -50
max_range = 50


def read_sample(path, name, preprocess=False):
    dataset = pandas.read_csv(path + name + '.txt', delimiter='\t', skiprows=4, skipfooter=1)
    x_signal = dataset.values[:, 0]
    y = dataset.values[:, 1]
    y_signal = normalize_data(pandas.DataFrame(y), max_range, min_range)
    if preprocess is True:
        smoothed_signal, params = smoother(y_signal[:,0])
        y_signal = np.reshape(smoothed_signal, [smoothed_signal.shape[0],1])
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
