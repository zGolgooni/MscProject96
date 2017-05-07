__author__ = 'Zeynab'
import csv, pandas
import numpy as np
from itertools import izip


path = '/Users/Zeynab/My data/95.10.15/code 1/'
file_name = 'baseline.txt'

dataset = pandas.read_csv(path + file_name, delimiter='\t', skiprows=4)
x_signal = dataset.values[:, 0]
y = dataset.values[:, 1]

np.savetxt('test.txt', (x_signal, y), delimiter='          ',fmt='%1.3f')

pandas.DataFrame()

with open('test2.txt', 'wb') as f:
    writer = csv.writer(f)
    for i in range(10):
        writer.writerows(str(x_signal[i]) +'\t'+str(y[i]))

f = open('test2.txt', 'w')
for i in range(10):
        f.write(str(x_signal[i]) +'\t'+str(y[i]))
np.savetxt('test2.txt', zip(x_signal, y), fmt='%.3f')