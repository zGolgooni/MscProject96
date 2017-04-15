__author__ = 'Zeynab'

from biosppy.signals import ecg
import numpy as np


def detect_r_points(signal, length, rate, algorithm='none'):
    length = signal.shape[0]
    ecg_out = ecg.ecg(signal=signal[:,0], sampling_rate=rate, show=False)
    filtered = ecg_out.__getitem__('filtered')
    temp_rpeaks = ecg_out.__getitem__('rpeaks')

    rpeaks = np.empty([temp_rpeaks.shape[0], 1], dtype=int)
    for i,r in enumerate(temp_rpeaks):
        #find maximum
        rpeaks[i] = r
        max_best_so_far = r
        end = False
        #while not end:
            #end = True
        for j in range(25):
            neighbor1 = max_best_so_far - j
            neighbor2 = max_best_so_far + j
            if abs(signal[neighbor1,0]) > abs(signal[max_best_so_far,0]):
                max_best_so_far = neighbor1
                    #end = False
            if abs(signal[neighbor2,0]) > abs(signal[max_best_so_far,0]):
                max_best_so_far = neighbor2
                    #end = False
        rpeaks[i] = max_best_so_far
    #print('finished finalizing r points! :)')

    return rpeaks
