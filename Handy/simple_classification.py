__author__ = 'Zeynab'
import numpy as np

#define standards
bpm_min = 40
bpm_max = 100


def classify_by_rpoints(signal, rpeaks, sampling_rate):
    print('in classify')
    length = signal.shape[0]
    label = 'Normal'
    problems = 0
    type = 'Normal'
    #number of beats and its rate
    total_seconds = length/sampling_rate
    total_beats = rpeaks.shape[0]
    bps_total = total_beats/total_seconds
    bpm_total = bps_total * 60
    print('Beats/Min = %f   (in %f sec)' % (bpm_total, total_seconds))

    if bpm_total <= bpm_min:
        label = 'Arrhythmic'
        type = 'Bradicardia'
        print('Bradicardia (Beats/Min = %d, less than %d' % (bpm_total, bpm_min))
        problems += 1
    elif bpm_total >= bpm_max:
        label = 'Arrhythmic'
        type = 'Tachicardia'
        print('Tachicardia (Beats/Min = %d, more than %d' % (bpm_total, bpm_max))
        problems += 1

    problems = 0
    #r-r interval
    rr_interval = []
    for j in range(0, rpeaks.shape[0]-1):
        rr_interval.append((rpeaks[j + 1] - rpeaks[j])/sampling_rate)
    rr_average = np.average(rr_interval)
    rr_variance = np.var(rr_interval)
    print('Avg of R-R = %f', rr_average)
    print('Var of R-R = %f', rr_variance)

#normal_mu, normal_std = norm.fit(normal_rmse)
    #rr_difference = rr_interval - rr_variance -> to calculate r-r avg from a clean and standard part not all of signal
    rr_problems = []
    for j, r in enumerate(rr_interval):
        distance = abs(r - rr_average)
        if distance >= rr_average/2:
            print('Problem at r in point (index rr_interval = %d): r-r = %f, avg = %f' % (j, r, rr_average))
            type = 'R-R problem'
            rr_problems.append(j)
            problems += 1

    #set threshold for r-r interval variance

    #r peaks amplitude
    r_values = signal[rpeaks]
    r_values_average = np.average(abs(r_values))
    r_values_var = np.var(abs(r_values))
    r_values_problems = []
    for j, r in enumerate(r_values):
        distance = abs(abs(r)-abs(r_values_average))
        if distance >= r_values_average/2:
            print('problem at r in point (index r_value = %d), r-value = %f, avg = %f' % (j, r, r_values_average))
            type = 'R value problem'
            r_values_problems.append(j)
            problems += 1

    #Assign final label
    if type == 'R-R problem':
        label = 'Arrhythmic'
    elif (type == 'R value problem') & (problems > 3):
        print('oh oh %d'%problems)
        label = 'Arrhythmic'

    return label, type
