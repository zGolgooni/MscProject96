__author__ = 'Zeynab'

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import LSTM
from keras.layers.core import Dense, Dropout, Activation

from Approach2.prepare_data import load_file, load_sample, split_samples

#list of files as csv
#split randomly to train and test file considering type of sample (output as list of indices)


main_path = '/Users/Zeynab/'
main_file = 'My data/In use/data1395.csv'
paths, names, sampling_rates, labels = load_file(main_path, main_file)

fraction = 0.2
train_samples, test_samples = split_samples(main_file, main_path, fraction)
#specify dimension of features considering approach of work

'''
Classify samples
sequence of raw time series
each entry refers to 5 peaks and related handy features of them, about 15 feature( base and each peak)
'''

total_rpeaks = 40
num_beats = 5
num_features = 1 + 5 + (2 * num_beats) - 1

train_x = np.empty((0, 1, num_features))
train_y = np.empty((0, 1))

for i in train_samples:
    sample_x, sample_y = load_sample(main_path+paths[i], names[i], labels[i], sampling_rates[i], num_beats, num_features, train=True)
    print('yep1 %d'%i)
    train_x = np.append(train_x, sample_x, axis=0)
    print('yep2 %s'%names[i])
    train_y = np.append(train_y, sample_y)
    print('yep3 %s'%labels[i])

#print('train samples is loaded:  x-> %d,%d,%d   y-> %d,%d,%d' % (train_x.shape[0],train_x.shape[1],train_x.shape[2],train_y.shape[0],train_y.shape[1],train_y.shape[2]))

hidden_nodes = 25
model = Sequential()
model.add(LSTM(hidden_nodes, input_dim=num_features, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss="binary_crossentropy", optimizer="rmsprop",metrics=['accuracy'])

model.fit(train_x, train_y, batch_size=100, nb_epoch=50, validation_split=0.15)
model.save_weights('new_calssification2_0.h5')

#test the train samples
print('**** Train samples: ****')

tp = 0
tn = 0
fp = 0
fn = 0
n = 0
for i in train_samples:
    print('i = %d' %i)
    sample_x, sample_y = load_sample(main_path+paths[i], names[i], labels[i], sampling_rates[i], num_beats, num_features, train=True)
    real_label = labels[i]
    num_arrhythmic = 0
    for index in range(sample_y.shape[0]):
        predicted = model.predict(np.array([sample_x[index]]))
        if predicted > 0.4:
            num_arrhythmic += 1
    if num_arrhythmic > 2:
        predicted_label = 'Arrhythmic'
    else:
        predicted_label = 'Normal'
    n += 1
    if real_label == 'Normal':
        if predicted_label == 'Normal':
            tn += 1
        else:
            fp += 1
    else:
        if predicted_label == 'Normal':
            fn += 1
        else:
            tp += 1
print('\n--->Result for data = train , samples (%d Arrhythmic, %d Normal)' % ( (fn+tp), (fp+tn)))
print('\t\ttp = %f, tn = %f, fp = %f, fn = %f, total-> %f, positive accuracy-> %f\n\n' % (tp, tn, fp, fn, ((tp+tn)/n),(tp/(tp+fn))))
print('------------------------------------')

#test the test samples
print('**** Test samples: ****')

tp = 0
tn = 0
fp = 0
fn = 0
n = 0
for i in test_samples:
    print('i = %d' %i)
    sample_x, sample_y = load_sample(main_path+paths[i], names[i], labels[i], sampling_rates[i], num_beats, num_features, train=True)
    real_label = labels[i]
    num_arrhythmic = 0
    for index in range(sample_y.shape[0]):
        predicted = model.predict(np.array([sample_x[index]]))
        if predicted > 0.4:
            num_arrhythmic += 1
    if num_arrhythmic > 2:
        predicted_label = 'Arrhythmic'
    else:
        predicted_label = 'Normal'
    n += 1
    if real_label == 'Normal':
        if predicted_label == 'Normal':
            tn += 1
        else:
            fp += 1
    else:
        if predicted_label == 'Normal':
            fn += 1
        else:
            tp += 1
print('\n--->Result for data = test , samples (%d Arrhythmic, %d Normal)' % ( (fn+tp), (fp+tn)))
print('\t\ttp = %f, tn = %f, fp = %f, fn = %f, total-> %f, positive accuracy-> %f\n\n' % (tp, tn, fp, fn, ((tp+tn)/n),(tp/(tp+fn))))

print('------------------------------------')

#test the test samples
print('**** Total samples: ****')

tp = 0
tn = 0
fp = 0
fn = 0
n = 0
for i in range(len(names)):
    print('i = %d' %i)
    sample_x, sample_y = load_sample(main_path+paths[i], names[i], labels[i], sampling_rates[i], num_beats, num_features, train=True)
    real_label = labels[i]
    num_arrhythmic = 0
    for index in range(sample_y.shape[0]):
        predicted = model.predict(np.array([sample_x[index]]))
        if predicted > 0.4:
            num_arrhythmic += 1
    if num_arrhythmic > 2:
        predicted_label = 'Arrhythmic'
    else:
        predicted_label = 'Normal'
    n += 1
    if real_label == 'Normal':
        if predicted_label == 'Normal':
            tn += 1
        else:
            fp += 1
    else:
        if predicted_label == 'Normal':
            fn += 1
        else:
            tp += 1
print('\n--->Result for data = total , samples (%d Arrhythmic, %d Normal)' % ( (fn+tp), (fp+tn)))
print('\t\ttp = %f, tn = %f, fp = %f, fn = %f, total-> %f, positive accuracy-> %f\n\n' % (tp, tn, fp, fn, ((tp+tn)/n),(tp/(tp+fn))))
print('------------------------------------')


'''
# temp test , train samples
tp = 0
tn = 0
fp = 0
fn = 0
n = 0
counter = 0
for i in range(train_x.shape[0]):
    sample_x = train_x[i]
    sample_y = train_y[i]

    #train_x = np.append(train_x, sample_x, axis=0)
    #train_y = np.append(train_y, sample_y)
    predicted = model.predict(np.array([sample_x]))

    if predicted > 0.4:
        predicted_label = 1
    else:
        predicted_label = 0
    n += 1

    if sample_y == 0:
        counter += 1
        if predicted_label == 0:
            tp += 1
        else:
            fn += 1
    else:
            if predicted_label == 1:
                tn += 1
            else:
                fp += 1
print('\n--->Result for data = train , samples (%d N, %d A)' % ( (fn+tp), (fp+tn)))
print('\t\ttp = %f, tn = %f, fp = %f, fn = %f, total-> %f\n\n' % (tp, tn, fp, fn, ((tp+tn)/n)))
'''
