__author__ = 'Zeynab'

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, MaxPooling1D, Flatten


import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import LSTM
from keras.layers.core import Dense, Dropout, Activation

from Approach1.prepare_data import load_file, load_sample, split_samples
from prepare_data33 import load_file, load_sample, split_samples #, print_globals,reset_globals

#list of files as csv
#split randomly to train and test file considering type of sample (output as list of indices)
num_experiments = 5
train = 93
test = 30
train_tp = np.empty([num_experiments,1])
train_tn = np.empty([num_experiments,1])
train_fp = np.empty([num_experiments,1])
train_fn = np.empty([num_experiments,1])

test_tp = np.empty([num_experiments,1])
test_tn = np.empty([num_experiments,1])
test_fp = np.empty([num_experiments,1])
test_fn = np.empty([num_experiments,1])

train_fractions = np.empty([num_experiments,train])
test_fraction = np.empty([num_experiments,test])

main_path = '/Users/Zeynab/'
main_file = 'My data/In use/data1395.csv'
paths, names, sampling_rates, labels = load_file(main_path, main_file)
for run in range(0, num_experiments):
    fraction = 0.2
    train_samples, test_samples = split_samples(main_file, main_path, fraction)
    #specify dimension of features considering approach of work


    features_dim1 = 1
    features_dim2 = 2000

    train_x = np.empty((0, features_dim1, features_dim2))
    train_y = np.empty((0, 2))

    for i in train_samples:
        sample_x, sample_y = load_sample(main_path+paths[i], names[i], labels[i], sampling_rates[i], features_dim1, features_dim2, train=True)
        train_x = np.append(train_x, sample_x, axis=0)
        train_y = np.append(train_y, sample_y)
    train_x = np.reshape(train_x,[train_x.shape[0], features_dim2, features_dim1])


    nb_filter = 45
    filter_length = 20

    batch_size = 100
    epochs = 50

    print('Build model...')
    model = Sequential()
    model.add(Conv1D(nb_filter=nb_filter,filter_length=filter_length, activation='relu',input_shape=(features_dim2,features_dim1)))
    model.add(MaxPooling1D())
    model.add(Flatten())
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
    model.fit(train_x, train_y, batch_size=batch_size, nb_epoch=epochs, validation_split=0.15)

    print('**** Train samples: ****')
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    n = 0
    for i in train_samples:
        sample_x, sample_y = load_sample(main_path+paths[i], names[i], labels[i], sampling_rates[i], features_dim1, features_dim2,train=False)
        sample_x = np.reshape(sample_x,[sample_x.shape[0],features_dim2,features_dim1])
        real_label = labels[i]
        num_arrhythmic = 0
        for index in range(sample_y.shape[0]):
            predicted = model.predict(np.array([sample_x[index]]))
            if predicted >= 0.5:
                num_arrhythmic += 1
        if num_arrhythmic > 3:
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
    train_tp[run] = tp
    train_tn[run] = tn
    train_fp[run] = fp
    train_fn[run] = fn

    print('**** Test samples: ****')
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    n = 0
    for i in test_samples:
        sample_x, sample_y = load_sample(main_path+paths[i], names[i], labels[i], sampling_rates[i], features_dim1, features_dim2,train=False)
        sample_x = np.reshape(sample_x,[sample_x.shape[0],features_dim2,features_dim1])
        real_label = labels[i]
        num_arrhythmic = 0
        for index in range(sample_y.shape[0]):
            predicted = model.predict(np.array([sample_x[index]]))
            if predicted >= 0.5:
                num_arrhythmic += 1
        if num_arrhythmic > 3:
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
    test_tp[run] = tp
    test_tn[run] = tn
    test_fp[run] = fp
    test_fn[run] = fn


#test the train samples
tp = np.average(train_tp)
tn = np.average(train_tn)
fp = np.average(train_fp)
fn = np.average(train_fn)
print('**** Total : Train samples: ****')
print('\n--->Result for data = train , samples (%d Arrhythmic, %d Normal)' % ( (fn+tp), (fp+tn)))
print('\t\ttp = %f, tn = %f, fp = %f, fn = %f, total-> %f, positive accuracy-> %f\n\n' % (tp, tn, fp, fn, ((tp+tn)/(tp+fp+tn+fn)),(tp/(tp+fn))))


tp = np.average(test_tp)
tn = np.average(test_tn)
fp = np.average(test_fp)
fn = np.average(test_fn)
print('**** Total : Test samples: ****')
print('\n--->Result for data = test , samples (%d Arrhythmic, %d Normal)' % ( (fn+tp), (fp+tn)))
print('\t\ttp = %f, tn = %f, fp = %f, fn = %f, total-> %f, positive accuracy-> %f\n\n' % (tp, tn, fp, fn, ((tp+tn)/(tp+fp+tn+fn)),(tp/(tp+fn))))
