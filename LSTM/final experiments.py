__author__ = 'Zeynab'

import csv
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import LSTM
from keras.layers.core import Dense, Dropout, Activation

from prepare_data import load_file, load_sample, split_samples #, print_globals,reset_globals


def run_experiment(lstm_hidden_node,batch,epoch,classification_threshold,input_dimension,rpeaks,each_points):
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
        fraction = 0.25
        train_samples, test_samples = split_samples(main_file, main_path, fraction)
        #specify dimension of features considering approach of work
        train_fractions[run] = train_samples
        test_fraction[run] = test_samples
        features_dim1 = (rpeaks * each_points) /input_dimension
        features_dim2 = input_dimension

        train_x = np.empty((0, features_dim1, features_dim2))
        train_y = np.empty((0, 2))

        for i in train_samples:
            sample_x, sample_y = load_sample(main_path+paths[i], names[i], labels[i], sampling_rates[i], features_dim1, features_dim2, train=True)
            train_x = np.append(train_x, sample_x, axis=0)
            train_y = np.append(train_y, sample_y)

        test_x = np.empty((0, features_dim1, features_dim2))
        test_y = np.empty((0, 2))
        for i in test_samples:
            sample_x, sample_y = load_sample(main_path+paths[i], names[i], labels[i], sampling_rates[i], features_dim1, features_dim2, train=False)
            test_x = np.append(test_x, sample_x, axis=0)
            test_y = np.append(test_y, sample_y)

        hidden_nodes = lstm_hidden_node
        model = Sequential()
        model.add(LSTM(hidden_nodes, input_dim=features_dim2, return_sequences=False))
        model.add(Dropout(0.5))
        #model.add(Dense(10, activation='relu'))
        #model.add(Dropout(0.5))

        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        model.compile(loss="binary_crossentropy", optimizer="rmsprop",metrics=['accuracy'])

        model.fit(train_x, train_y, batch_size=batch, nb_epoch=epoch, validation_split=0.15)
        #model.save_weights('classification960120_1.h5')

        #test the train samples
        print('**** Train samples: ****')
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        n = 0
        for i in train_samples:
            #print('i = %d' %i)
            sample_x, sample_y = load_sample(main_path+paths[i], names[i], labels[i], sampling_rates[i], features_dim1, features_dim2, train=False)
            real_label = labels[i]
            num_arrhythmic = 0
            for index in range(sample_y.shape[0]):
                predicted = model.predict(np.array([sample_x[index]]))
                if predicted >= 0.5:
                    num_arrhythmic += 1
            if num_arrhythmic >= classification_threshold:
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

        train_tp[run] = tp
        train_tn[run] = tn
        train_fp[run] = fp
        train_fn[run] = fn

        #test the test samples
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        n = 0
        for i in test_samples:
            #print('i = %d' %i)
            sample_x, sample_y = load_sample(main_path+paths[i], names[i], labels[i], sampling_rates[i], features_dim1, features_dim2,train=False)
            real_label = labels[i]
            num_arrhythmic = 0
            for index in range(sample_y.shape[0]):
                predicted = model.predict(np.array([sample_x[index]]))
                if predicted >= 0.5:
                    num_arrhythmic += 1
            if num_arrhythmic >= classification_threshold:
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
        test_tp[run] = tp
        test_tn[run] = tn
        test_fp[run] = fp
        test_fn[run] = fn

    #test the test samples
    test_tp = np.average(test_tp)
    test_tn = np.average(test_tn)
    test_fp = np.average(test_fp)
    test_fn = np.average(test_fn)
    test_total_acc = np.average((test_tp + test_tn)/test)
    test_positive_acc = np.average(test_tp/(test_tp + test_fn))


    #test the train samples
    train_tp = np.average(train_tp)
    train_tn = np.average(train_tn)
    train_fp = np.average(train_fp)
    train_fn = np.average(train_fn)
    train_total_acc = np.average((train_tp + train_tn)/train)
    train_positive_acc = np.average(train_tp/(train_tp + train_fn))

    train_test_data = [train, test]
    return test_total_acc,train_total_acc,test_positive_acc,train_positive_acc,train_test_data,test_fp,test_fn,test_tp,test_tn,train_fp,train_fn,train_tp,train_tn


lstm_hidden_node = [10,15,20,25,30]
batch = [25,50,100,150,200]
epoch = [50,75,100]
classification_threshold = [1,2,3]

#rpeaks =[1,2,3,4]
#each_points = [400,500,600,800,1000]
rpeaks = 4
each_points = 500
input_dimension = [50,80,100,250,500]


with open('Results_960126.csv', 'w') as csvfile:
    fieldnames = ['model #LSTM hidden node','model #batch','model #epoch', 'classification threshold','data #rpeaks #each points','data input dimension','test total_acc','train total_acc','test positive_acc','train positive_acc','train/test data','test-fp', 'test-fn','test-tp', 'test-tn', 'train-fp','train-fn','train-tp', 'train-tn']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    counter = 0
    for ct in classification_threshold:
        print('Classification threshold = %d' %ct)
        for id in input_dimension:
            print('input dimension = %d' %id)
            for b in batch:
                print('batch = %d' %b)
                for e in epoch:
                    print('Epoch = %d' %e)
                    for lhn in lstm_hidden_node:
                        print('lstm hidden node = %d' %lhn)
                        test_total_acc,train_total_acc,test_positive_acc,train_positive_acc,train_test_data,test_fp,test_fn,test_tp,test_tn,train_fp,train_fn,train_tp,train_tn= run_experiment(lhn,b,e,ct,id,rpeaks,each_points)
                        writer.writerow({'model #LSTM hidden node':lhn,'model #batch':b,'model #epoch':e, 'classification threshold':ct,'data #rpeaks #each points':[rpeaks, each_points],'data input dimension':[id],'test total_acc':test_total_acc,'train total_acc': train_total_acc,'test positive_acc': test_positive_acc,'train positive_acc':train_positive_acc,'train/test data':train_test_data,'test-fp':test_fp, 'test-fn':test_fn,'test-tp':test_tp, 'test-tn':test_tn, 'train-fp':train_fp,'train-fn':train_fn,'train-tp':train_tp, 'train-tn':train_tn})
                        counter += 1
                        print('oooooooooooh   %d ohhhhhhhhhhhh' %counter)
