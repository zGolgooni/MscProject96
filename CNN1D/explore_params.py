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
import csv
from conv1d.prepare_data import load_file, load_sample, split_samples #, print_globals,reset_globals


def run_experiment(data_dimension1=4,data_dimension2=500,features_dim1=1,features_dim2=2000,nb_filter1 = 32,filter_length1 = 20,nb_filter2 = 5,filter_length2 = 10,batch_size = 20,epochs = 70):
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

    main_path = '/home/mll/Golgooni/'
    main_file = 'My data/In use/data1395.csv'
    paths, names, sampling_rates, labels = load_file(main_path, main_file)
    for run in range(0, num_experiments):
        fraction = 0.2
        train_samples, test_samples = split_samples(main_file, main_path, fraction)
        #specify dimension of features considering approach of work


        train_x = np.empty((0, features_dim1, features_dim2))
        train_y = np.empty((0, 2))

        for i in train_samples:
            sample_x, sample_y = load_sample(main_path+paths[i], names[i], labels[i], sampling_rates[i], features_dim1, features_dim2, train=True, dimension1= data_dimension1, dimension2=data_dimension2)
            train_x = np.append(train_x, sample_x, axis=0)
            train_y = np.append(train_y, sample_y)
        train_x = np.reshape(train_x,[train_x.shape[0], features_dim2, features_dim1])

        print('Build model...')
        model = Sequential()
        model.add(Conv1D(nb_filter=nb_filter1,filter_length=filter_length1, activation='relu',input_shape=(features_dim2,features_dim1)))
        model.add(MaxPooling1D())
        model.add(Conv1D(nb_filter=nb_filter2,filter_length=filter_length2, activation='relu'))
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


data_dimension1 = 4
data_dimension2 = 500

features_dim1 = 1
features_dim2 = data_dimension2 * data_dimension1

nb_filter1 = [16,32,64]
filter_length1 = [15,25,50,100,200]

nb_filter2 = [16,8]
filter_length2 = [20,10,5]

batch_size = 20
epochs = 50

classification_threshold = 3
#classification_threshold = [1,2,3]


with open('Results_960128_1.csv', 'w') as csvfile:
    fieldnames = ['nb_filter1','filter_length1','nb_filter2','filter_length2','model #batch','model #epoch', 'classification threshold','data #rpeaks #each points','data input dimension','test total_acc','train total_acc','test positive_acc','train positive_acc','train/test data','test-fp', 'test-fn','test-tp', 'test-tn', 'train-fp','train-fn','train-tp', 'train-tn']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    counter = 0

    for nf1 in nb_filter1:
        for fl1 in filter_length1:
            for nf2 in nb_filter2:
                for fl2 in filter_length2:
                    test_total_acc,train_total_acc,test_positive_acc,train_positive_acc,train_test_data,test_fp,test_fn,test_tp,test_tn,train_fp,train_fn,train_tp,train_tn= run_experiment(data_dimension1=data_dimension1,data_dimension2=data_dimension2,features_dim1=features_dim1,features_dim2=features_dim2,nb_filter1 = nf1,filter_length1 = fl1,nb_filter2 = nf2,filter_length2 = fl2,batch_size = batch_size,epochs = epochs)
                    writer.writerow({'nb_filter1':nf1,'filter_length1':fl1,'nb_filter2':nf2,'filter_length2':fl2,'model #batch':batch_size,'model #epoch':epochs, 'classification threshold':classification_threshold,'data #rpeaks #each points':[data_dimension1, data_dimension2],'data input dimension':[features_dim1, features_dim2],'test total_acc':test_total_acc,'train total_acc': train_total_acc,'test positive_acc': test_positive_acc,'train positive_acc':train_positive_acc,'train/test data':train_test_data,'test-fp':test_fp, 'test-fn':test_fn,'test-tp':test_tp, 'test-tn':test_tn, 'train-fp':train_fp,'train-fn':train_fn,'train-tp':train_tp, 'train-tn':train_tn})
                    counter += 1
                    print('oooooooooooh   %d ohhhhhhhhhhhh' %counter)