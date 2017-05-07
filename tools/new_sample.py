import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
from Handy.simple_classification import classify_by_rpoints
from tools.plot import plot_sample
from tools.load_data import read_sample
from tools.r_detection import detect_r_points
from conv1d.prepare_data import load_file


def new_sample(path, name, sampling_rate,i):
    x_signal, y_signal =read_sample(path, name,sampling_rate, preprocess=False)

    rpeaks = detect_r_points(y_signal,y_signal.shape[0], sampling_rate)
    print('y_shape %d'%y_signal.shape[0])
    print('rpeaks_shape %d'%rpeaks.shape[0])

    label, type = classify_by_rpoints(y_signal, rpeaks, sampling_rate)
    '''
    plot_length = 40000
    trace1 = go.Scatter(y=y_signal[:plot_length], x=x_signal[:plot_length], name='Signal')
    trace2 = go.Scatter(y=y_signal[rpeaks], x=x_signal[rpeaks],mode='markers', name='rpeaks')
    figure = go.Figure(data=[trace1, trace2])
    py.plot(figure, filename=name)
    print('Plotting is done! :)')
    '''
    print('% s Suggested label by simple handy classification = %s' %(name,label))
    if labels[i] != label:
        trace1 = go.Scatter(y=y_signal[:], x=x_signal[:], name='Signal')
        trace2 = go.Scatter(y=y_signal[rpeaks[:,0]], x=x_signal[rpeaks[:,0]],mode='markers', name='rpeaks')
        layout = go.Layout(title=names[i])
        figure = go.Figure(data=[trace1, trace2], layout=layout)
        #py.plot(figure, filename=names[i])

        #print('Plotting is done! :)  sampling rate = %s' %sampling_rates[i])
    return label

main_path = '/Users/Zeynab'
main_file = '/Desktop/Data_960211.csv'
paths, names, sampling_rates, labels = load_file(main_path, main_file)
fp = 0
fn = 0
for i in range(0,len(names)):
    print('------- %d :     %s---------'%(i, labels[i]))
    label = new_sample(main_path+paths[i], names[i],sampling_rates[i],i)

    #if (labels[i] == 'Normal') & (label == 'Arrhythmic'):
    if labels[i] != label:
        print('///////oh oh %d/////'%i)
        if label =='Normal':
            fn += 1
        else:
            fp += 1