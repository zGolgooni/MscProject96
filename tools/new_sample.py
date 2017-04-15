__author__ = 'Zeynab'

import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
from Handy.simple_classification import classify_by_rpoints
from tools.plot import plot_sample
from tools.load_data import read_sample
from tools.r_detection import detect_r_points


def new_sample(path, name):
    x_signal, y_signal =read_sample(path, name, preprocess=False)
    if x_signal[2]-x_signal[1] == 0.001:
            sampling_rate = 1000
    else:
            sampling_rate = 2000
    rpeaks = detect_r_points(y_signal,y_signal.shape[0], sampling_rate)
    label, type = classify_by_rpoints(y_signal, rpeaks, sampling_rate)
    plot_length = 40000
    trace1 = go.Scatter(y=y_signal[:plot_length], x=x_signal[:plot_length], name='Signal')
    trace2 = go.Scatter(y=y_signal[rpeaks], x=x_signal[rpeaks],mode='markers', name='rpeaks')
    figure = go.Figure(data=[trace1, trace2])
    py.plot(figure, filename=name)
    print('Suggested label by simple handy classification = %s' %label)
    print('Plotting is done! :)')
