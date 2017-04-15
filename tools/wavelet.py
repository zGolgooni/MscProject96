__author__ = 'Zeynab'
from Approach3.prepare_data import read_sample, load_file
from tools.r_detection import detect_r_points
import pywt
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools


main_path = '/Users/Zeynab/'
main_file = 'My data/In use/data1395.csv'
paths, names, sampling_rates, labels = load_file(main_path, main_file)

i = 5
path = main_path + paths[i]
name = names[i]
x_signal, y_signal = read_sample(path, name, preprocess=True)

cA, cD = pywt.dwt(y_signal[:60000,0],'db1')
print(cA.shape[0])
print(cD.shape[0])
rpeaks = detect_r_points(y_signal, 60000, sampling_rates[i])

a = pywt.idwt(cA[:100], None, 'db1', 'smooth')

trace1 = go.Scatter(y=y_signal[:20000], x=x_signal[:20000], name='Signal')
trace2 = go.Scatter(y=a[:20000], x=x_signal[:20000],mode='markers', name='inverse dwt')
figure = go.Figure(data=[trace1, trace2])
#plt.plot(x_signal[:20000], y_signal[:20000], 'go', x_signal[rpeaks], y_signal[rpeaks], 'b-')
py.plot(figure, filename='idwt')
print('Plotting is done! :)')
