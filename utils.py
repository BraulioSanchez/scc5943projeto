import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Bidirectional, LSTM, GRU
from bokeh.plotting import figure
from bokeh.io import show, export_svgs
from IPython.display import SVG, display

def to_1dimension(df, step_size):
    X, y = [], []
    for i in range(len(df)-step_size-1):
        data = df[i:(i+step_size), 0]
        X.append(data)
        y.append(df[i + step_size, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
    return X, y

def create_model(model_name='LSTM', units=0, activation='linear', time_ahead=1):
    model = Sequential()
    if model_name == 'LSTM':
        model.add(LSTM(units, input_shape=(1, time_ahead)))
    elif model_name == 'GRU':
        model.add(GRU(units, input_shape=(1, time_ahead)))
    else:
        raise ValueError('Nome de Modelo Incorreto')
    model.add(Dense(1))
    model.add(Activation('linear'))
    return model

def plot_series_prediction(true_values, train_predict, test_predict, time_ahead=1, title=None,
                           xlabel=None, ylabel=None, color=['green','red','blue'], legend=[None,None,None]):    
    TOOLS = 'pan,wheel_zoom,box_zoom,reset,save,box_select'
    #x axis
    xx = np.array(range(true_values.shape[0]))
    xx1 = np.array(range(time_ahead,len(train_predict)+time_ahead))
    xx2 = np.array(range(len(train_predict)+(time_ahead*2)+1,len(true_values)-1))
    
    #figure
    p = figure(title=title, tools=TOOLS)
    p.line(xx, true_values.squeeze(), legend=legend[0], line_color=color[0], line_width=2)
    p.line(xx1, train_predict.squeeze(), legend=legend[1], line_color=color[1], line_width=1)    
    p.line(xx2, test_predict.squeeze(), legend=legend[2], line_color=color[2], line_width=1)
    p.axis[0].axis_label = xlabel
    p.axis[1].axis_label = ylabel
    p.legend.location = "top_left"
    show(p)