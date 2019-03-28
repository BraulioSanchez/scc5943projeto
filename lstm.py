import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import argparse

parser = argparse.ArgumentParser(description='Long-Short Term Memory')
parser.add_argument('--data', help='Dataset with Prices Stock', dest='DATA')
parser.add_argument('--epochs', help='Epochs for training', dest='EPOCHS', default=5)
parser.add_argument('--model', help='Model for Prediction', dest='MODEL', default='LSTM')
args = parser.parse_args()

import keras
from keras.layers import Dense, Activation, LSTM, GRU
import tensorflow as tf
from utils import to_1dimension, create_model, plot_series_prediction

#ocultar alertas do módulo keras
tf.logging.set_verbosity(tf.logging.ERROR)

#constantes
TEST_SIZE = .3
TIME_AHEAD = 1 #dias a prever
BATCH_SIZE = 1
UNITS = 25

df = pd.read_csv(args.DATA)
df = df.drop(['Name', 'volume'], axis=1)
#print(df.shape)
#print(df.head())

'''df.plot(figsize=(20,10), linewidth=3, fontsize=20)
plt.xlabel('Days')
plt.show()'''
prices = df.open
scaler = MinMaxScaler(feature_range=(0,1))
prices = scaler.fit_transform(np.reshape(prices.values, (len(prices), 1)))
#print(prices)

train, test = train_test_split(prices, test_size=TEST_SIZE, shuffle=False)
print('dimensions of train:', train.shape)
print('dimensions of test:', test.shape)

X_train, y_train = to_1dimension(train, TIME_AHEAD)
X_test, y_test = to_1dimension(test, TIME_AHEAD)

#LSTM
model = create_model(model_name=args.MODEL, units=UNITS, time_ahead=TIME_AHEAD)
'''Para otimização, usamos o algoritmo ADAM.
    Em séries temporais, os métodos de otimização adaptativa
    tendem a obter melhores resultados do que os métodos
    tradicionais de descida de gradiente estocástica.'''
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=int(args.EPOCHS), batch_size=BATCH_SIZE, verbose=2)

#recuperamos os valores originais em y_test e prices
y_test_origin = scaler.inverse_transform([y_test])
prices_origin = scaler.inverse_transform(prices)

pred_test = model.predict(X_test)
pred_test = scaler.inverse_transform(pred_test)
error = mean_squared_error(y_test_origin[0], pred_test[:,0])
print('MSE: %.2f' % error)

pred_train = model.predict(X_train)
pred_train = scaler.inverse_transform(pred_train)
plot_series_prediction(prices_origin, pred_train, pred_test, time_ahead=TIME_AHEAD,
                        title='Predictions', xlabel='Days', ylabel='Prices of AMZN Stock',
                        legend=['Opening prices', 'Training set', 'Test prediction'])