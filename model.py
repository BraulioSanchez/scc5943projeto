from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM, RepeatVector, TimeDistributed
import tensorflow as tf

import os
import numpy as np
import datetime as dt

tf.logging.set_verbosity(tf.logging.ERROR)

class Model():

    def __init__(self):
        self.model = Sequential()

    def build(self, configs, dataloader):
        train_x, train_y = dataloader.to_supervised(configs['data']['inputs'], configs['data']['days'])
        
        n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
        train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
        
        for layer in configs['model']['layers']:
            neurons = layer['neurons'] if 'neurons' in layer else None
            activation = layer['activation'] if 'activation' in layer else None
            rate = layer['rate'] if 'rate' in layer else None
            return_sequences = layer['return_sequences'] if 'return_sequences' in layer else None

            if layer['type'] == 'lstm':
                self.model.add(LSTM(units=neurons,
                                    activation=activation,
                                    input_shape=(n_timesteps, n_features),
                                    return_sequences=return_sequences))
            if layer['type'] == 'repeat_vector':
                self.model.add(RepeatVector(n_outputs))
            if layer['type'] == 'dropout':
                self.model.add(Dropout(rate))
            if layer['type'] == 'time_distributed':
                if layer['layer']['type'] == 'dense':
                    neurons = layer['layer']['neurons'] if 'neurons' in layer['layer'] else None
                    activation = layer['layer']['activation'] if 'activation' in layer['layer'] else None
                    self.model.add(TimeDistributed(Dense(units=neurons, activation=activation)))

        self.model.compile(loss=configs['model']['loss'], optimizer=configs['model']['optimizer'])
        self.model.summary()

        self.model.fit(train_x, train_y, epochs=configs['training']['epochs'], batch_size=configs['training']['batch_size'],
                       verbose=configs['training']['verbose'])

        save_model = os.path.join(configs['model']['save_dir'], dt.datetime.now().strftime('%d%m%Y-%H%M%S') + '.h5')
        if not os.path.exists(configs['model']['save_dir']):
            os.makedirs(configs['model']['save_dir'])

        self.model.save(save_model)

    def __forecast(self, history, inputs):
        data = np.array(history)
        data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))

        input_x = data[-inputs:, :] #latest info for the input
        input_x = input_x.reshape((1, input_x.shape[0], input_x.shape[1]))
        yhat = self.model.predict(input_x, verbose=0)
        yhat = yhat[0]
        return yhat

    def predict(self, train, test, inputs):
        history = [x for x in train]
        predictions = list()
        for i in range(len(test)):
            yhat_sequence = self.__forecast(history, inputs)
            predictions.append(yhat_sequence)
            history.append(test[i, :])
        return np.array(predictions)

if __name__ == "__main__":
    import json
    configs = json.load(open('config.json', 'r'))

    from data_loader import DataLoader
    dataloader = DataLoader(os.path.join(configs['data']['save_dir'], configs['data']['symbol'] + '.csv'),
                        configs['data']['columns'])
    dataloader.train_test_split(configs['data']['days'], configs['data']['train_test_split'])
    
    model = Model()
    model.build(configs, dataloader)
    yhat = model.predict(dataloader.train, dataloader.test, configs['data']['inputs'])

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,8))
    plt.plot(dataloader.test[0,:,0].flatten(), label = 'Real without normalization')
    plt.plot(yhat.flatten(), label = 'Predicted')
    plt.legend()
    plt.show()

    from keras.utils import plot_model
    plot_model(model.model, show_shapes=True, to_file="lstm-model-1.png")