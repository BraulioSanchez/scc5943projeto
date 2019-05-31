from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM, RepeatVector, TimeDistributed
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)

class Model():

    def __init__(self):
        self.model = Sequential()

    def build(self, configs):
        for layer in configs['model']['layers']:
            neurons = layer['neurons'] if 'neurons' in layer else None
            activation = layer['activation'] if 'activation' in layer else None
            input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
            input_dim = layer['input_dim'] if 'input_dim' in layer else None
            return_sequences = layer['return_sequences'] if 'return_sequences' in layer else None
            output_shape = layer['output_shape'] if 'output_shape' in layer else None

            if layer['type'] == 'lstm':
                self.model.add(LSTM(units=neurons,
                                    activation=activation,
                                    input_shape=(input_timesteps, input_dim),
                                    return_sequences=return_sequences))
            if layer['type'] == 'repeat_vector':
                self.model.add(RepeatVector(output_shape))
            if layer['type'] == 'time_distributed':
                if layer['layer']['type'] == 'dense':
                    neurons = layer['layer']['neurons'] if 'neurons' in layer['layer'] else None
                    activation = layer['layer']['activation'] if 'activation' in layer['layer'] else None
                    self.model.add(TimeDistributed(Dense(units=neurons, activation=activation)))


if __name__ == "__main__":
    import json
    configs = json.load(open('config.json', 'r'))
    
    model = Model()
    model.build(configs)

    from keras.utils import plot_model
    plot_model(model.model, show_shapes=True, to_file="lstm-model-1.png")