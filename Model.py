from keras.models import Sequential

class Model():

    def __init__(self):
        self.model = Sequential()

    def build(self, configs):
        for layer in configs['model']['layers']:
            print(layer)


if __name__ == "__main__":
    import json
    configs = json.load(open('config.json', 'r'))

    model = Model()

    model.build(configs)