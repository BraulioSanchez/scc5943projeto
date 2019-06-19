import pandas as pd
import numpy as np

class DataLoader():

    def __init__(self, filename, cols):
        dataframe = pd.read_csv(filename)
        data = dataframe.get(cols).values
        # for testing purpose
        tricky = 2800
        self.data = data[:tricky]

    def train_test_split(self, days, train_test_split):
        train, test = self.data[:train_test_split], self.data[train_test_split:]
        self.train = np.array(np.split(train, len(train)/days))
        self.test = np.array(np.split(test, len(test)/days))

    def to_supervised(self, inputs, days):
        data = self.train.reshape((self.train.shape[0]*self.train.shape[1], self.train.shape[2]))
        X, y = [], []
        in_start = 0
        for _ in range(len(data)):
            in_end = in_start + inputs
            out_end = in_end + days
            if out_end < len(data):
                X.append(data[in_start:in_end, :])
                y.append(data[in_end:out_end, 0])
            in_start += 1
        return np.array(X), np.array(y)

if __name__ == "__main__":
    import json
    import os
    configs = json.load(open('config.json', 'r'))

    dataloader = DataLoader(os.path.join(configs['data']['save_dir'], configs['data']['symbol'] + '.csv'),
                        configs['data']['columns'])
    dataloader.train_test_split(configs['data']['days'], configs['data']['train_test_split'])
    print(dataloader.data.shape, dataloader.train.shape, dataloader.test.shape)

    
    X_train, y_train = dataloader.to_supervised(configs['data']['inputs'], configs['data']['days'])
    print(X_train.shape, y_train.shape)
    print(X_train[:2])
    print(y_train[:2])