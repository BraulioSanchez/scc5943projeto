import pandas as pd
import numpy as np

class DataLoader():

    def __init__(self, filename, split, cols):
        dataframe = pd.read_csv(filename)
        i_split = int(len(dataframe) * split)
        self.data = dataframe.get(cols).values
        self.data_train = dataframe.get(cols).values[:i_split]
        self.data_test = dataframe.get(cols).values[i_split:]
        self.len_train = len(self.data_train)
        self.len_test = len(self.data_test)
        self.len_train_windows = None

    def get_train_data(self, seq_len):
        data_x = []
        data_y = []
        for i in range(self.len_train - seq_len):
            x, y = self.__get_next_window(i, seq_len)
            data_x.append(x)
            data_y.append(y)

        return np.array(data_x), np.array(data_y)

    def get_test_data(self, seq_len):
        data_windows = []
        for i in range(self.len_test - seq_len):
            data_windows.append(self.data_test[i:i+seq_len])

        data_windows = np.array(data_windows).astype(float)
        
        X = data_windows[:,:-1]
        y = data_windows[:,-1,[0]]
        return X, y

    def __get_next_window(self, i, seq_len):
        window = self.data_train[i:i+seq_len]
        
        x = window[:-1]
        y = window[-1, [0]]
        return x, y

if __name__ == "__main__":
    import json
    import os
    configs = json.load(open('config.json', 'r'))

    dataloader = DataLoader(os.path.join(configs['data']['save_dir'], configs['data']['symbol'] + '.csv'),
                        configs['data']['train_test_split'],
                        configs['data']['columns'])
    print(dataloader.data.shape, dataloader.data_train.shape, dataloader.data_test.shape)
    
    X_train, y_train = dataloader.get_train_data(configs['data']['sequence_length'])
    print('train:')
    print(X_train.shape, y_train.shape)
    print(X_train[:2])
    print(y_train[:2])

    X_test, y_test = dataloader.get_test_data(configs['data']['sequence_length'])
    print('test:')
    print(X_test.shape, y_test.shape)
    print(X_test[:2])
    print(y_test[:2])
