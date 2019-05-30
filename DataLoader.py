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
        self.len_train_windows = None

    def get_train_data(self, seq_len):
        data_x = []
        data_y = []
        for i in range(self.len_train - seq_len):
            x, y = self.get_next_window(i, seq_len)
            data_x.append(x)
            data_y.append(y)

        return np.array(data_x), np.array(data_y)

    def get_next_window(self, i, seq_len):
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
                        configs['data']['column'])
    print(dataloader.data.shape, dataloader.data_train.shape, dataloader.data_test.shape)
    
    X_train, y_train = dataloader.get_train_data(configs['data']['sequence_length'])
    print(X_train.shape, y_train.shape)
