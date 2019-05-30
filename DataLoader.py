import pandas as pd

class DataLoader():

    def __init__(self, filename, split, cols):
        dataframe = pd.read_csv(filename)
        i_split = int(len(dataframe) * split)
        self.data = dataframe.get(cols).values
        self.data_train = dataframe.get(cols).values[:i_split]
        self.data_test = dataframe.get(cols).values[i_split:]
        self.len_train = len(self.data_train)
        self.len_train_windows = None

if __name__ == "__main__":
    import json
    import os
    configs = json.load(open('config.json', 'r'))

    data = DataLoader(os.path.join(configs['data']['save_dir'], configs['data']['ticker'] + '.csv'),
                        configs['data']['train_test_split'],
                        configs['data']['column'])
    print(data.data.shape, data.data_train.shape, data.data_test.shape)
