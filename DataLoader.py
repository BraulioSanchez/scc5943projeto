import pandas as pd

class DataLoader():

    def __init__(self, filename, split, cols):
        dataframe = pd.read_csv(filename)
        i_split = int(len(dataframe) * split)
        self.data_train = dataframe.get(cols).values[:i_split]
        self.data_test = dataframe.get(cols).values[i_split:]
        self.len_train = len(self.data_train)
        self.len_train_windows = None

if __name__ == "__main__":
    filename = './NVDA.csv'
    data = DataLoader(filename, 0.85, ['Close'])
    print(data.data_train.shape, data.data_test.shape)
