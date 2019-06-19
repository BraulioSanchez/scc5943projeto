from get_data import GetData
from data_loader import DataLoader
from pre_processing import PreProcessing
from model import Model

import os
import json
import numpy as np
import pandas as pd

def main():
    configs = json.load(open('config.json', 'r'))

    # download and process all the datasets involved
    # includes AMZN
    GetData(configs['data']['symbol'],
                        configs['data']['start'],
                        configs['data']['end'],
                        configs).get_stock_data()
    amzn_dataloader = DataLoader(os.path.join(configs['data']['save_dir'], configs['data']['symbol'] + '.csv'),
                        configs['data']['columns'])

    preprocessing = PreProcessing()
    preprocessing.denoise(amzn_dataloader.data, configs)

    all_data = {configs['data']['symbol']: preprocessing.denoised}

    # and the correlated ones
    for correlate in configs['data']['correlates_to']:
        GetData(correlate,
                configs['data']['start'],
                configs['data']['end'],
                configs).get_stock_data()
        dataloader = DataLoader(os.path.join(configs['data']['save_dir'], correlate + '.csv'),
                        configs['data']['columns'])
        preprocessing = PreProcessing()
        preprocessing.denoise(dataloader.data, configs)
        all_data.update({correlate: preprocessing.denoised})

    # save all data preprocessed
    dataframe = pd.DataFrame(all_data)
    filename = os.path.join(configs['preprocessing']['save_dir'], configs['preprocessing']['filename'])
    dataframe.to_csv(filename, index=False)

    dataloader = DataLoader(filename,
                            configs['data']['correlates_to'])
    dataloader.train_test_split(configs['data']['days'], configs['data']['train_test_split'])

    model = Model()
    # build and train model
    model.build(configs, dataloader)

    from keras.utils import plot_model
    plot_model(model.model, show_shapes=True, to_file="autoencoder-lstm-multivariable-for-prediction.png")

    yhat = model.predict(dataloader.train, dataloader.test, configs['data']['inputs'])

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,8))
    plt.plot(dataloader.test[0,:,0].flatten(), label = 'Real')
    plt.plot(yhat.flatten(), label = 'Predicted')
    plt.legend()
    plt.show()

    yhat = model.predict(dataloader.train, dataloader.train, configs['data']['inputs'])
    plt.figure(figsize=(10,8))
    plt.plot(dataloader.train[0,:,0].flatten(), label = 'Real')
    plt.plot(yhat.flatten(), label = 'Predicted')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()