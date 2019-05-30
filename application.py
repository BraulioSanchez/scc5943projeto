from GetData import GetData
from DataLoader import DataLoader
from PreProcessing import PreProcessing

import os
import json
import numpy as np
import pandas as pd

if __name__ == "__main__":
    configs = json.load(open('config.json', 'r'))

    GetData(configs['data']['ticker'],
                        configs['data']['start'],
                        configs['data']['end'],
                        configs).get_stock_data()
    amzn_dataloader = DataLoader(os.path.join(configs['data']['save_dir'], configs['data']['ticker'] + '.csv'),
                        configs['data']['train_test_split'],
                        configs['data']['column'])

    preprocessing = PreProcessing()
    preprocessing.denoise(amzn_dataloader.data, configs)

    all_data = {configs['data']['ticker']: preprocessing.denoised}

    for correlate in configs['data']['correlates_to']:
        GetData(correlate,
                configs['data']['start'],
                configs['data']['end'],
                configs).get_stock_data()
        dataloader = DataLoader(os.path.join(configs['data']['save_dir'], correlate + '.csv'),
                        configs['data']['train_test_split'],
                        configs['data']['column'])
        preprocessing.denoise(amzn_dataloader.data, configs)
        all_data.update({correlate: preprocessing.denoised})

    dataframe = pd.DataFrame(all_data)

    dataframe.to_csv(os.path.join(configs['preprocessing']['save_dir'], configs['preprocessing']['filename']),
                        index=False)

    '''close = data_amzn.stock_data.Close
    preprocess = PreProcessing()
    preprocess.denoise(close)
    preprocess.to_csv()

    import matplotlib.pyplot as plt
    plt.plot(preprocess.preprocessed.scaled, color='red')
    plt.plot(preprocess.preprocessed.denoised, color='blue')
    plt.legend(['Scaled prices', 'Denoised prices'])
    plt.grid(True)
    plt.show()'''