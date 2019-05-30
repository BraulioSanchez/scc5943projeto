from GetData import GetData
from PreProcessing import PreProcessing

import json

if __name__ == "__main__":
    configs = json.load(open('config.json', 'r'))

    data_amzn = GetData(configs['data']['ticker'],
                        configs['data']['start'],
                        configs['data']['end'],
                        configs)
    data_amzn.get_stock_data()

    close = data_amzn.stock_data.Close
    preprocess = PreProcessing()
    preprocess.denoise(close)
    preprocess.to_csv()

    import matplotlib.pyplot as plt
    plt.plot(preprocess.preprocessed.scaled, color='red')
    plt.plot(preprocess.preprocessed.denoised, color='blue')
    plt.legend(['Scaled prices', 'Denoised prices'])
    plt.grid(True)
    plt.show()