import numpy as np
import pandas as pd
import pywt
from sklearn.preprocessing import MinMaxScaler

class PreProcessing:
    def denoise(self, series, configs):
        #performs scale
        scaler = MinMaxScaler(feature_range=(-1,1))
        scaled = scaler.fit_transform(np.array(series).reshape(-1,1)).reshape(series.shape)

        #WaveShrink
        cA, cD = pywt.dwt(scaled, configs['preprocessing']['denoise']['wavelet'])
        #threshold selection
        thr = np.std(scaled)/23
        cA_shrinked = pywt.threshold(cA, thr, mode=configs['preprocessing']['denoise']['thr_mode'])
        cD_shrinked = pywt.threshold(cD, thr, mode=configs['preprocessing']['denoise']['thr_mode'])
        #reconstructs data from the given shrinked coefficients
        denoised = pywt.idwt(cA_shrinked, cD_shrinked, configs['preprocessing']['denoise']['wavelet'])

        if len(denoised) > len(scaled):
            denoised = denoised[:-1]

        self.scaled = scaled.flatten()
        self.denoised = denoised.flatten()

if __name__ == "__main__":
    stock_data = pd.read_csv('./data/MSI.csv')
    close = stock_data.Close

    import json
    configs = json.load(open('config.json', 'r'))

    preprocess = PreProcessing()
    preprocess.denoise(close,configs)

    import matplotlib.pyplot as plt
    plt.plot(preprocess.denoised, color='blue')
    plt.plot(preprocess.scaled, color='red')
    plt.grid(True)
    plt.show()