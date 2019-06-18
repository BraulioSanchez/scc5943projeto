import numpy as np
import pandas as pd
import pywt
from sklearn.preprocessing import MinMaxScaler

class PreProcessing:
    def denoise(self, series, configs):
        #performs scale
        #scaler = MinMaxScaler(feature_range=(-1,1))
        #scaled = scaler.fit_transform(np.array(series).reshape(-1,1)).reshape(series.shape)
        normalized = (series - series.mean()) / (series.max() - series.min())

        #WaveShrink
        cA, cD = pywt.dwt(normalized, configs['preprocessing']['denoise']['wavelet'])
        #threshold selection
        thr = np.std(normalized)/23
        cA_shrinked = pywt.threshold(cA, thr, mode=configs['preprocessing']['denoise']['thr_mode'])
        cD_shrinked = pywt.threshold(cD, thr, mode=configs['preprocessing']['denoise']['thr_mode'])
        #reconstructs data from the given shrinked coefficients
        denoised = pywt.idwt(cA_shrinked, cD_shrinked, configs['preprocessing']['denoise']['wavelet'])

        if len(denoised) > len(normalized):
            denoised = denoised[:-1]

        # what???!!!
        if len(denoised.shape) > 1:
            denoised = [x[0] for x in denoised]

        self.normalized = normalized
        self.denoised = denoised

if __name__ == "__main__":
    stock_data = pd.read_csv('./data/V.csv')
    close = stock_data.Close

    import json
    configs = json.load(open('config.json', 'r'))

    preprocess = PreProcessing()
    preprocess.denoise(close,configs)

    print(close.shape, preprocess.denoised.shape)

    import matplotlib.pyplot as plt
    plt.plot(preprocess.denoised, color='blue', label='Denoised')
    plt.plot(preprocess.normalized, color='red', label='Normalized')
    plt.grid(True)
    plt.title('WaveShink')
    plt.legend()
    plt.show()