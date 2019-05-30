import numpy as np
import pandas as pd
import pywt
from sklearn.preprocessing import MinMaxScaler


class PreProcessing:
    def denoise(self, series, thr_mode='garotte'):
        #performs scale
        scaler = MinMaxScaler(feature_range=(-1,1))
        series_sc = scaler.fit_transform(np.array(series).reshape(-1,1)).reshape(series.shape)

        #WaveShrink
        cA, cD = pywt.dwt(series_sc, 'haar')
        #threshold selection
        thr = np.std(series_sc)/23
        cA_shrinked = pywt.threshold(cA, thr, mode=thr_mode)
        cD_shrinked = pywt.threshold(cD, thr, mode=thr_mode)
        #reconstructs data from the given shrinked coefficients
        series_rec = pywt.idwt(cA_shrinked, cD_shrinked, 'haar')

        if len(series_rec) > len(series_sc):
            series_rec = series_rec[:-1]

        data_frame = {'scaled':series_sc, 'denoised':series_rec}

        self.preprocessed = pd.DataFrame(data_frame)

    def to_csv(self):
        self.preprocessed.to_csv('./preprocessing/denoised-series.csv', index=False)

if __name__ == "__main__":
    stock_data = pd.read_csv('./MSFT.csv')
    close = stock_data.Close

    preprocess = PreProcessing()
    preprocess.denoise(close)

    import matplotlib.pyplot as plt
    plt.plot(preprocess.preprocessed.scaled, color='red')
    plt.plot(preprocess.preprocessed.denoised, color='blue')
    plt.legend(['Scaled prices', 'Denoised prices'])
    plt.grid(True)
    plt.show()