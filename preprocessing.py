import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt


class PreProcessing:
    def denoise(self, series, thr_mode='garotte'):
        #WaveShrink
        cA, cD = pywt.dwt(series, 'haar')
        #threshold selection
        thr = np.std(series)/23
        cA_shrinked = pywt.threshold(cA, thr, mode=thr_mode)
        cD_shrinked = pywt.threshold(cD, thr, mode=thr_mode)
        #reconstructs data from the given shrinked coefficients
        series_rec = pywt.idwt(cA_shrinked, cD_shrinked, 'haar')

        if len(series_rec) > len(series):
            series_rec = series_rec[:-1]

        data_frame = {'original':series, 'denoised':series_rec}

        self.preprocessed = pd.DataFrame(data_frame)

    def to_csv(self):
        self.preprocessed.to_csv('./preprocessing/denoised-series.csv')

if __name__ == "__main__":
    stock_data = pd.read_csv('./AMZN.csv')
    close = stock_data.Close

    preprocess = PreProcessing()
    preprocess.denoise(close)

    import matplotlib.pyplot as plt
    plt.plot(preprocess.preprocessed.original, color='red')
    plt.plot(preprocess.preprocessed.denoised, color='blue')
    plt.legend(['Original prices', 'Denoised prices'])
    plt.grid(True)
    plt.show()