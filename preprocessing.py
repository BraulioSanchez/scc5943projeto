import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt


class PreProcessing:
    def __init__(self, split, feature_split, window, slide, file):
        self.split = split
        self.feature_split = feature_split
        self.window = window
        self.slide = slide
        self.stock_data = pd.read_csv(file)

    # wavelet transform and create autoencoder data
    def make_wavelet_train(self, wavelet='haar', thr_mode='garotte'):
        train_data = []
        test_data = []
        log_train_data = []

        #for i in range((len(self.stock_data)//10)*10 - 11):
        columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']
        stock = [[], [], [], [], []]
        i = 0
        while i <= len(self.stock_data) - (self.window-1):
            train = []
            log_ret = []
            for j in range(1, 6): #Open, High, Low, Close, Adj Close
                #x = np.array(self.stock_data.iloc[i: i + 11, j])
                x = np.array(self.stock_data.iloc[i: i + self.window, j])

                #WaveShrink
                (cA, cD) = pywt.dwt(x, wavelet)
                #threshold selection
                n = 1/32
                cA_treshold = n * np.std(cA)
                cD_treshold = n * np.std(cD)
                cA_shrinked = pywt.threshold(cA, cA_treshold, mode=thr_mode)
                cD_shrinked = pywt.threshold(cD, cD_treshold, mode=thr_mode)
                #reconstructs data from the given shrinked coefficients
                x_rec = pywt.idwt(cA_shrinked, cD_shrinked, wavelet)
                if len(x_rec) > self.window:
                    x_rec = x_rec[:-1]

                stock[j-1] = stock[j-1].__add__(x_rec.tolist())

                #Log Return
                log = np.diff(np.log(x_rec))*100

                #Moving Average Convergence Divergence with 5 days difference
                macd = np.mean(x[5:]) - np.mean(x)

                sd = np.std(x)

                log_ret = np.append(log_ret, log)
                x_tech = np.append(macd*10, sd)
                train = np.append(train, x_tech)

            i += self.window
            train_data.append(train)
            log_train_data.append(log_ret)

        plt.plot(self.stock_data.Close)
        plt.plot(stock[4])
        plt.show()
        trained = pd.DataFrame(train_data)
        trained.to_csv("preprocessing/indicators.csv")
        log_train = pd.DataFrame(log_train_data, index=None)
        log_train.to_csv("preprocessing/log_train.csv")
        rbm_train = pd.DataFrame(log_train_data[0:int(self.split*self.feature_split*len(log_train_data))], index=None)
        rbm_train.to_csv("preprocessing/rbm_train.csv")
        rbm_test = pd.DataFrame(log_train_data[int(self.split*self.feature_split*len(log_train_data))+1:
                                               int(self.feature_split*len(log_train_data))])
        rbm_test.to_csv("preprocessing/rbm_test.csv")
        #for i in range((len(self.stock_data) // 10) * 10 - 11):
        windows = (len(self.stock_data)//self.slide)*self.slide-self.window
        i = 0
        while i < windows:
            y = 100 * np.log(self.stock_data.Close[i + 11] / self.stock_data.Close[i + 10])
            #y = 100 * np.log(self.stock_data.iloc[i + 11, 5] /
            #                 self.stock_data.iloc[i + 10, 5])
            test_data.append(y)

            i += self.slide
        test = pd.DataFrame(test_data)
        test.to_csv("preprocessing/test_data.csv")

    def make_test_data(self):
        test_stock = []
        # stock_data_test = pd.read_csv("stock_data_test.csv", index_col=0)

        windows = (len(self.stock_data)//self.slide) * self.slide - self.window
        #for i in range((len(self.stock_data) // 10) * 10 - 11):
        i = 0
        while i < windows:
            #l = self.stock_data.iloc[i+11, 5]
            l = self.stock_data.Close[i+self.window] #Close
            test_stock.append(l)
            test = pd.DataFrame(test_stock)
            test.to_csv("preprocessing/test_stock.csv")

            i += self.slide

        stock_test_data = np.array(test_stock)[int(self.feature_split*len(test_stock) +
                                               self.split*(1-self.feature_split)*len(test_stock)):]
        stock = pd.DataFrame(stock_test_data, index=None)
        stock.to_csv("stock_data_test.csv")

if __name__ == "__main__":
    preprocess = PreProcessing(0.8, 0.25, 11, 1, 'MSFT.csv')
    preprocess.make_wavelet_train()
    preprocess.make_test_data()

'''
close = amzn.Close
#close = ts
ca, cd = pywt.dwt(close, 'haar')
k = .9
m = .9
deviation = 0.06
#deviation = 0.007859421109793626
mode = 'garotte'
#thr = deviation * math.sqrt(2*math.log(len(close),2))
thr = 1*np.std(close)/32
cat = pywt.threshold(ca, thr, mode=mode)
cdt = pywt.threshold(cd, thr, mode=mode)
close_rec = pywt.idwt(cat, cdt, 'haar')

plt.close('all')
plt.subplot(211)
plt.grid('on')
plt.plot(close, color='red')
plt.plot(close_rec, color='blue')

k = .8
m = .9
#deviation = 0.06
deviation = 0.007859421109793626
mode = 'soft'
#thr = deviation * math.sqrt(2*math.log(len(close),2))
thr = 1*np.std(close)
cat = pywt.threshold(ca, thr, mode=mode)
cdt = pywt.threshold(cd, thr, mode=mode)
close_rec = pywt.idwt(cat, cdt, 'haar')

plt.subplot(212)
plt.grid('on')
plt.plot(close, color='red')
plt.plot(close_rec, color='blue')

plt.show()
'''