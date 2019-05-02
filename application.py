from get_data import GetData
from preprocessing import PreProcessing

ticker = 'CTAS'
start = '2000-01-01'
end = '2019-10-01'
data = GetData(ticker, start, end)
data.get_stock_data()

close = data.stock_data.Close
preprocess = PreProcessing()
preprocess.denoise(close)
preprocess.to_csv()

import matplotlib.pyplot as plt
plt.plot(preprocess.preprocessed.original, color='red')
plt.plot(preprocess.preprocessed.denoised, color='blue')
plt.legend(['Original prices', 'Denoised prices'])
plt.grid(True)
plt.show()