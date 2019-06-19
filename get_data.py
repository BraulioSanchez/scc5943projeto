import pandas_datareader.data as pdr
import fix_yahoo_finance as fix
import os

# make pandas datareader optional
#fix.pdr_override()


class GetData:
    def __init__(self, symbol, start, end, configs):
        self.symbol = symbol
        self.start = start
        self.end = end

        if not os.path.exists(configs['data']['save_dir']):
            os.makedirs(configs['data']['save_dir'])

        self.filename = os.path.join(configs['data']['save_dir'], symbol + ".csv")  

    # get stock data
    def get_stock_data(self):
        self.stock_data = pdr.get_data_yahoo(self.symbol, self.start, self.end)
        self.stock_data.to_csv(self.filename)


if __name__ == "__main__":
    import json
    configs = json.load(open('config.json', 'r'))

    data = GetData(configs['data']['symbol'],
                    configs['data']['start'],
                    configs['data']['end'],
                    configs)
    data.get_stock_data()
