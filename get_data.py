import pandas_datareader.data as pdr
import fix_yahoo_finance as fix

# make pandas datareader optional
fix.pdr_override()


class GetData:
    def __init__(self, symbol, start, end):
        self.symbol = symbol
        self.start = start
        self.end = end
        self.file = symbol + ".csv"

    # get stock data
    def get_stock_data(self):
        self.stock_data = pdr.get_data_yahoo(self.symbol, self.start, self.end)
        self.stock_data.to_csv(self.file)


if __name__ == "__main__":
    data = GetData("NVDA", "2000-01-01", "2019-10-01")
    data.get_stock_data()
