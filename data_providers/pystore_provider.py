### Data Handling ###
import pystore
import pandas as pd

### Local ###
from .base import DataProvider


class PyStoreDataProvider(DataProvider):
    """Reads price data from a PyStore database.
    """

    def __init__(self, directory):
        super().__init__()
        import pystore
        self.directory = directory
        pystore.set_path(self.directory)
        self.ohlcv_store = pystore.store("OHLCV")
        self.fd_store = pystore.store("FD")
        self.ohlcv_eod_collection = self.ohlcv_store.collection("EOD")
        self.fd_q_collection = self.fd_store.collection("Q")
        self.available_symbols = self.list_symbols()

    def list_symbols(self, source=None):
        if source:
            return self.ohlcv_eod_collection.list_items(source=source)
        return self.ohlcv_eod_collection.list_items()

    def validate_symbols(self, symbols):
        if isinstance(symbols, list):
            missing_symbols = set(symbols) - self.available_symbols
            if missing_symbols:
                raise Exception("The symbol{} {} {} not available".format("s" if len(missing_symbols) > 1 else "",
                                                                          ", ".join(missing_symbols),
                                                                          "are" if len(missing_symbols) > 1 else "is"))
        elif symbols not in self.available_symbols:
            raise Exception("{} is not available".format(symbols))

    def get_ohlcv(self, symbols=None, start=None, end=None):
        if symbols:
            self.validate_symbols(symbols)
        else:
            symbols = self.available_symbols

        result = pd.DataFrame()
        if isinstance(symbols, list):
            for symbol in symbols:
                df = self.ohlcv_eod_collection.item(symbol).to_pandas()
                result[symbol] = df.stack()
        else:
            df = self.ohlcv_eod_collection.item(symbols).to_pandas()
            result[symbols] = df.stack()

        result.columns.name = "ConId"
        result.index.names = ["Date", "Field"]
        result = result.swaplevel(0, 1).sort_index()

        if start and end:
            result = result.loc[pd.IndexSlice[:, str(start):str(end), :], :]
        elif start and not end:
            result = result.loc[pd.IndexSlice[:, str(start):, :], :]
        elif not start and end:
            result = result.loc[pd.IndexSlice[:, :str(end), :], :]

        result.index = result.index.map(lambda x: (x[0].capitalize(),) + x[1:])

        return result

    def get_fundamentals(self, symbols=None, start=None, end=None):
        if symbols:
            self.validate_symbols(symbols)
        else:
            symbols = self.available_symbols

        result = pd.DataFrame()
        if isinstance(symbols, list):
            for symbol in symbols:
                df = self.fd_q_collection.item(symbols).to_pandas()
                result[symbol] = df.stack()
        else:
            df = self.fd_q_collection.item(symbols).to_pandas()
            result[symbols] = df.stack()

        if start and end:
            result = result.loc[pd.IndexSlice[str(start):str(end), ], :]
        elif start and not end:
            result = result.loc[pd.IndexSlice[str(start):, ], :]
        elif not start and end:
            result = result.loc[pd.IndexSlice[:str(end), ], :]

        result.index = result.index.map(lambda x: (x[0].capitalize(),) + x[1:])

        return result
