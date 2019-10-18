### Data Handling ###
import pandas as pd

### Local ###
from .base import DataProvider


class QuantRocketDataProvider(DataProvider):
    """Reads price data from a CSV file with combined
       data for all tickers.
    """

    def __init__(self, file):
        super().__init__()
        self.file = file
        self.prices = None

    def ingest(self):
        prices = pd.read_csv(self.file)
        indexes = ["Field", "Date"]
        if "Time" in list(prices.columns):
            indexes = ["Field", "Date", "Time"]
        prices = prices.set_index(indexes).sort_index()
        prices.columns.name = "ConId"

        # Deduplicate index (apparently some QuantRocket exports contain duplicate rows)
        duplicates = prices.index.duplicated(keep="last")
        prices = prices.loc[duplicates == False, :]

        self.prices = prices
        self.available_symbols = self.list_symbols()

    def validate_symbols(self, symbols):
        if isinstance(symbols, list):
            missing_symbols = set(map(str, symbols)) - self.available_symbols
            if missing_symbols:
                raise Exception("The symbol{} {} {} not available".format("s" if len(missing_symbols) > 1 else "",
                                                                          ", ".join(missing_symbols),
                                                                          "are" if len(missing_symbols) > 1 else "is"))
        elif str(symbols) not in self.available_symbols:
            raise Exception("{} is not available".format(symbols))

    def list_symbols(self):
        return set(self.prices.columns)

    def get_ohlcv(self, symbols=None, start=None, end=None):
        if symbols:
            self.validate_symbols(symbols)
        else:
            symbols = self.available_symbols

        if isinstance(symbols, list):
            result = self.prices[map(str, symbols)]
        else:
            result = self.prices[[str(symbols)]]

        if start and end:
            result = result.loc[pd.IndexSlice[:, start.format(
                "YYYY-MM-DD HH:mm:ss"):end.format("YYYY-MM-DD HH:mm:ss"), :], :]
        elif start and not end:
            result = result.loc[pd.IndexSlice[:, start.format("YYYY-MM-DD HH:mm:ss"):, :], :]
        elif not start and end:
            result = result.loc[pd.IndexSlice[:, :end.format("YYYY-MM-DD HH:mm:ss"), :], :]

        return result
