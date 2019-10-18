### Local ###
from .base import DataProvider


class CSVBunchDataProvider(DataProvider):
    """Reads price data from a directory of CSV files
       where each file contains price data for one symbol.
    """

    def __init__(self, directory):
        # TODO: This should probably use Dask DataFrames:
        # https://docs.dask.org/en/latest/dataframe.html
        super().__init__()
        self.directory = directory
