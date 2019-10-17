### System ###
from abc import ABC, abstractmethod


class DataProvider(ABC):
    """Abstract base class for loading price data in a unified format.
    The specific subclass is responsible for finding the data,
    loading it performantly and possibly caching it.
    """

    def ingest(self):
        """Prepares data for usage by ingesting it and converting it into an efficient format."""
        pass

    def validate_symbols(self, symbols):
        """Should raise an exception if any given symbol has no data available."""
        pass

    def list_symbols(self):
        """Lists all available symbols in this DataProvider."""
        pass

    def range(self):
        """Finds the earliest and latest date for which data can be supplied."""
        pass

    def get_ohlcv(self, symbols=None, start=None, end=None):
        """Returns a properly formatted DataFrame of OHLCV data for use within MoonLine.

        Args:
            symbols (list): If given, returns a subset of the data for the given symbols
            start: If given, clips the data from the start
            end: If given, clips the data from the end
        """
        pass

    def get_fundamentals(self, symbols=None, start=None, end=None):
        """Returns a properly formatted DataFrame of fundamentals data for use within MoonLine.

        Args:
            symbols (list): If given, returns a subset of the data for the given symbols
            start: If given, clips the data from the start
            end: If given, clips the data from the end
        """
        pass
