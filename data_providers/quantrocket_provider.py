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
        if args.start_date and args.end_date:
            prices = prices.loc[pd.IndexSlice[:, str(args.start_date):str(args.end_date), :], :]
        elif args.start_date and not args.end_date:
            prices = prices.loc[pd.IndexSlice[:, str(args.start_date):, :], :]
        elif not args.start_date and args.end_date:
            prices = prices.loc[pd.IndexSlice[:, :str(args.end_date), :], :]
        prices.columns.name = "ConId"

        # Figure out the first row where all column values are present
        # first_no_nan_date = prices.loc[~prices.isnull().sum(1).astype(bool)].iloc[0].name[1]

        # Deduplicate index (apparently some QuantRocket exports contain duplicate rows)
        duplicates = prices.index.duplicated(keep="last")
        prices = prices.loc[duplicates == False, :]

        self.prices = prices
