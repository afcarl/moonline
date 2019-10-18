### Local ###
from .base import DataProvider


class MarketstoreDataProvider(DataProvider):
    """Reads price data from a Marketstore database.
    """

    def __init__(self, url):
        super().__init__()
        self.url = url

    """
    with timeit("Loading price data from PyStore"):
        client = pms.Client(args.database)

        symbol_data_files = {}

        requested_assets = {asset.symbol: str(asset.conid) for asset in Asset}

        if args.start_date and args.end_date:
            result = client.query(pms.Params(list(requested_assets.keys()), "1D", "OHLCV",
                                             start=str(args.start_date), end=str(args.end_date))).all()
        elif args.start_date and not args.end_date:
            result = client.query(pms.Params(list(requested_assets.keys()), "1D",
                                             "OHLCV", start=str(args.start_date))).all()
        elif not args.start_date and args.end_date:
            result = client.query(pms.Params(list(requested_assets.keys()),
                                             "1D", "OHLCV", end=str(args.end_date))).all()

        for symbol, data in result.items():
            symbol = symbol.split("/")[0]
            symbol_data_files[requested_assets[symbol]] = data.df()

        proto_df = {}
        for symbol, data in symbol_data_files.items():
            for index, row in data.iterrows():
                date = arrow.get(index, tzinfo="Europe/London")
                proto_df[(symbol, date.format("YYYY-MM-DD"))] = row.values

        prices = pd.DataFrame.from_dict(proto_df, orient="index", columns=[
                                        "Open", "High", "Low", "Close", "Volume"])
        prices.index = pd.MultiIndex.from_tuples(prices.index)
        prices.index.names = ["ConId", "Date"]
        prices.columns.name = "Field"
        prices = prices.stack().unstack("ConId").swaplevel(0, 1).sort_index()
    """
