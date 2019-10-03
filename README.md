# MoonLine
A Zipline-inspired backtesting implementation. MoonLine utilizes the vectorized Moonshot interface but transforms it into an event-driven one, making it much easier to port Zipline algorithms to Moonshot.

## Development

This project uses [poetry](https://poetry.eustace.io/) for development and release management.
```
$ git clone git@github.com:boosting-alpha-bv/moonline.git
$ cd moonline/
$ poetry install
```

### Strategy Development
Moonline files are self-contained and act like a regular Moonshot algorithm definition in QuantRocket. To install a Moonline algorithm, simply drop the file into the `moonshot` directory, after which it will become instantly accessible in the backtester.

Moonline also comes with a full-featured standalone backtester based on Moonshot, enabling you to run a backtest completely independently, from its singular Moonline file.

```python
class MyStrategy(MoonLineStrategy):

    # Specify the calendar we're generally active in
    CALENDAR = "NYSE"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Schedule a method to be called every day that can be traded
        self.schedule_interval(self.rebalance, "D", 1)

    def rebalance(self, data):
        # 'data' contains the stock data between now
        # and the last time the method was called
        print(data)

        # Order AAPL stocks for 30% of our total cash
        self.order_target_percent("AAPL", 0.3)

    # This method is called at the highest resolution available
    # It's mostly a convenience method and can be skipped with
    # an empty implementation like the below
    def handle_data(self, data):
        pass
```

## Usage
```bash
usage: moonline.py [-h] -i prices.csv -l listings.csv [-s YYYY-MM-DD]
                   [-e YYYY-MM-DD] [-o results.csv] [-w weights.csv] [-y]

optional arguments:
  -h, --help            show this help message and exit
  -i prices.csv, --input prices.csv
                        (required) A CSV file containing price data to
                        backtest on
  -l listings.csv, --listings listings.csv
                        The file containing InteractiveBrokers listings data
  -s YYYY-MM-DD, --start-date YYYY-MM-DD
                        The day to start the backtest from
  -e YYYY-MM-DD, --end-date YYYY-MM-DD
                        The day to end the backtest at
  -o results.csv, --output results.csv
                        The file to output backtest results to (default:
                        results.csv)
  -w weights.csv, --weights weights.csv
                        The file to save calculated weights to
  -y, --yes             If given, automatically answers script questions with
                        'yes'
```

### Running a Backtest
```bash
$ poetry run python moonline.py -i <price_data> -l <listings_data>
```
