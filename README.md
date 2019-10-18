# MoonLine
A Zipline-inspired backtesting implementation. MoonLine utilizes the vectorized Moonshot interface but transforms it into an event-driven one, making it much easier to port Zipline algorithms to Moonshot.

## Disclaimer
MoonLine is in active (and early) development, so bugs are to be expected. If you have an issue that you suspect may be a bug (or a feature request), please log an issue in the [Issue Tracker](https://github.com/boosting-alpha-bv/moonline/issues). Specifically, intraday backtesting has not been tested much, so keep an eye out if you experience weird behavior when backtesting on data with higher resolution than End-of-Day bars.

## Development
This project uses [poetry](https://poetry.eustace.io/) for development and release management.
```
$ git clone git@github.com:boosting-alpha-bv/moonline.git
$ cd moonline/
$ poetry install
```

If you do not want to use `poetry`, you can install the requirements manually via `pip` by looking them up in the [pyproject.toml](https://github.com/boosting-alpha-bv/moonline/blob/master/pyproject.toml#L11-L21) file.

### Running
To run MoonLine using `poetry`, use the following command:
```bash
$ poetry run python moonline.py -q <price_data> -l <listings_data> <...optional arguments>
```

Alternatively you can just run it with your regular Python installation. However, **beware of the fact that MoonLine is being developed on Python 3.7** and as such, lower version may not work. It is recommended to use [pyenv](https://github.com/pyenv/pyenv) to manage Python versions on your local machine.

### Generating Documentation
```bash
$ poetry run pdoc --config show_type_annotations=True --html --force moonline.py
```

We are currently in the process of documenting the entire program, but for the moment several parts are undocumented.

### Working on Documentation
`pdoc` can be switched into hot-reload mode by appending the following:
```bash
$ poetry run pdoc --config show_type_annotations=True --html --force moonline.py --http :
```

This will open a webserver on `localhost:8080` which will auto-reload whenever a change is made to docstrings in the observed modules.

### Strategy Development
MoonLine files are self-contained and act like a regular Moonshot algorithm definition in [QuantRocket](https://www.quantrocket.com/). To install a MoonLine algorithm, simply drop the file into the `moonshot` directory, after which it will become instantly accessible in the backtester.

MoonLine also comes with a **full-featured standalone backtester** based on Moonshot, enabling you to run a backtest completely independently, from its singular MoonLine file. This includes data ingestion, backtesting and tear sheet generation.

```python
class MyStrategy(MoonLineStrategy):

    # Specify the calendar we're generally active in
    CALENDAR = "NYSE"
    AAPL = Asset("AAPL", "NASDAQ")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Schedule a method to be called every day (that can be traded)
        self.schedule_interval(self.rebalance, "D", 1)

    def rebalance(self, data):
        # 'data' contains the stock data between now
        # and the last time the method was called
        print(data)
        >>> Field                                              Close    High     Low     Open    Volume
        >>> ConId
        >>> Asset(ConId=265598, Symbol=AAPL, Exchange=NA... 243.6414  31.240  29.210  29.2500  979300.0

        # Order AAPL stocks for 30% of our total cash
        self.order_target_percent("AAPL", 0.3)

    # This method is called at the highest resolution available
    # It's mostly a convenience method and can be skipped with
    # an empty implementation like the one below
    def handle_data(self, data):
        pass
```

### On Data
MoonLine supports multiple input formats for data, which will be expanded in the future as well. Currently, it is advised to use the QuantRocket-inspired format as it is the most robust.  
This format of data is ingested via the `-q` switch and consists of a singular CSV file containing the price data for one or more assets. MoonLine supports both end-of-day as well as intraday input files. These differ mainly in that intraday price files contain an additional index level called "Time".  
As long as it is possible to create a CSV file with the structure as shown below, MoonLine will be able to ingest it and use the contained data for backtesting.

#### Asset Names and ConId's
Since MoonLine is intended to be used with QuantRocket, all assets have to be referred to by their ConId (Contract ID) as given by InteractiveBrokers. This ID is unique for any given Stock-PrimaryExchange pair and is used internally to resolve the valid exchanges, trading calendars and available trading hours. To make resolving the Name-ConId relationship easier, a custom library called [quantrocket-utils](https://github.com/boosting-alpha-bv/quantrocket-utils) is used which provides an `Asset` class which does the heavy lifting for us. See its repository for usage instructions of this class.

All references to assets in MoonLine are made via these `Asset` objects and they are also required to actually be able to order stocks.

They can be created either by giving a symbol name and an exchange or a ConId:
```python
spy = Asset("SPY", "ARCA")
# or
# the below is only possible if no conflicting exchanges are found and may raise an Exception
spy = Asset("SPY")
# or (NEW)
# same as the above, but will not throw an Exception
# this also comes with the caveat that no metadata will be gathered for this object
spy = Asset("SPY", ignore_exchange=True)
# or
spy = Asset(756733)
# or
spy = Asset("756733")
```

**Notice:** It is inteded to slowly transition away from this dependency on InteractiveBrokers, however this might take a while.
It is currently possibly to test-drive the removal of InteractiveBrokers ConId's by setting `ignore_exchange=True` during Asset creation.
The currently implemented sample algorithm uses this InteractiveBrokers-free method to showcase how it works.  
It is important to note that in case of using `ignore_exchange=True`, the input data's column names must be symbol names instead of ConId's.

Because of this dependency, MoonLine requires the full listings catalogue of InteractiveBrokers to reference asset metadata at runtime. This is ingested via the `-l` switch.

#### End-of-Day Sample
| Field    | Date          | 265598        |
| -------- | ------------: | ------------: |
| Close    | 2009-05-20    | 73.305        |
| Close    | 2009-05-21    | 74.764        |
| Close    | 2009-05-22    | 75.845        |
| High     | 2009-05-20    | 169.74        |
| High     | 2009-05-21    | 170.65        |
| High     | 2009-05-22    | 171.34        |
| Low      | 2009-05-20    | 137.52        |
| Low      | 2009-05-21    | 138.54        |
| Low      | 2009-05-22    | 139.25        |
| Open     | 2009-05-20    | 289.41        |
| Open     | 2009-05-21    | 290.76        |
| Open     | 2009-05-22    | 291.23        |
| Volume   | 2009-05-20    | 73292100.0    |
| Volume   | 2009-05-21    | 73291931.0    |
| Volume   | 2009-05-22    | 73234500.0    |

```
Field,Date,265598
Close,2009-05-20,73.305
Close,2009-05-21,74.764
Close,2009-05-22,75.845
High,2009-05-20,169.74
High,2009-05-21,170.65
High,2009-05-22,171.34
Low,2009-05-20,137.52
Low,2009-05-21,138.54
Low,2009-05-22,139.25
Open,2009-05-20,289.41
Open,2009-05-21,290.76
Open,2009-05-22,291.23
Volume,2009-05-20,73292100.0
Volume,2009-05-21,73291931.0
Volume,2009-05-22,73234500.0
```

#### Intraday Sample
| Field    | Date          | Time        | 265598        |
| -------- | ------------: | ----------: | ------------: |
| Close    | 2009-05-20    | 15:45:00    | 73.305        |
| Close    | 2009-05-20    | 16:00:00    | 74.764        |
| Close    | 2009-05-20    | 16:15:00    | 75.845        |
| High     | 2009-05-20    | 15:45:00    | 169.74        |
| High     | 2009-05-20    | 16:00:00    | 170.65        |
| High     | 2009-05-20    | 16:15:00    | 171.34        |
| Low      | 2009-05-20    | 15:45:00    | 137.52        |
| Low      | 2009-05-20    | 16:00:00    | 138.54        |
| Low      | 2009-05-20    | 16:15:00    | 139.25        |
| Open     | 2009-05-20    | 15:45:00    | 289.41        |
| Open     | 2009-05-20    | 16:00:00    | 290.76        |
| Open     | 2009-05-20    | 16:15:00    | 291.23        |
| Volume   | 2009-05-20    | 15:45:00    | 73292100.0    |
| Volume   | 2009-05-20    | 16:00:00    | 73291931.0    |
| Volume   | 2009-05-20    | 16:15:00    | 73234500.0    |

```
Field,Date,Time,265598
Close,2009-05-20,15:45:00,73.305
Close,2009-05-20,16:00:00,74.764
Close,2009-05-20,16:15:00,75.845
High,2009-05-20,15:45:00,169.74
High,2009-05-20,16:00:00,170.65
High,2009-05-20,16:15:00,171.34
Low,2009-05-20,15:45:00,137.52
Low,2009-05-20,16:00:00,138.54
Low,2009-05-20,16:15:00,139.25
Open,2009-05-20,15:45:00,289.41
Open,2009-05-20,16:00:00,290.76
Open,2009-05-20,16:15:00,291.23
Volume,2009-05-20,15:45:00,73292100.0
Volume,2009-05-20,16:00:00,73291931.0
Volume,2009-05-20,16:15:00,73234500.0
```

## Usage
```bash
usage: moonline.py [-h] [-q prices.csv] [-p dir] -l listings.csv
                   [-s YYYY-MM-DD] [-e YYYY-MM-DD] [-o results.csv]
                   [-w weights.csv] [-y] [-c]

optional arguments:
  -h, --help            show this help message and exit
  -q prices.csv, --quantrocket prices.csv
                        A CSV file containing quantrocket-formatted price data
  -p dir, --pystore dir
                        The directory containing PyStore data
  -l listings.csv, --listings listings.csv
                        The file containing InteractiveBrokers listings data
  -s YYYY-MM-DD, --start-date YYYY-MM-DD
                        The day to start the backtest from
  -e YYYY-MM-DD, --end-date YYYY-MM-DD
                        The day to end the backtest at
  -o results.csv, --output results.csv
                        The file to output backtest results to
  -w weights.csv, --weights weights.csv
                        The file to save calculated weights to
  -y, --yes             If given, automatically answers script questions with
                        'yes'
  -c, --clear-cache     If given, ignores cached data
```

### On Speed
MoonLine makes heavy use of caching to speed up similar runs. As such, it is natural that the first run of a specific backtest takes a while longer than the subsequent ones. It automatically maintains its cache and invalidates it if the backtest parameters change, so no user input or maintenance is required on this front.

If you'd still like to do a completely un-cached run, use the `-c` flag.
