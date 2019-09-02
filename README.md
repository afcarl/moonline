# MoonLine
A Zipline-inspired backtesting implementation. MoonLine utilizes the vectorized Moonshot interface but transforms it into an event-driven one, making it much easier to port Zipline algorithms to Moonshot.

## Usage
Moonline files are self-contained and act like a regular Moonshot algorithm definition in QuantRocket. To install a Moonline algorithm, simply drop the file into the `moonshot` directory, after which it will become instantly accessible in the backtester.

Moonline also comes with a rudimentary standalone backtester, enabling you to run a backtest completely independently, from its singular Moonline file.

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
