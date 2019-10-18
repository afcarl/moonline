### Warnings ###
import warnings
warnings.filterwarnings("ignore")

### System ###
import os
import csv
import pickle
import datetime
from glob import glob
from enum import Enum, IntEnum
from abc import ABC, abstractmethod
from collections import defaultdict

### Local ###
try:
    from codeload.lib.quantrocket_utils import initialize as assets_init, Asset, timeit, is_quantrocket
    with timeit("Loading Listings"):
        assets_init("/codeload/satellite_scripts/script_data/listings.csv")
except ImportError:
    from quantrocket_utils import initialize as assets_init, Asset, timeit, is_quantrocket

### Display ###
from tqdm import tqdm

### Data Handling ###
import numpy as np
import pandas as pd

### Date Handling ###
import arrow

### QuantRocket ###
from moonshot import Moonshot
from moonshot.commission import PercentageCommission
try:
    import ib_trading_calendars as tc
except:
    import trading_calendars as tc


class SchedulerMode(Enum):
    """Modes that a TimeScheduler can operate in.

    Attributes:
        INTERVAL: Triggers on specific intervals, i.e. Daily, Monthly or Yearly.
        DATETIME: Triggers when both the date and time components of a given
            datetime object occur.
        DATE: Triggers when the date component of a given date object occurs.
        TIME: Triggers when the time component of a given time object occurs.
    """
    INTERVAL = 1
    DATETIME = 2
    DATE = 3
    TIME = 4


class TimeScheduler():
    """Schedules a given method to run at a specific date and/or time.

    Attributes:
        mode (SchedulerMode): A SchedulerMode instance.
        method (function): The method to run.
        calendar: The trading calendar this scheduler is active in.
        timezone: The timezone this scheduler is active in.
    """

    def __init__(self, data: dict, mode: SchedulerMode):
        """Internal constructor for TimeScheduler objects.

        Note:
            `__init__` is private in this case and only for internal use.
            Refer to the supplied class methods for object creation.

        Args:
            data (dict): A dictionary of data to initialize the object.
                This is generally created by the static class methods.
            mode (SchedulerMode): The mode this scheduler should operate in.
        """
        self.mode = mode
        self.method = data.get("method")
        self.calendar = tc.get_calendar(data.get("calendar"))
        self.timezone = self.calendar.tz

        self.has_date = True
        self.has_time = False
        self.datetime = None

        if self.mode == SchedulerMode.INTERVAL:
            if data.get("time"):
                self.has_time = True
                self.datetime = arrow.get(data.get("time"), "HH:mm:ss")
            self.interval = data.get("interval")
            self.frequency = data.get("frequency")
            self.interval_list = defaultdict(set)
            self.frequency_counter = 0
        elif self.mode == SchedulerMode.DATETIME:
            self.has_time = True
            self.datetime = data.get("datetime")
        elif self.mode == SchedulerMode.DATE:
            self.datetime = data.get("date")
        elif self.mode == SchedulerMode.TIME:
            self.has_date = False
            self.has_time = True
            self.datetime = data.get("time")

        if self.datetime:
            self.datetime = self.datetime.replace(tzinfo=self.timezone).to("UTC")

    @classmethod
    def schedule_datetime(cls, method, date: str, time: str, calendar: str="NYSE"):
        """Schedule a method to run at a specific date and time.

        Example:
            ```python
            TimeScheduler.schedule_datetime("2018-01-01", "10:30:00")
            ```
        """
        datetime = arrow.get("{} {}".format(date, time), "YYYY-MM-DD HH:mm:ss")
        data = {
            "datetime": datetime,
            "calendar": calendar,
            "method": method,
        }
        return cls(data, mode=SchedulerMode.DATETIME)

    @classmethod
    def schedule_date(cls, method, date: str, calendar: str="NYSE"):
        """Schedule a method to run at a specific date.

        Example:
            ```python
            TimeScheduler.schedule_date("2018-01-01")
            ```
        """
        date = arrow.get(date, "YYYY-MM-DD")
        data = {
            "date": date,
            "calendar": calendar,
            "method": method,
        }
        return cls(data, mode=SchedulerMode.DATE)

    @classmethod
    def schedule_time(cls, method, time: str, calendar: str="NYSE"):
        """Schedule a method to run at a specific time.

        Example:
            ```python
            TimeScheduler.schedule_time("10:30:00")
            ```
        """
        time = arrow.get(time, "HH:mm:ss")
        data = {
            "time": time,
            "calendar": calendar,
            "method": method,
        }
        return cls(data, mode=SchedulerMode.TIME)

    @classmethod
    def schedule_interval(cls, method, interval: str, frequency: int, time: str=None, calendar: str="NYSE"):
        """Schedule a method to run at a specific interval and optional time.

        Example:
            ```python
            TimeScheduler.schedule_interval("D", 3)
            ```
        """
        data = {
            "interval": interval,
            "time": time,
            "frequency": frequency,
            "calendar": calendar,
            "method": method,
        }
        return cls(data, mode=SchedulerMode.INTERVAL)

    def time_check(self, time: str, current_datetime: datetime.datetime) -> bool:
        """Checks whether the current time is within market hours.

        Returns:
            bool: The return value. True for success, False otherwise.
        """
        market_open, market_close = self.calendar.schedule.loc[current_datetime.date()]
        market_open, market_close = arrow.get(market_open), arrow.get(market_close)
        if time:
            # If we have been given data with a time index
            if not market_open.time() <= current_datetime.time() <= market_close.time():
                # Current time is not in open range, either too early or too late
                return False
        if self.has_time:
            # If we have defined a specific time to trigger at
            if not market_open.time() <= self.datetime.time() <= market_close.time():
                # Trigger time is not in open range, either too early or too late
                return False
            if self.datetime.time() != current_datetime.time():
                # Current time is not equal to trigger time, so no match
                return False
        return True

    def check(self, time: str, current_datetime: datetime.datetime) -> bool:
        if current_datetime.date() in self.calendar.schedule.index:
            # Exchange is open
            if time and not self.time_check(time, current_datetime):
                return False
            if self.frequency_counter % self.frequency != 0:
                self.frequency_counter += 1
                return False
            self.frequency_counter += 1
            self.interval_list[current_datetime.year].add(current_datetime.month)
            return True
        else:
            # Exchange is closed
            return False

    def should_run(self, date: str, time: str=None) -> bool:
        if time:
            current_datetime = arrow.get("{} {}".format(date, time), "YYYY-MM-DD HH:mm:ss")
        else:
            current_datetime = arrow.get(date, "YYYY-MM-DD")
        current_datetime = current_datetime.replace(tzinfo=self.timezone).to("UTC")

        if self.mode == SchedulerMode.INTERVAL:
            # Interval Scheduling
            if self.interval == "Y":
                if current_datetime.year in self.interval_list:
                    # We've already triggered this year
                    return False
                first_trading_day = arrow.get(self.calendar.schedule[str(current_datetime.year)].iloc[0][0])
                if not current_datetime.date() == first_trading_day.date():
                    return False
                return self.check(time, current_datetime)
            elif self.interval == "M":
                if current_datetime.month in self.interval_list[current_datetime.year]:
                    # We've already triggered this month
                    return False
                first_trading_day = arrow.get(self.calendar.schedule[current_datetime.format("YYYY-MM")].iloc[0][0])
                if not current_datetime.date() == first_trading_day.date():
                    return False
                return self.check(time, current_datetime)
            elif self.interval == "D":
                if current_datetime.day in self.interval_list:
                    # We've already triggered today
                    return False
                return self.check(time, current_datetime)
        else:
            # Normal scheduling
            if self.mode == SchedulerMode.DATETIME:
                return self.datetime == current_datetime
            elif self.mode == SchedulerMode.DATE:
                return self.datetime.date() == current_datetime.date()
            elif self.mode == SchedulerMode.TIME:
                return self.datetime.time() == current_datetime.time()

    def run(self, *args, **kwargs):
        self.method(*args, **kwargs)

    def __repr__(self):
        if self.mode == SchedulerMode.INTERVAL:
            if self.frequency == 1:
                interval_name_map = {"Y": "Yearly", "M": "Monthly", "D": "Daily"}
                if self.has_time:
                    return "TimeScheduler({} at {})".format(interval_name_map[self.interval], self.datetime.time())
                else:
                    return "TimeScheduler({})".format(interval_name_map[self.interval])
            else:
                interval_name_map = {"Y": "years", "M": "months", "D": "days"}
                if self.has_time:
                    return "TimeScheduler(Every {} {} at {})".format(self.frequency, interval_name_map[self.interval], self.datetime.time())
                else:
                    return "TimeScheduler(Every {} {})".format(self.frequency, interval_name_map[self.interval])
        elif self.mode == SchedulerMode.DATETIME:
            return "TimeScheduler({})".format(self.datetime.format("YYYY-MM-DD HH:mm:ss"))
        elif self.mode == SchedulerMode.DATE:
            return "TimeScheduler({})".format(self.datetime.format("YYYY-MM-DD"))
        elif self.mode == SchedulerMode.TIME:
            return "TimeScheduler({})".format(self.datetime.format("HH:mm:ss"))


class HistoryFrequency(IntEnum):
    """Frequencies that a History object can supply.

    TODO: Description
    """
    SECOND_1 = 1
    SECOND_5 = 5
    SECOND_10 = 10
    SECOND_15 = 15
    SECOND_30 = 30
    MINUTE_1 = 60
    MINUTE_2 = 120
    MINUTE_3 = 180
    MINUTE_5 = 300
    MINUTE_10 = 600
    MINUTE_15 = 900
    MINUTE_20 = 1200
    MINUTE_30 = 1800
    HOUR_1 = 3600
    HOUR_2 = 7200
    HOUR_3 = 10800
    HOUR_4 = 14400
    HOUR_8 = 28800
    DAY_1 = 86400
    WEEK_1 = 604800
    MONTH_1 = 2629800

    def __init__(self, seconds):
        self.seconds = seconds

    def time_string(self):
        second_time_string_map = {1: "1s", 5: "5s", 10: "10s", 15: "15s", 30: "30s", 60: "1Min", 120: "2Min",
                                  180: "3Min", 300: "5Min", 600: "10Min", 900: "15Min", 1200: "20Min", 1800: "30Min",
                                  3600: "1H", 7200: "2H", 10800: "3H", 14400: "4H", 28800: "8H", 86400: "1D",
                                  604800: "1W", 2629800: "1M"}
        return second_time_string_map[self.seconds]


class History():

    def __init__(self, data, default_frequency):
        self.data = data.unstack(level=1)
        self.default_frequency = default_frequency
        self.current_datetime = None
        self.start_date = data.index.levels[0][0]
        self.end_date = data.index.levels[0][-1]

    def update_time(self, datetime):
        self.current_datetime = pd.to_datetime(datetime.isoformat(), format="%Y-%m-%dT%H:%M:%S.%f")

    def current(self):
        """Return the current set of values at algorithm time."""
        return self.history(1)

    def history(self, bar_count, frequency=None):
        """Return a window of 'bar_count' bars into the past, starting from algorithm time.

        Data will automatically be resampled if the base data is not in the correct format.
        Missing data is filled using the fill-forward method.
        Resampling from large strides into small ones can be RAM intensive (i.e. Daily source bars to second bars).
        """
        if not frequency:
            frequency = self.default_frequency

        if bar_count <= 0:
            raise Exception("You must request a positive amount of bars. Bars requested: {}".format(bar_count))

        data = self.data.resample(frequency.time_string(), level=0).last().ffill().stack(level=1)
        # data = self.data.groupby([pd.Grouper(level="Datetime", freq=frequency.time_string()), pd.Grouper(level="ConId")]).last()
        # sliced = data.loc[self.current_datetime -
        # pd.Timedelta(seconds=frequency.seconds * bar_count -
        # 1):self.current_datetime]
        sliced_index = data.index.levels[0].to_series(keep_tz=True)[:self.current_datetime]
        index_values = pd.DatetimeIndex(sliced_index)[-bar_count:]
        sliced = data.loc[pd.IndexSlice[index_values, :], :]

        new_tuples = []
        reformatted_data = sliced.copy()
        index_values = reformatted_data.index.tolist()
        if frequency < HistoryFrequency.DAY_1:
            for datetime, conid in index_values:
                new_tuples.append((datetime.date(), datetime.time(), conid))
            reformatted_data.index = pd.MultiIndex.from_tuples(new_tuples, names=["Date", "Time", "ConId"])
        else:
            for datetime, conid in index_values:
                new_tuples.append((datetime.date(), conid))
            reformatted_data.index = pd.MultiIndex.from_tuples(new_tuples, names=["Date", "ConId"])

        return reformatted_data

    def __repr__(self):
        return "History(start={}, end={}, frequency={})".format(self.start_date.date(),
                                                                self.end_date.date(),
                                                                self.default_frequency.name)


class StrategyMode(Enum):
    """Modes that a MoonLineStrategy can operate in.

    TODO: Description
    """
    DAILY = 1
    INTRADAY = 2


class MoonLineStrategy(ABC):

    CALENDAR = "NYSE"

    def __init__(self, mode, history, pipeline):
        self.mode = mode
        self.history = history
        self.pipeline = pipeline
        self.scheduled_methods = []
        self.current_datetime = None
        self.orders = defaultdict(dict)
        self.timezone = tc.get_calendar(self.CALENDAR).tz

    def schedule_datetime(self, *args, **kwargs):
        if kwargs.get("calendar"):
            kwargs["calendar"] = self.CALENDAR
        scheduled_method = TimeScheduler.schedule_datetime(*args, **kwargs)
        self.scheduled_methods.append(scheduled_method)

    def schedule_date(self, *args, **kwargs):
        if kwargs.get("calendar"):
            kwargs["calendar"] = self.CALENDAR
        scheduled_method = TimeScheduler.schedule_date(*args, **kwargs)
        self.scheduled_methods.append(scheduled_method)

    def schedule_time(self, *args, **kwargs):
        if kwargs.get("calendar"):
            kwargs["calendar"] = self.CALENDAR
        scheduled_method = TimeScheduler.schedule_time(*args, **kwargs)
        self.scheduled_methods.append(scheduled_method)

    def schedule_interval(self, *args, **kwargs):
        if kwargs.get("calendar"):
            kwargs["calendar"] = self.CALENDAR
        scheduled_method = TimeScheduler.schedule_interval(*args, **kwargs)
        self.scheduled_methods.append(scheduled_method)

    def order_target_percent(self, asset, percentage):
        if self.mode == StrategyMode.INTRADAY:
            self.orders[asset.conid][(pd.Timestamp(self.current_datetime.date()),
                                      self.current_datetime.time())] = percentage
        elif self.mode == StrategyMode.DAILY:
            self.orders[asset.conid][(pd.Timestamp(self.current_datetime.date()))] = percentage

    def generate_signals(self):
        df = pd.DataFrame(data=self.orders)
        df.columns.name = "ConId"
        if self.mode == StrategyMode.INTRADAY:
            df.index.names = ("Date", "Time")
        elif self.mode == StrategyMode.DAILY:
            df.index.names = ("Date",)
        df.columns = df.columns.map(int)

        now = arrow.utcnow().to(self.timezone)
        if self.mode == StrategyMode.INTRADAY:
            date, time = df.iloc[-1].name
            latest_available_date = arrow.get(pd.Timestamp.combine(date, time),
                                              self.timezone).shift(days=1)
        elif self.mode == StrategyMode.DAILY:
            latest_available_date = arrow.get(df.iloc[-1].name,
                                              self.timezone).shift(days=1)
        diff = now - latest_available_date
        if 1 <= diff.days < 3:
            for new_date in arrow.Arrow.range("day", latest_available_date, now):
                if new_date.weekday() in (5, 6):
                    continue
                df.loc[pd.Timestamp(new_date.format("YYYY-MM-DD HH:mm:ss"))] = None
        df = df.ffill()

        return df

    def run_handle_data(self, datetime, data):
        self.current_datetime = datetime
        self.handle_data(data)

    @abstractmethod
    def handle_data(self, data):
        """
        Called with every iteration, at the smallest resolution.
        Strategies can ignore this callback by implementing an empty override.
        """
        pass


class CapeShillerETFsUS(MoonLineStrategy):

    FORCE_TRADE = [arrow.get("2019-07-25")]

    CALENDAR = "NYSE"
    CAPE_SHILLER_STOCK = Asset("SPY", "ARCA")
    RETL = Asset("RETL", "ARCA")
    NUGT = Asset("NUGT", "ARCA")
    TMF = Asset("TMF", "ARCA")
    TYD = Asset("TYD", "ARCA")
    SPXS = Asset("SPXS", "ARCA")
    SPXL = Asset("SPXL", "ARCA")
    TECL = Asset("TECL", "ARCA")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_price = 0

        self.schedule_interval(self.rebalance, "D", 1)
        self.schedule_interval(self.cape_check, "M", 1)
        self.schedule_interval(self.regime_check, "M", 1)

        self.last_regime = 3
        self.regime_data = defaultdict(list)
        self.cape_history = []
        if is_quantrocket:
            self.cape_shiller_data = pd.read_csv("/" + os.path.join("codeload",
                                                                    "satellite_scripts",
                                                                    "script_data",
                                                                    "cape_shiller_data.csv"))
        else:
            self.cape_shiller_data = pd.read_csv("data/cape_shiller_data.csv")
        self.cape_shiller_data = self.cape_shiller_data.set_index("Date")
        self.cape_shiller_data.index = pd.to_datetime(self.cape_shiller_data.index, format="%Y-%m-%dT%H:%M:%S.%f")

        now = arrow.utcnow().to(self.timezone)
        latest_available_date = arrow.get(self.cape_shiller_data.iloc[-1].name, self.timezone).shift(days=1)
        if now.month > latest_available_date.month:
            for new_date in arrow.Arrow.range("month", latest_available_date, now):
                if new_date.month <= latest_available_date.month:
                    continue
                self.cape_shiller_data.loc[pd.Timestamp(new_date.replace(day=1).format("YYYY-MM-DD HH:mm:ss"))] = None
        self.cape_shiller_data = self.cape_shiller_data.ffill()

    def cape_check(self, data):
        latest = self.cape_shiller_data.loc[self.current_datetime.date().replace(day=1)].values[0]
        self.cape_history.append(latest)

    def regime_check(self, data, force=True):
        change = 0
        if len(self.cape_history) > 1:
            change = (self.cape_history[-1] - self.cape_history[-2]) / self.cape_history[-2]

        try:
            cur_price = data.loc[self.CAPE_SHILLER_STOCK]["Open"]
            self.last_price = cur_price
        except KeyError:
            cur_price = self.last_price
        prices = self.history.history(101).xs(self.CAPE_SHILLER_STOCK, level="ConId")["Open"]

        if len(prices) < 26:
            return 3

        avg = prices[-20:].mean()
        std = prices[-20:].std()
        lower_band = avg - 2 * std
        upper_band = avg + 2 * std

        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()

        macd_buy_signal = macd[-4:-2].mean() < signal[-4:-2].mean() and macd[-1] >= signal[-1]
        macd_sell_signal = macd[-4:-2].mean() > signal[-4:-2].mean() and macd[-1] < signal[-1]

        check1 = cur_price <= lower_band
        check2 = cur_price >= upper_band
        check3 = (macd_buy_signal or macd_sell_signal) and (check1 or check2)

        regime = 3

        if check3 and not force:
            return self.last_regime

        if change >= 0.015:
            if not check2:
                regime = 4
            else:
                regime = 2
        else:
            if change > 0.0075:
                if check2:
                    regime = 2
                else:
                    regime = 3
            else:
                if change > -0.0075:
                    if check1 or macd_buy_signal:
                        regime = 5
                    elif check2 or macd_sell_signal:
                        regime = 1
                    else:
                        regime = 2
                else:
                    if check2 or macd_sell_signal:
                        regime = 0
                    elif check1 or macd_buy_signal:
                        regime = 2
                    else:
                        regime = 1

        self.last_regime = regime

        return regime

    def rebalance(self, data):
        self.regime_check(data, force=False)
        regime = self.last_regime
        self.regime_data["date"].append(self.current_datetime.datetime)
        self.regime_data["regime"].append(regime)

        if regime == 4:
            etfs = {
                self.RETL: 0,
                self.NUGT: 0,
                self.TMF: 0,
                self.TYD: 0,
                self.SPXS: 0,
                self.SPXL: 0.5,
                self.TECL: 0.5,
            }
        elif regime == 3:
            etfs = {
                self.RETL: 0,
                self.NUGT: 0,
                self.TMF: 0,
                self.TYD: 0,
                self.SPXS: 0,
                self.SPXL: 1.0,
                self.TECL: 0,
            }
        elif regime == 2:
            etfs = {
                self.RETL: 0,
                self.NUGT: 0,
                self.TMF: 0.3,
                self.TYD: 0.2,
                self.SPXS: 0,
                self.SPXL: 0.5,
                self.TECL: 0,
            }
        elif regime == 5:
            etfs = {
                self.RETL: 0,
                self.NUGT: 0.4,
                self.TMF: 0,
                self.TYD: 0,
                self.SPXS: 0,
                self.SPXL: 0.6,
                self.TECL: 0,
            }
        elif regime == 1:
            etfs = {
                self.RETL: 0,
                self.NUGT: 0,
                self.TMF: 0.6,
                self.TYD: 0.4,
                self.SPXS: 0,
                self.SPXL: 0,
                self.TECL: 0,
            }
        elif regime == 0:
            etfs = {
                self.RETL: 0,
                self.NUGT: 0,
                self.TMF: 0,
                self.TYD: 0,
                self.SPXS: 1.0,
                self.SPXL: 0,
                self.TECL: 0,
            }

        for asset, weight in etfs.items():
            self.order_target_percent(asset, weight)

    def handle_data(self, data):
        pass


class EuropeStockCommission(PercentageCommission):
    IB_COMMISSION_RATE = 0.0005
    MIN_COMMISSION = 5.0


class MoonLineContainer(Moonshot):

    COMMISSION_CLASS = EuropeStockCommission

    CODE = "moonline-strategy-eu"
    DB = "cape-shiller-etfs-eu-v4-eod"
    CALENDAR = "LSE"
    TIMEZONE = "Europe/London"

    def intraday_strategy(self, shifted_prices):
        with timeit("Preparing history data for quick resampling"):
            new_tuples = []
            history_data = shifted_prices.copy()
            index_values = history_data.index.tolist()
            for date, time, conid in tqdm(index_values, total=len(index_values), unit="rows"):
                new_tuples.append((pd.to_datetime("{} {}".format(date, time), format="%Y-%m-%d %H:%M:%S"), conid))
            history_data.index = pd.MultiIndex.from_tuples(new_tuples, names=["Datetime", "ConId"])
            history_data.index = pd.MultiIndex.from_tuples([(x[0], Asset(int(x[1]))) for x in history_data.index])

        # Figure out bar size
        dates = shifted_prices.index.levels[0]
        times = shifted_prices.index.levels[1]

        datetime_start = arrow.get("{} {}".format(dates[0], times[0]), "YYYY-MM-DD HH:mm:ss")
        datetime_stop = arrow.get("{} {}".format(dates[0], times[1]), "YYYY-MM-DD HH:mm:ss")
        combined = datetime_stop - datetime_start

        frequency = HistoryFrequency(combined.total_seconds())
        print("Bar size is {}".format(frequency.name))

        # Prepare DataFrame and required objects
        with timeit("Initializing strategy and history objects"):
            shifted_prices.index = pd.MultiIndex.from_tuples([(x[0], x[1], Asset(int(x[2])))
                                                              for x in shifted_prices.index])
            shifted_prices.index.names = ["Date", "Time", "ConId"]
            history = History(history_data, frequency)
            pipeline = Pipeline()
            strategy = CapeShillerETFsUS(mode=StrategyMode.INTRADAY, history=history, pipeline=pipeline)

        # Main Loop
        with timeit("Running backtest"):
            days_grouped = shifted_prices.groupby(level="Date")
            for day_index, day_data in tqdm(days_grouped, total=len(days_grouped), unit="days"):
                hours_grouped = day_data.loc[day_index].groupby(level="Time")
                for time_index, time_data in tqdm(hours_grouped, total=len(hours_grouped), unit="hours", leave=False):
                    datetime = arrow.get("{} {}".format(day_index, time_index)).replace(tzinfo=strategy.timezone)
                    date, time = str(datetime.date()), str(datetime.time())
                    data = time_data.loc[time_index]
                    # Update the History object
                    history.update_time(datetime)
                    # Update the Pipeline object
                    pipeline.update_time(datetime)
                    # Run the main data handling function
                    strategy.run_handle_data(datetime, data=data)
                    # Run all scheduled functions
                    for scheduled_method in strategy.scheduled_methods:
                        if scheduled_method.should_run(date, time):
                            scheduled_method.run(data=data)

        return strategy

    def daily_strategy(self, shifted_prices):
        with timeit("Preparing history data for quick resampling"):
            new_tuples = []
            history_data = shifted_prices.copy()
            index_values = history_data.index.tolist()
            for date, conid in tqdm(index_values, total=len(index_values), unit="rows"):
                new_tuples.append((pd.to_datetime(date, format="%Y-%m-%d"), conid))
            history_data.index = pd.MultiIndex.from_tuples(new_tuples, names=["Datetime", "ConId"])
            history_data.index = pd.MultiIndex.from_tuples([(x[0], Asset(x[1], ignore_exchange=True)) for x in history_data.index])

        # Figure out bar size
        dates = shifted_prices.index.levels[0]

        datetime_start = arrow.get(dates[-3])
        datetime_stop = arrow.get(dates[-2])
        combined = datetime_stop - datetime_start

        try:
            frequency = HistoryFrequency(combined.total_seconds())
        except:
            datetime_start = arrow.get(dates[-6])
            datetime_stop = arrow.get(dates[-5])
            combined = datetime_stop - datetime_start

            frequency = HistoryFrequency(combined.total_seconds())

        print("Bar size is {}".format(frequency.name))

        # Prepare DataFrame and required objects
        with timeit("Initializing strategy and history objects"):
            shifted_prices.index = pd.MultiIndex.from_tuples([(x[0], Asset(x[1], ignore_exchange=True)) for x in shifted_prices.index])
            shifted_prices.index.names = ["Date", "ConId"]
            history = History(history_data, frequency)
            pipeline = Pipeline()
            strategy = CapeShillerETFsUS(mode=StrategyMode.DAILY, history=history, pipeline=pipeline)

        # Main Loop
        with timeit("Running backtest"):
            grouped = shifted_prices.groupby(level="Date")
            for day_index, day_data in tqdm(grouped, total=len(grouped), unit="days"):
                datetime = arrow.get(day_index)
                date = str(datetime.date())
                data = day_data.loc[day_index]
                # Update the History object
                history.update_time(datetime)
                # Update the Pipeline object
                pipeline.update_time(datetime)
                # Run the main data handling function
                strategy.run_handle_data(datetime, data=data)
                # Run all scheduled functions
                for scheduled_method in strategy.scheduled_methods:
                    if scheduled_method.should_run(date):
                        scheduled_method.run(data=data)

        return strategy

    def prices_to_signals(self, prices):
        filled = prices.bfill().ffill()
        with timeit("Preparing price dataframe"):
            all_tickers = list(prices.columns)
            all_fields = sorted(list(filled.index.levels[0]))

            proto_df = defaultdict(lambda: defaultdict(int))

            if "Time" in prices.index.names:
                for ticker in all_tickers:
                    for field in all_fields:
                        for timestamp, value in filled[ticker][field].iteritems():
                            date, time = timestamp
                            proto_df[field][(date, time, ticker)] = value

                shifted_prices = pd.DataFrame(proto_df).sort_index()
                shifted_prices.columns.name = "Field"
                shifted_prices.index.names = ("Date", "Time", "ConId")
            else:
                for ticker in all_tickers:
                    for field in all_fields:
                        for timestamp, value in filled[ticker][field].iteritems():
                            proto_df[field][(timestamp, ticker)] = value

                shifted_prices = pd.DataFrame(proto_df).sort_index()
                shifted_prices.columns.name = "Field"
                shifted_prices.index.names = ("Date", "ConId")

        if "Time" in prices.index.names:
            strategy = self.intraday_strategy(shifted_prices)
        else:
            strategy = self.daily_strategy(shifted_prices)

        # Once the strategy has run, we can generate the dataframe containing the signals Moonshot expects
        signals = strategy.generate_signals()

        return signals

    def signals_to_target_weights(self, signals, prices):
        return signals

    def target_weights_to_positions(self, weights, prices):
        positions = weights.copy()
        return positions

    def positions_to_gross_returns(self, positions, prices):
        if "Time" in prices.index.names:
            closes = prices.loc["Close"].bfill().ffill().xs("15:45:00", level="Time")
        else:
            closes = prices.loc["Close"].bfill().ffill()
        closes.columns = pd.to_numeric(closes.columns)
        positions.columns = pd.to_numeric(positions.columns)
        gross_returns = closes.pct_change() * positions.shift()
        gross_returns.index.name = "Date"
        return gross_returns


class UniverseDefinition(Enum):
    QTradeableStocksUS = 1
    Q500US = 2
    Q1500US = 3
    Q3000US = 4


class Pipeline():

    def __init__(self):
        self.client = pms.Client(args.database)
        self.last_result = {}
        self.last_time = None
        self.last_month = None
        self.current_datetime = None
        self.current_datetime_formatted = None

        self.cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "moonline-cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache = defaultdict(dict)
        for file in glob(os.path.join(self.cache_dir, "*.pkl")):
            ext_stripped = os.path.splitext(os.path.basename(file))[0]
            cache_file_time, category = os.path.splitext(ext_stripped)
            self.cache[cache_file_time][category.lstrip(".")] = file

    def update_time(self, datetime):
        self.current_datetime = datetime
        self.current_datetime_formatted = datetime.format("YYYY-MM-DD HH-mm-ss")

    def calculate_average_dollar_volume(self):
        if self.current_datetime_formatted in self.cache and self.cache[self.current_datetime_formatted].get("adv"):
            with open(self.cache[self.current_datetime_formatted]["adv"], "rb") as f:
                adv_list = pickle.load(f)
            return adv_list

        adv_list = {}
        available_symbols = self.client.list_symbols()
        if not available_symbols:
            return {}

        result = self.client.query(pms.Params(available_symbols, "1D", "OHLCV",
                                              end=self.current_datetime_formatted,
                                              limit=201)).all()
        for symbol, data in result.items():
            symbol = symbol.split("/")[0]
            if len(data.array) < 200:
                continue
            current_dollar_volume = np.mean(data.array["Volume"][:-1])
            adv_list[symbol] = current_dollar_volume

        with open(os.path.join(self.cache_dir, "{}.adv.pkl".format(self.current_datetime_formatted)), "wb") as f:
            pickle.dump(adv_list, f, protocol=pickle.HIGHEST_PROTOCOL)

        return adv_list

    def calculate_market_capitalization(self, adv_list):
        if not adv_list:
            return {}

        if self.current_datetime_formatted in self.cache and self.cache[self.current_datetime_formatted].get("mcp"):
            with open(self.cache[self.current_datetime_formatted]["mcp"], "rb") as f:
                marketcap_list = pickle.load(f)
            return marketcap_list

        marketcap_list = {}
        adv_symbols = set(adv_list.keys())
        result = self.client.query(pms.Params(adv_symbols, "1D", "FD",
                                              end=self.current_datetime_formatted,
                                              limit=1)).all()
        for symbol, data in result.items():
            symbol = symbol.split("/")[0]
            if sum(data.array["Marketcap"] > 0) > 0:
                marketcap_list[symbol] = adv_list[symbol]

        with open(os.path.join(self.cache_dir, "{}.mcp.pkl".format(self.current_datetime_formatted)), "wb") as f:
            pickle.dump(marketcap_list, f, protocol=pickle.HIGHEST_PROTOCOL)

        return marketcap_list

    def get_universe(self, universe_definition: UniverseDefinition):
        if (self.current_datetime.year, self.current_datetime.month) != self.last_time:
            self.last_time = (self.current_datetime.year, self.current_datetime.month)
            adv_list = self.calculate_average_dollar_volume()
            selected_stocks = self.calculate_market_capitalization(adv_list)
            full_sorted = [symbol for symbol, adv in sorted(selected_stocks.items(), key=lambda x: x[1], reverse=True)]
            self.last_result[UniverseDefinition.Q500US] = full_sorted[:500]
            self.last_result[UniverseDefinition.Q1500US] = full_sorted[:1500]
            self.last_result[UniverseDefinition.Q3000US] = full_sorted[:3000]
            self.last_result[UniverseDefinition.QTradeableStocksUS] = full_sorted
        # TODO: return OHLCV values instead of just the tickers
        return self.last_result[universe_definition]


def main(args):
    if args.pystore_dir:
        from data_providers import PyStoreDataProvider
        data_provider = PyStoreDataProvider(args.pystore_dir)
        prices = data_provider.get_ohlcv([asset.symbol for asset in Asset])
    else:
        with timeit("Loading price data from disk"):
            prices = pd.read_csv(args.input_file)
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

    try:
        try:
            arrow.get(prices.index.levels[1][0], "YYYY-MM-DD")
        except:
            arrow.get(prices.index.levels[1][0].strftime("%Y-%m-%d"), "YYYY-MM-DD")
    except arrow.parser.ParserMatchError:
        print("The input data is in an unsupported format.\nPlease double-check your input files.")
        sys.exit(1)

    with timeit("Loading securities master"):
        securities_master = None
        if not args.clear_cache and os.path.exists("securities_master.bin"):
            with open("securities_master.bin", "rb") as f:
                securities_master = pickle.load(f)
            if set(map(str, securities_master.index)) != set(map(str, prices.columns)):
                securities_master = None

        if securities_master is None:
            securities_master = pd.read_csv(args.listings_file).set_index("ConId").sort_index()
            # Remove all unused entries for performance improvements
            securities_master = securities_master[securities_master.index.isin(set(prices.columns))]
            with open("securities_master.bin", "wb") as f:
                pickle.dump(securities_master, f, protocol=pickle.HIGHEST_PROTOCOL)

    cs = MoonLineContainer()
    # This is normally assigned by QuantRocket
    cs._securities_master = securities_master

    allocation = 1.0
    label_conids = False

    signals = cs.prices_to_signals(prices)
    weights = cs.signals_to_target_weights(signals, prices)
    weights = weights * allocation
    weights = cs._constrain_weights(weights, prices)
    positions = cs.target_weights_to_positions(weights, prices)
    gross_returns = cs.positions_to_gross_returns(positions, prices)
    commissions = cs._get_commissions(positions, prices)
    slippages = cs._get_slippage(positions, prices)
    returns = gross_returns.fillna(0) - commissions - slippages
    turnover = cs._positions_to_turnover(positions)

    total_holdings = (positions.fillna(0) != 0).astype(int)

    results_are_intraday = "Time" in signals.index.names

    all_results = dict(Signal=signals, Weight=weights, AbsWeight=weights.abs(),
                       AbsExposure=positions.abs(), NetExposure=positions,
                       Turnover=turnover, TotalHoldings=total_holdings,
                       Commission=commissions, Slippage=slippages, Return=returns)

    # validate that custom backtest results are daily if results are daily
    for custom_name, custom_df in cs._backtest_results.items():
        if "Time" in custom_df.index.names and not results_are_intraday:
            raise Exception("custom DataFrame '{0}' won't concat properly with 'Time' in index, "
                            "please take a cross-section first, for example: "
                            "`my_dataframe.xs('15:45:00', level='Time')`".format(custom_name))

    all_results.update(cs._backtest_results)

    if cs.BENCHMARK:
        all_results["Benchmark"] = cs._get_benchmark(prices, daily=not results_are_intraday)

    results = pd.concat(all_results)

    names = ["Field", "Date"]
    if results.index.nlevels == 3:
        names.append("Time")

    results.index.set_names(names, inplace=True)

    if label_conids:
        symbols = cs._securities_master.Symbol
        sec_types = cs._securities_master.SecType
        currencies = cs._securities_master.Currency
        symbols = symbols.astype(str).str.cat(currencies, sep=".").where(sec_types == "CASH", symbols)
        symbols_with_conids = symbols.astype(str) + "(" + symbols.index.astype(str) + ")"
        results.rename(columns=symbols_with_conids.to_dict(), inplace=True)

    # truncate at requested start_date
    if args.start_date:
        results = results.iloc[results.index.get_level_values(
            "Date") >= pd.Timestamp(args.start_date.format("YYYY-MM-DD"))]

    results.to_csv(args.output_file)

    if args.weights_file:
        weights.to_csv(args.weights_file)

    with timeit("Rendering Tearsheets"):
        pdf_file = os.path.splitext(os.path.basename(args.output_file))[0]
        Tearsheet.from_moonshot_csv(args.output_file, pdf_filename="{}.pdf".format(pdf_file))


def check(args):
    if args.input_file and not os.path.isfile(args.input_file):
        print("The input file does not exist!")
        sys.exit(1)

    if os.path.isfile(args.output_file):
        if not args.yes:
            print("The output file exists. Do you want to overwrite it?")
            result = input("[y]es/[n]o: ").lower()
            if not result in ["y", "yes"]:
                print("Aborted")
                sys.exit(0)
        os.remove(args.output_file)

    dirs = os.path.dirname(args.output_file)
    if dirs:
        os.makedirs(dirs, exist_ok=True)

    if args.weights_file and os.path.isfile(args.weights_file):
        if not args.yes:
            print("The weights file exists. Do you want to overwrite it?")
            result = input("[y]es/[n]o: ").lower()
            if not result in ["y", "yes"]:
                print("Aborted")
                sys.exit(0)
        os.remove(args.weights_file)

        dirs = os.path.dirname(args.weights_file)
        if dirs:
            os.makedirs(dirs, exist_ok=True)

    if args.start_date and args.end_date and args.start_date > args.end_date:
        print("End date must be larger than start date!")
        sys.exit(1)


if __name__ == '__main__':
    import sys
    import argparse

    ### Charting Libraries ###
    from moonchart import Tearsheet

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, dest="input_file",
                        metavar="prices.csv", help="A CSV file containing price data to backtest on")
    parser.add_argument("-l", "--listings", type=str, dest="listings_file", required=True,
                        metavar="listings.csv", help="The file containing InteractiveBrokers listings data")
    parser.add_argument("-s", "--start-date", type=str, dest="start_date",
                        metavar="YYYY-MM-DD", help="The day to start the backtest from")
    parser.add_argument("-e", "--end-date", type=str, dest="end_date",
                        metavar="YYYY-MM-DD", help="The day to end the backtest at")
    parser.add_argument("-o", "--output", type=str, dest="output_file", default="results.csv",
                        metavar="results.csv", help="The file to output backtest results to")
    parser.add_argument("-w", "--weights", type=str, dest="weights_file",
                        metavar="weights.csv", help="The file to save calculated weights to")
    parser.add_argument("-p", "--pystore", type=str, dest="pystore_dir",
                        metavar="dir", help="The directory containing PyStore data")
    parser.add_argument("-y", "--yes", action="store_true", dest="yes",
                        help="If given, automatically answers script questions with 'yes'")
    parser.add_argument("-c", "--clear-cache", action="store_true", dest="clear_cache",
                        help="If given, ignores cached data")
    args = parser.parse_args()

    if args.start_date:
        args.start_date = arrow.get(args.start_date, "YYYY-MM-DD")
    if args.end_date:
        args.end_date = arrow.get(args.end_date, "YYYY-MM-DD")

    check(args)

    with timeit("Loading Listings"):
        assets_init(args.listings_file)

    try:
        main(args)
        print("Done")
    except KeyboardInterrupt:
        print("Aborted")
        sys.exit()
