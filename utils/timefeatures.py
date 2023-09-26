from typing import List

import numpy as np
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset


class TimeFeature:
    def __init__(self):
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        pass

    def __repr__(self):
        return self.__class__.__name__ + "()"

## SecondOfMinute，MinuteOfHour，HourOfDay， DayOfWeek都是从0开始索引，所以下方分别/59.0, /59.0, /23.0, /6.0
## DayOfMonth，MonthOfYear, DayOfYear则都是从1开始索引，因此下方分别(  -1)/30.0, (  -1)/11.0, (  -1)/365.0,
## https://github.com/zhouhaoyi/Informer2020/issues/291  归一化到[-0.5, 0.5]是Informer作者们测试后的结果
class SecondOfMinute(TimeFeature):
    """Second of minute encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second / 59.0 - 0.5  #[-0.5, 0.5]
        # return index.second / 59.0  #[0, 1]
        # return 2 * index.second / 59.0 - 1  #[-1, 1]

class MinuteOfHour(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5
        # return index.minute / 59.0  #[0, 1]
        # return 2 * index.minute / 59.0 - 1  #[-1, 1]

class HourOfDay(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0 - 0.5
        # return index.hour / 23.0  #[0, 1]
        # return 2 * index.hour / 23.0 - 1  #[-1, 1]

class DayOfWeek(TimeFeature):
    """Day of week encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5
        # return index.dayofweek / 6.0  #[0, 1]
        # return 2 * index.dayofweek / 6.0 - 1  #[-1, 1]

class DayOfMonth(TimeFeature):
    """Day of month encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5
        # return (index.day - 1) / 30.0  #[0, 1]
        # return 2 * (index.day - 1) / 30.0 - 1  #[-1, 1]

class DayOfYear(TimeFeature):
    """Day of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5
        # return (index.dayofyear - 1) / 365.0  #[0, 1]
        # return 2 * (index.dayofyear - 1) / 365.0 - 1  #[-1, 1]

class MonthOfYear(TimeFeature):
    """Month of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1) / 11.0 - 0.5
        # return (index.month - 1) / 11.0  #[0, 1]
        # return 2 * (index.month - 1) / 11.0 - 1  #[-1, 1]

class WeekOfYear(TimeFeature):
    """Week of year encoded as value between [-0.5, 0.5]"""
    ## 一个类实例要变成一个可调用对象，只需要实现一个特殊方法__call__()。
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.isocalendar().week - 1) / 52.0 - 0.5
        # return (index.isocalendar().week - 1) / 52.0   #[0, 1]
        # return 2 * (index.isocalendar().week - 1) / 52.0 - 1  #[-1, 1]

## -> List[int] 表示该函数应返回一个整数列表。那么下面的 -> List[TimeFeature]应该表示 函数返回一个TimeFeature类型列表
## 注意此处是 大写List，而非 list
def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    """
    Returns a list of time features that will be appropriate for the given frequency string.
    Parameters
    ----------
    freq_str
        Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.
    """

    features_by_offsets = {
        offsets.YearEnd: [],
        offsets.QuarterEnd: [MonthOfYear],
        offsets.MonthEnd: [MonthOfYear],
        offsets.Week: [DayOfMonth, WeekOfYear],
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Minute: [
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
        offsets.Second: [
            SecondOfMinute,
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
    }
    ## pandas.tseries.frequencies中函数to_offset
    offset = to_offset(freq_str)

    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):  ## isinstance()是一个内置函数,用于判断一个对象是否是一个已知的类型,类似type()
            return [cls() for cls in feature_classes]

    supported_freq_msg = f"""
    Unsupported frequency {freq_str}
    The following frequencies are supported:
        Y   - yearly
            alias: A
        M   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
        S   - secondly
    """
    raise RuntimeError(supported_freq_msg)


def time_features(dates, freq='h'):
    return np.vstack([feat(dates) for feat in time_features_from_frequency_str(freq)])
    ## 以freq=day时为例，feat会依次是 类DayofWeek(dates是DatetimeIndex类型数据), DayofMonth(), DayofYear()
    ## index.day (对应类 DayOfMonth) 是 1-31,1-28或29,1-31,1-30,1-31,1-30,1-31,1-31,1-30,1-31,1-30,1-31 并循环
    ## index.dayofweek (对应类 DayofWeek) 是 0,1,2,3,4,5,6并循环
    ## index.dayofyear (对应类 DayofYear) 是 1,2,3,4,5,...365或其他数 并循环