# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 16:13:07 2019

@author: chen zhang
"""

import datetime as dt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.dates import DateFormatter, WeekdayLocator, DayLocator, MONDAY,YEARLY
from mpl_finance import candlestick_ohlc
from matplotlib.pylab import date2num

import seaborn as sns
sns.set_style('white')

df_data = pd.read_csv('my_data.csv')
# 注意 这里datetime 是 str 不是datetime64
#df_data.datetime = df_data.datetime.apply(pd.to_datetime)
df_data.set_index('datetime', inplace=True)
df_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

ys = df_data.Close[:300]
# 需要对数据源进行处理 行情中断 10：15-10：29 以及不连续的处理
from processing import RW


def HS(ys, w, pflag):
    return Patterns