# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 17:34:07 2019

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

ys = df_data.Close[:3400]
# 需要对数据源进行处理 行情中断 10：15-10：29 以及不连续的处理

# TODO fibonacci

from processing import RW

def HSAR(ys, w, x, pflag=0):
    """
    The HSAR(*) identifies HSARs at time t conditioning information up to time t-1.
    Function RW from processing is embedded.

    :param ys: column vector of price series with time index
    :param w: width of the rolling window (total 2w+1)
    :param x: desired percentage that will give the bounds for the HSARs (e.g., 5%)
    :param pflag: plot a graph if 1
    :return:
        SAR: horizontal support and resistance levels
        Bounds: bounds of bins used to classify the peaks and bottoms
        Freq: frequencies for each bin
        x_act: actual percentage of the bins' distance
    """
    Peaks, Bottoms = RW(ys, w, pflag=0)
    L = Peaks.append(Bottoms)
    L1 = L.min()/ (1+x/2)
    Ln = L.max() * (1+x/2)
    n = np.log(Ln/L1) / np.log(1+x)
    N_act = np.int(np.round(n))
    x_act = (Ln/L1)**(1/N_act)-1
    Bounds = (L1 * (1+x_act))**(pd.Series(np.arange(0, N_act)))
    Freq = np.zeros((N_act, 1))

    return SAR, Bounds, Freq, x_act


