# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 14:17:07 2019

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
df_data.datetime = df_data.datetime.apply(pd.to_datetime)
df_data.set_index('datetime', inplace=True)
df_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']


ds_ys = df_data.Close[:120]

def RW(ds_ys, w, pflag=0):
    """
    ds_ys: column vector of price series
    w: width of the rolling window
    pflag: plot a graph equals 1
    
    returns: 
        Peaks: data Series with datetime index and price
        Bottoms: data Series with datetime index and price
    """
    l = len(ds_ys)
    ls_ix = ds_ys.index.tolist()
    ls_ix_peaks = []
    ls_ix_bottoms = []
    for i in range(w+1, l-w):
        if (ds_ys.iloc[i] > np.max(ds_ys.iloc[i-w-1: i-1])) and \
            (ds_ys.iloc[i] > np.max(ds_ys.iloc[i+1: i+w+1])):
                print(i)
                ls_ix_peaks.append(ls_ix[i])
        if (ds_ys.iloc[i] < np.min(ds_ys.iloc[i-w-1: i-1])) and \
            (ds_ys.iloc[i] < np.min(ds_ys.iloc[i+1: i+w+1])):
                print(i)
                ls_ix_bottoms.append(ls_ix[i])
    ds_peaks = pd.Series(index=ls_ix_peaks, data=ds_ys.loc[ls_ix_peaks])
    ds_bottoms = pd.Series(index=ls_ix_bottoms, data=ds_ys.loc[ls_ix_bottoms])
            
    if pflag == 1:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.plot(ds_ys)
        ax.scatter(x=ds_peaks.index, y=ds_peaks, marker='o', color='r', alpha=0.5)
        ax.scatter(x=ds_bottoms.index, y=ds_bottoms, marker='o', color='g', alpha=0.5)
        plt.show()
        
    return ds_peaks, ds_bottoms

