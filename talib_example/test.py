# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 13:16:47 2019

@author: chenzhang
"""

import os

import numpy as np
import pandas as pd
import datetime as dt

import talib

from matplotlib.dates import DateFormatter, WeekdayLocator, DayLocator, MONDAY,YEARLY
from mpl_finance import candlestick2_ochl
import matplotlib.pyplot as plt
from matplotlib.pylab import date2num

df_data = pd.read_csv('my_data.csv')
df_data.datetime = df_data.datetime.apply(pd.to_datetime)

df_data.set_index('datetime', drop=True, inplace=True)

Open = df_data.loc[:, 'SHFE.cu1805.open']
High = df_data.loc[:, 'SHFE.cu1805.high']
Low = df_data.loc[:, 'SHFE.cu1805.low']
Close = df_data.loc[:, 'SHFE.cu1805.close']

MACD, Signalline, MACDhist = talib.MACD(Close, fastperiod=12, slowperiod=26, signalperiod=9)

########################################
integer = talib.CDLDOJI(Open, High, Low, Close)
integer = talib.CDL3WHITESOLDIERS(Open, High, Low, Close)
integer = integer/100*Close
x = integer.where(integer!=0)

fig = plt.figure(figsize=(30, 15))
y=len(Close)
date = np.linspace(0,y,y)
candleAr = []
ax1 = plt.subplot2grid((10,4),(0,0),rowspan=5,colspan=4)
candlestick2_ochl(ax1,Open[:1000],Close[:1000],High[:1000],Low[:1000],width=1,colorup='r',colordown='g', alpha=0.75)
ax1.scatter(date[:1000], x[:1000], marker='.')
plt.savefig('2.png', dpi=300)
#plt.show()

