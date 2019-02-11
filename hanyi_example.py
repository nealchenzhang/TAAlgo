# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 15:51:13 2019

@author: chenzhang
"""
import os

import datetime as dt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.dates import DateFormatter, WeekdayLocator, DayLocator, MONDAY,YEARLY
from mpl_finance import candlestick_ohlc
from matplotlib.pylab import date2num

import seaborn as sns
sns.set_style('white')

from zigzag_patterns import HS


df_ys = pd.read_csv('./Data/m_i_1d.csv')
df_ys.datetime = df_ys.datetime.apply(pd.to_datetime)
df_ys.datetime = df_ys.datetime.apply(lambda x: str(x)) 
df_ys.set_index('datetime',inplace=True)
ls_cols = df_ys.columns.tolist()
str_Close = [i for i in ls_cols if i[-6:]=='.close'][0]
ys = df_ys.loc[:, str_Close]

HS(ys, pflag=1, method='RW', w=30, savefig=True, figname='test')
