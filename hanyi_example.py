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

from zigzag_patterns import HS
from zigzag_patterns import HS_plotting

import seaborn as sns
sns.set_style('white')

ls_file = os.listdir('./Data')
ls_file = [i for i in ls_file if '.csv' in i and '_' in i]

for file in ls_file:

    figname = file.split('.')[0]
    df_ys = pd.read_csv('./Data/'+ file)
    df_ys.datetime = df_ys.datetime.apply(pd.to_datetime)
    df_ys.datetime = df_ys.datetime.apply(lambda x: str(x))
    df_ys.set_index('datetime', inplace=True)
    ls_cols = df_ys.columns.tolist()
    str_Close = [i for i in ls_cols if i[-6:] == '.close'][0]
    df_ys = df_ys.loc[:, str_Close]

    start = 0
    end = 250
    i = round(len(df_ys) / 125)
    j = len(df_ys) - i * 125
    for num in range(1, i):
        print(num)
        ys = df_ys.iloc[start: end]

        dict_Patterns, Peaks, Bottoms = HS(ys, pflag=0, method='RW', w=1, iteration=0)
        HS_plotting(ys, Peaks, Bottoms, dict_Patterns, figname=figname+str(num), savefig=True)
        start += 125
        end += 125

    ys = df_ys[start:]
    dict_Patterns, Peaks, Bottoms = HS(ys, pflag=0, method='RW', w=1, iteration=0)
    HS_plotting(ys, Peaks, Bottoms, dict_Patterns, figname=figname+str(i+1), savefig=True)

