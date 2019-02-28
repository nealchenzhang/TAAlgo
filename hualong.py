# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 16:13:07 2019

@author: chen zhang
"""

import datetime as dt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#from matplotlib.dates import DateFormatter, WeekdayLocator, DayLocator, MONDAY,YEARLY
#from mpl_finance import candlestick_ohlc
#from matplotlib.pylab import date2num

import seaborn as sns
sns.set_style('white')

from preprocessing import RW
from preprocessing import TP
from preprocessing import PIPs
from preprocessing import PB_plotting

from technical_indicators import SMA


def wave_plotting(ys, Peaks, Bottoms, **kwargs):
    ls_x = ys.index.tolist()
    num_x = len(ls_x)
    ls_time_ix = np.linspace(0, num_x - 1, num_x)
    ls_p = Peaks.index.tolist()
    ls_b = Bottoms.index.tolist()
    ls_peaks_time = [ys.index.get_loc(x) for x in ls_p]
    ls_bottoms_time = [ys.index.get_loc(x) for x in ls_b]

    ds_MA = kwargs['MA']

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.plot(ls_time_ix, ys.values)
    ax.plot(ls_time_ix, ds_MA.values)
    ax.scatter(x=ls_peaks_time, y=Peaks.values, marker='o', color='r', alpha=0.5)
    ax.scatter(x=ls_bottoms_time, y=Bottoms.values, marker='o', color='g', alpha=0.5)

    # for i in ls_peaks_time:
    #     ax.text(x=i, y=Peaks.loc[ls_x[i]],
    #             s=ls_x[i], withdash=True,
    #             )
    new_xticklabels = [ls_x[np.int(i)] for i in list(ax.get_xticks()) if i in ls_time_ix]
    new_xticklabels = [ls_x[0]] + new_xticklabels
    new_xticklabels.append(ls_x[-1])
    ax.set_xticklabels(new_xticklabels)
    for tick in ax.get_xticklabels():
        tick.set_rotation(15)
    #        plt.savefig('1.jpg')
    plt.show()

def wave_check(ys, pflag, method='RW', **kwargs):
    l = len(ys)
    if method == 'RW':
        Peaks, Bottoms = RW(ys, w=kwargs['w'], iteration=kwargs['iteration'])
        Peaks, Bottoms = RW(ys, w=1, iteration=2)
    elif method == 'TP':
        Peaks, Bottoms = TP(ys, iteration=kwargs['iteration'])

    MA = SMA(ys, w=5)['SMA']
    wave_plotting(ys, Peaks, Bottoms, MA=MA)

    ls_x = ys.index.tolist()
    ls_p = Peaks.index.tolist()
    ls_b = Bottoms.index.tolist()
    P_idx = [ys.index.get_loc(x) for x in ls_p]
    B_idx = [ys.index.get_loc(x) for x in ls_b]

    P_idx = pd.Series(index=ls_p, data=[1]*len(ls_p))
    B_idx = pd.Series(index=ls_b, data=[2]*len(ls_b))
    PB_idx = P_idx.append(B_idx)
    PB_idx.sort_index(inplace=True)
    m = len(PB_idx.index)

    Pot_Wave_up1 = [2, 1]
    Pot_Wave_down1 = [1, 2]
    Pot_Index = [0] * m

    for i in range(m-1, 0, -1):
        print(i)
        if PB_idx.iloc[i-1: i+1].values.tolist() == Pot_Wave_up1:
            Pot_Index[i-1] = 1
        elif PB_idx.iloc[i-1: i+1].values.tolist() == Pot_Wave_down1:
            Pot_Index[i-1] = 2

    PNidx = [i for i, x in enumerate(Pot_Index) if x == 1]
    PIidx = [i for i, x in enumerate(Pot_Index) if x == 2]

    Wave_up = ys.loc[(PB_idx.iloc[PNidx[-1]:PNidx[-1]+2]).index.tolist()].copy()
    Wave_down = ys.loc[(PB_idx.iloc[PIidx[-1]:PIidx[-1]+2]).index.tolist()].copy()

    dict_results = {
        'Up': Wave_up,
        'Down': Wave_down
    }

    return dict_results

    # PIP method
    # find first wave
    # PIPxy1 = PIPs(ys, n_PIPs=3)
    # ls_PIPx1 = PIPxy1.index.tolist()
    # ls_PIPy1 = PIPxy1.values.tolist()
    #
    # ls_x = ys.index.tolist()
    # num_x = len(ls_x)
    # ls_time_ix = np.linspace(0, num_x-1, num_x)
    # ls_PIPx1_time = [ys.index.get_loc(x) for x in ls_PIPx1]
    #
    # fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    # ax.plot(ls_time_ix, ys.values)
    # ax.scatter(x=ls_PIPx1_time, y=PIPxy1.values, marker='o', color='r', alpha=0.5)
    #
    # new_xticklabels = [ls_x[np.int(i)] for i in list(ax.get_xticks()) if i in ls_time_ix]
    # new_xticklabels = [ls_x[0]] + new_xticklabels
    # new_xticklabels.append(ls_x[-1])
    # ax.set_xticklabels(new_xticklabels)
    # for tick in ax.get_xticklabels():
    #     tick.set_rotation(15)
    # plt.show()

    return signal


if __name__ == '__main__':

    df_ys = pd.read_csv('./Data/IF1903_1min.csv')
    df_ys.datetime = df_ys.datetime.apply(pd.to_datetime)
    df_ys.datetime = df_ys.datetime.apply(lambda x: str(x))
    df_ys.set_index('datetime',inplace=True)
    ls_cols = df_ys.columns.tolist()
    str_Close = [i for i in ls_cols if i[-6:] == '.close'][0]
    ys = df_ys.loc[:, str_Close]

    ys = ys[:40]
    
    
    MA = SMA(ys, w=20)['SMA']
    df_tmp = pd.DataFrame(columns=['Close', 'MA'], index=ys.index.tolist())

    df_tmp.loc[:, 'Close'] = ys
    df_tmp.loc[:, 'MA'] = MA
    
    tmp = df_tmp.dropna()
    ys = tmp[:30]
