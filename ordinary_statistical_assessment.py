# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 11:05:23 2019

@author: chen zhang
"""

import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt


from technical_indicators import SMA

strategy = SMA
dict_results = strategy(ys, w=5)


def ordinary_statistical_assessment(dict_results, HP=5):

    signal = dict_results['signal']

    ls_x = signal.index.tolist()
    num_x = len(ls_x)
    ls_time_ix = np.linspace(0, num_x-1, num_x)

    start_ix = 0

    for ix in ls_time_ix:
        if signal[np.int(ix)] == 1 or signal[np.int(ix)] == -1:
            start_ix = np.int(ix)
            break

    buy_signal = signal.where(signal == 1).dropna()
    sell_signal = signal.where(signal == -1).dropna()
    buy_signal_idx = buy_signal.index.tolist()
    sell_signal_idx = sell_signal.index.tolist()

    ls_buy_time = [signal.index.get_loc(x) for x in buy_signal_idx]
    ls_sell_time = [signal.index.get_loc(x) for x in sell_signal_idx]

    close = -signal.shift(HP)

    # position = pd.Series(data=np.nan, index=ls_x)
    # position[start_ix-1] = 0
    # pos_chg = 0
    #
    # for i in range(start_ix, num_x):
    #     if i >= start_ix + HP:
    #         if position[i - 1] == 0:
    #             close[i] = 0
    #     else:
    #         close[i] = 0
    #     pos_chg = signal[i] + close[i]
    #     # print(pos_chg)
    #     position[i] = position[i-1] + pos_chg
    #     # print(position[i-1])
    #
    # Rtn_long = 0
    # Rtn_short = 0
    # Rtn__sell = 0
    #
    # N_long = position.where(position > 0).count()
    # N_short = position.where(position < 0).count()


    return None


if __name__ == '__main__':
    df_ys = pd.read_csv('./Data/ru_i_15min.csv')
    #    df_ys = pd.read_csv('./Data/IF1903_1min.csv')
    df_ys.datetime = df_ys.datetime.apply(pd.to_datetime) + dt.timedelta(minutes=14, seconds=59)
    df_ys.datetime = df_ys.datetime.apply(lambda x: str(x))
    df_ys.set_index('datetime', inplace=True)
    ls_cols = df_ys.columns.tolist()
    str_Close = [i for i in ls_cols if i[-6:] == '.close'][0]
    ys = df_ys.loc[:, str_Close]


    ys = ys[-300:]

