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
    
    open_signal = pd.Series(data=0, index=ls_x)
    close_signal = pd.Series(data=0, index=ls_x)
    position = pd.Series(data=0, index=ls_x)
    pos_chg = pd.Series(data=0, index=ls_x)
    position[start_ix] = pos_chg[start_ix] = signal[start_ix]
   
    for t in range(start_ix+HP, num_x):
        if (signal[t-HP] == -1 or 1) and all(signal[t-HP+1:t+1]==0):
            close_signal[t] = -signal[t-HP]
        elif -signal[t-HP] in signal[t-HP+1:t+1]:
            close_signal[t] = -signal[t-HP]

    pos_chg = signal + close_signal
    position = pos_chg.cumsum()
    
    for t in range(start_ix, num_x):
        if np.abs(position[t]) >= np.abs(position[t-1]):
            if np.abs(pos_chg[t]) > signal[t] == 0:
                print(t)
                pos_chg[t] = 0
    position = pos_chg.cumsum()
    
#    TODO:
    ##########################
#    for t in range(start_ix, num_x):
#        if np.abs(position[t]) > np.abs(position[t-1]):
#            open_signal[t] = signal[t]
#            close_signal[t] = 0
#        elif np.abs(position[t]) < np.abs(position[t-1]):
#            open_signal[t] = 0
#            close_signal[t] = pos_chg[t]
#    
#    pos_chg_1 = open_signal + close_signal
#    position_1 = pos_chg_1.cumsum()
    ##########################
    x = pd.DataFrame(columns=['signal','open'])
    x['signal'] = signal
    x['open'] = open_signal
    x['close'] = close_signal
    x['pos_chg'] = pos_chg
#    x['position_0'] = position_0
#    x['pos_chg_1'] = pos_chg_1
#    x['position_1'] = position_1
    x['ps'] = position
    x = x.reset_index()
    

    signal[:20] = [0, 0, 1, -1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, -1, 0, 0, 0, 0]

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

