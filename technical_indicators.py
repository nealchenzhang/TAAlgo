# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 13:49:23 2019

@author: chen zhang
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style('white')

###############################################################################
# Moving Averages
###############################################################################


def SMA(ys, w=5):
    """

    :param ys: column vector of price series with str time index
    :param w: lag number
    :return
    """
    MA = ys.rolling(window=w).apply(np.mean)

    if ys[-1] > MA[-1] and ys[-2] < MA[-2]:
        signal = 1
    elif ys[-1] < MA[-1] and ys[-2] > MA[-2]:
        signal = -1
    else:
        signal = 0

    dict_results = {
        'SMA': MA,
        'signal': signal
    }
    return dict_results


def LWMA(ys, w=5):
    """

    :param ys: column vector of price series with str time index
    :param w: lag number
    :return
    """
    MA = ys.rolling(window=w).apply(lambda x: np.average(x, weights=np.arange(w, 0, -1)))

    if ys[-1] > MA[-1] and ys[-2] < MA[-2]:
        signal = 1
    elif ys[-1] < MA[-1] and ys[-2] > MA[-2]:
        signal = -1
    else:
        signal = 0

    dict_results = {
        'LWMA': MA,
        'signal': signal
    }
    return dict_results


def EWMA(ys, w=5):
    exponential = 2 / (w + 1)
    MA = pd.Series(0.0, index=ys.index.tolist())
    MA[w - 1] = SMA(ys, w)['SMA'][w-1]
    MA[:w-1] = np.nan
    for i in range(w, len(ys)):
        MA[i] = exponential * ys[i] + (1 - exponential) * MA[i - 1]

    if ys[-1] > MA[-1] and ys[-2] < MA[-2]:
        signal = 1
    elif ys[-1] < MA[-1] and ys[-2] > MA[-2]:
        signal = -1
    else:
        signal = 0

    dict_results = {
        'EWMA': MA,
        'signal': signal
    }

    return dict_results


###############################################################################
# Moving Averages Crossovers
###############################################################################


def MAC(ys, ws, wl, method='SMA'):
    if method == 'SMA':
        MAs = SMA(ys, ws)['SMA']
        MAl = SMA(ys, wl)['SMA']

    MAC = MAs - MAl

    if MAC[-1] > 0 and MAC[-2] < 0:
        signal = 1
    elif MAC[-1] < 0 and MAC[-2] > 0:
        signal = -1
    else:
        signal = 0

    dict_results = {
        'MAC': MAC,
        'signal': signal
    }

    return dict_results

###############################################################################
# Moving Averages Convergence Divergence
###############################################################################


def MACD(ys, ws=12, wl=26, wsignal=9):
    MACD = EWMA(ys, ws)['EWMA'] - EWMA(ys, wl)['EWMA']
    SL = EWMA(MACD, wsignal)['EWMA']

    if MACD[-1] > SL[-1] and MACD[-2] < SL[-2]:
        signal = 1
    elif MACD[-1] < SL[-1] and MACD[-2] > SL[-2]:
        signal = -1
    else:
        signal = 0

    dict_results = {
        'MACD': MACD,
        'SL': SL,
        'signal': signal
    }

    return dict_results


if __name__ == '__main__':
    df_ys = pd.read_csv('./Data/ru_i_15min.csv')
    df_ys.datetime = df_ys.datetime.apply(pd.to_datetime)
    df_ys.datetime = df_ys.datetime.apply(lambda x: str(x))
    df_ys.set_index('datetime', inplace=True)
    ls_cols = df_ys.columns.tolist()
    str_Close = [i for i in ls_cols if i[-6:] == '.close'][0]
    ys = df_ys.loc[:, str_Close]



