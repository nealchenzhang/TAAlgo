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
    SMA = ys.rolling(window=w).apply(np.mean)
    return SMA


def LWMA(ys, w=5):
    """
    :param ys: column vector of price series with str time index
    :param w: lag number
    :return Lwma: weighted MA price Series
    """
    LWMA = ys.rolling(window=w).apply(lambda x: np.average(x, weights=np.arange(w, 0, -1)))
    return LWMA


def EWMA(ys, w=5, exponential=2/(5+1)):
    EWMA = pd.Series(0.0, index=ys.index.tolist())
    EWMA[w - 1] = SMA(ys, w)[w-1]
    EWMA[:w-1] = np.nan
    for i in range(w, len(ys)):
        EWMA[i] = exponential * ys[i] + (1 - exponential) * EWMA[i - 1]
    return EWMA


def MA_signal(ys, w, method='SMA', **kwargs):
    """

    :param ys:
    :param method:
            'SMA': simple moving average
            'LWMA': linearly weighted moving average
            'EWMA': exponential moving average
    :return signal: 1, -1 or 0:
    """
    if method == 'SMA':
        MA = SMA(ys, w)
    if method == 'LWMA':
        MA = LWMA(ys, w)
    if method == 'EWMA':
        MA = EWMA(ys, w, exponential=kwargs['exponential'])

    if ys[-1] > MA[-1] and ys[-2] < MA[-2]:
        signal = 1
    elif ys[-1] < MA[-1] and ys[-2] > MA[-2]:
        signal = -1
    else:
        signal = 0

    return signal

###############################################################################
# Moving Averages Crossovers
###############################################################################


def MAC(ys, ws, wl, method='SMA'):
    if method == 'SMA':
        MAs = SMA(ys, ws)
        MAl = SMA(ys, wl)

    MAC = MAs - MAl

    if MAC[-1] > 0 and MAC[-2] < 0:
        signal = 1
    elif MAC[-1] < 0 and MAC[-2] > 0:
        signal = -1
    else:
        signal = 0

    return MAC, signal

###############################################################################
# Moving Averages Convergence Divergence
###############################################################################


def MACD(ys, ws=12, wl=26, wsignal=9, exponential=0.2):
    MACD = EWMA(ys, ws, exponential) - EWMA(ys, wl, exponential)
    SL = EWMA(MACD, wsignal, exponential)

    if MACD[-1] > SL[-1] and MACD[-2] < SL[-2]:
        signal = 1
    elif MACD[-1] < SL[-1] and MACD[-2] > SL[-2]:
        signal = -1
    else:
        signal = 0

    return MACD, SL, signal


if __name__ == '__main__':
    df_ys = pd.read_csv('./Data/ru_i_15min.csv')
    df_ys.datetime = df_ys.datetime.apply(pd.to_datetime)
    df_ys.datetime = df_ys.datetime.apply(lambda x: str(x))
    df_ys.set_index('datetime', inplace=True)
    ls_cols = df_ys.columns.tolist()
    str_Close = [i for i in ls_cols if i[-6:] == '.close'][0]
    ys = df_ys.loc[:, str_Close]

    MACD, SL, signal = MACD(ys)


