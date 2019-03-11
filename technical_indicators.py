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
    y = MACD.dropna()

    signalline = EWMA(y, wsignal)['EWMA']
    SL = pd.Series(np.nan, index=ys.index.tolist())
    SL[signalline.index.tolist()] = signalline

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


def MACD_adj(ys, ws=12, wl=26, wsignal=9):
    MACD = EWMA(ys, ws)['EWMA'] - EWMA(ys, wl)['EWMA']
    y = MACD.dropna()

    signalline = EWMA(y, wsignal)['EWMA']
    SL = pd.Series(np.nan, index=ys.index.tolist())
    SL[signalline.index.tolist()] = signalline

    if SL[-1] > 0 and (MACD[-1] > SL[-1] and MACD[-2] < SL[-2]):
        signal = 1
    elif SL[-1] < 0 and (MACD[-1] < SL[-1] and MACD[-2] > SL[-2]):
        signal = -1
    else:
        signal = 0

    dict_results = {
        'MACD': MACD,
        'SL': SL,
        'signal': signal
    }

    return dict_results


###############################################################################
# Relative Strength Index
###############################################################################


def RSI(ys, w=14, ul=70, dl=30):
    l = len(ys)

    ys_chg_p = (ys.diff(1)).apply(lambda x: max(x, 0))
    ys_chg_n = (ys.diff(1)).apply(lambda x: np.abs(min(x, 0)))

    RS = pd.Series(data=np.nan, index=ys.index.tolist())
    RSI = pd.Series(data=np.nan, index=ys.index.tolist())
    for t in range(w, l):
        # print(t)
        if ys_chg_n.iloc[t-w+1:t+1].sum() == 0:
            RS.iloc[t] = np.nan
            RSI.iloc[t] = 100
        else:
            RS.iloc[t] = ys_chg_p.iloc[t - w + 1:t + 1].sum() / ys_chg_n.iloc[t - w + 1:t + 1].sum()
            RSI.iloc[t] = 100 - 100 / (1 + RS.iloc[t])

    if RSI[-1] > dl and RSI[-2] < dl:
        signal = 1
    elif RSI[-1] < ul and RSI[-2] > ul:
        signal = -1
    else:
        signal = 0

    dict_results = {
        'RSI': RSI,
        'signal': signal
    }

    return dict_results


###############################################################################
# Bollinger Bands
###############################################################################


def BB(ys, w=20, k=2):

    BB_mid = SMA(ys, w)['SMA']

    # diff_square = (ys - BB_mid).apply(np.square)
    # sigma = (diff_square.rolling(window=w).mean()).apply(np.sqrt)

    sigma = ys.rolling(window=w).apply(np.std)

    BB_up = BB_mid + k * sigma
    BB_low = BB_mid - k * sigma
    
    ls_ix = ys.index.tolist()
    
    signal = pd.Series(data=np.nan, index=ls_ix)
    
    for i in range(w-1, len(ls_ix)-1):
        if (ys.iloc[i] > BB_up.loc[ls_ix[i]] and ys.iloc[i+1] < BB_up.loc[ls_ix[i+1]]):
            signal.loc[ls_ix[i+1]] = -1
        elif (ys.iloc[i] < BB_low.loc[ls_ix[i]] and ys.iloc[i+1] > BB_low.loc[ls_ix[i+1]]):
            signal.loc[ls_ix[i+1]] = 1
        else:
            signal.loc[ls_ix[i+1]] = 0

    dict_results = {
        'Mid': BB_mid,
        'Up': BB_up,
        'Low': BB_low,
        'signal': signal
    }

    return dict_results


if __name__ == '__main__':
    import talib
    df_ys = pd.read_csv('./Data/ru_i_15min.csv')
    df_ys = pd.read_csv('./Data/IF1903_1min.csv')
    df_ys.datetime = df_ys.datetime.apply(pd.to_datetime)
    df_ys.datetime = df_ys.datetime.apply(lambda x: str(x))
    df_ys.set_index('datetime', inplace=True)
    ls_cols = df_ys.columns.tolist()
    str_Close = [i for i in ls_cols if i[-6:] == '.close'][0]
    ys = df_ys.loc[:, str_Close]

    ys = ys[-300:]
    #
    # results = MACD(ys)
    # macd, macdsignal, macdhist = talib.MACD(ys, fastperiod=12, slowperiod=26, signalperiod=9)
    # x = macd - results['MACD']
    # y = results['MACD']
    # y.plot()

    ta_RSI = talib.RSI(ys, 14)
    my = RSI(ys)['RSI']

    upperband, middleband, lowerband = talib.BBANDS(ys, 20)
