# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 11:05:23 2019

@author: chen zhang
"""

import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

from scipy import stats

from technical_indicators import SMA

strategy = SMA
dict_results = strategy(ys, w=5)


def pair_test_of_signal(rtn_signal, rtn, **kwargs):

    print('H0: strategy signal generated return <= unconditional mean daily return')

    mean_signal, mean_market = rtn_signal.mean(), rtn.mean()
    std_signal, std_market = rtn_signal.std(ddof=1), rtn.std(ddof=1)
    n_signal, n_market = rtn_signal.size, rtn.size
    df = n_signal + n_market - 2
    try:
        mean_signal = kwargs['mean']
    except KeyError:
        pass

    sed = np.sqrt(((n_signal - 1) * std_signal ** 2 + (n_market - 1) * std_market ** 2) / df)

    t_stat = (mean_signal - mean_market) / (sed * np.sqrt(1 / n_signal + 1 / n_market))

    # Critical t-value: one-tailed
    one_tailed_alpha = [0.1, 0.05, 0.01]
    print('-' * 40)
    print('Calculated t_stats is {}.\nWith df = {}'.format(t_stat, df))
    for alpha in one_tailed_alpha:
        c_t = stats.t.ppf(1 - alpha, df=df)
        if t_stat > c_t or t_stat < -c_t:
            print('Reject the null hypothesis at the {:.2%} level of significance'.format(alpha))
        else:
            print('We failed to reject the null hypothesis at the {:.2%} level of significance'.format(alpha))

    # Check p_value
    pval = stats.t.sf(np.abs(t_stat), df)

    return pval


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
    position[start_ix] = open_signal[start_ix] = signal[start_ix]
    ls_open_ix = []

    for t in range(start_ix, num_x):
        if np.abs(position[t-1] + signal[t]) > np.abs(position[t-1]):
            # Open
            open_signal[t] = signal[t]
            ls_open_ix.append(t)
        elif np.abs(position[t-1] + signal[t]) < np.abs(position[t-1]):
            # Close
            close_signal[t] = -open_signal[ls_open_ix[0]]
            ls_open_ix = ls_open_ix[1:]
        if len(ls_open_ix) != 0 and t == ls_open_ix[0] + HP:
            # Close HP
            close_signal[t] = -open_signal[ls_open_ix[0]]
            ls_open_ix = ls_open_ix[1:]

        pos_chg[t] = open_signal[t] + close_signal[t]
        position[t] = pos_chg[t] + position[t-1]

    ##################################################################################
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
    x['rtn'] = (ys / ys.shift(1)).apply(np.log)
    ####################################################################################
    N_buy_signal = signal.where(signal > 0).count()
    N_sell_signal = signal.where(signal < 0).count()
    N_long = position.where(position > 0).count()
    N_short = position.where(position < 0).count()
    log_return = (ys / ys.shift(1)).apply(np.log)
    Rtn_long = log_return.loc[position.where(position > 0).dropna().index.tolist()]
    Rtn_short = log_return.loc[position.where(position < 0).dropna().index.tolist()]
    Rtn_long_mean = Rtn_long.mean()
    Rtn_short_mean = Rtn_short.mean()

    Rtn_LS = (Rtn_long.append(-Rtn_short)).sort_index()
    Rtn_LS_mean = (N_long * Rtn_long_mean - N_short * Rtn_short_mean) / (N_long + N_short)

    p_value_buy = pair_test_of_signal(Rtn_long, log_return)
    p_value_sell = pair_test_of_signal(Rtn_short, log_return)
    p_value_BS = pair_test_of_signal(Rtn_LS, log_return, mean=Rtn_LS_mean)

    dict_assessment = {
            'N_buy': N_buy_signal,
            'N_sell': N_sell_signal,
            'N_long_days': N_long,
            'N_short_days': N_short,
            
            }

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

