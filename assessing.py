# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 15:41:25 2019

@author: chen zhang
"""

import numpy as np
import pandas as pd
from scipy import stats

# Holding Periods
# Price target
# Predefined time limit
# Stop loss

# Passive or fixed time periods

# Assessing the Performance of Trading Signals
def pair_tests(ys, signal, h=5, rtn_type='log'):
    """
    
    :param ys: original data Series
    :param signal: signal Series generated with time index
    :param h: predefined fixed holding periods
    :param rtn_type: definition of return types
                    'log': logarithmic returns
                    'mean': arithmetic returns
    :return: h
    """
    ls_ix = ys.index.tolist()
    ls_ix_signal = signal.index.tolist()
    ls_ix_signal_nan = [i for i in ls_ix_signal if i not in signal.dropna().index.tolist()]
    rtn = (ys.shift(h) / ys).apply(np.log)
    rtn.loc[ls_ix_signal_nan] = np.nan
    # TODO: check the normality of rtn

    rtn_signal = rtn.copy()
    rtn_signal.loc[signal.where(signal == 0).dropna().index.tolist()] = np.nan
    rtn_signal.loc[signal.where(signal == 1).dropna().index.tolist()] *= 1
    rtn_signal.loc[signal.where(signal == -1).dropna().index.tolist()] *= -1
    rtn_signal.loc[ls_ix_signal_nan] = np.nan
    rtn_signal.dropna(inplace=True)
    rtn.dropna(inplace=True)

    print('Null Hypothesis: rtn_signal = rtn')
    print('Alternative Hypothesis: rtn_signal > rtn')
    mean_signal, mean_market = rtn_signal.mean(), rtn.mean()
    std_signal, std_market = rtn_signal.std(ddof=1), rtn.std(ddof=1)
    n_signal, n_market = rtn_signal.size, rtn.size
    se_signal, se_market = std_signal / np.sqrt(n_signal), std_market / np.sqrt(n_market)

    sed = np.sqrt(se_signal ** 2 + se_market ** 2)
    df = n_signal + n_market - 2
    t_stat = (mean_signal - mean_market) / sed

    # Critical t-value: one-tailed
    one_tailed_alpha = [0.1, 0.05, 0.01]
    print('-' * 40)
    print('Calculated t_stats is {}.\nWith df = {}'.format(t_stat, df))
    for alpha in one_tailed_alpha:
        c_t = stats.t.ppf(1 - alpha, df=df)
        if t_stat > c_t:
            print('Reject the null hypothesis at the {:.2%} level of significance'.format(alpha))
            print('Good to go with fixed {} holding period'.format(h))
        else:
            print('We failed to reject the null hypothesis at the {:.2%} level of significance'.format(alpha))

    return h

# TODO: Bernoulli trials
def Bernoulli_trials(p, N):
    """
    p: probability of success: 50%
    N: number of trials
    x: number of successful cases
    """
    mean: p*N
    sigma: (N*p*(1-p))**(1/2)
    z-stat = (x-p*N) / (N*p*(1-p))**(1/2)

def Bootstrap_Approach():

    return None
    
# Assessing the Performance of Predicting Returns    
    
if __name__ == '__main__':
    
    df_ys = pd.read_csv('./Data/ru_i_15min.csv')
#    df_ys = pd.read_csv('./Data/IF1903_1min.csv')
    df_ys.datetime = df_ys.datetime.apply(pd.to_datetime)
    df_ys.datetime = df_ys.datetime.apply(lambda x: str(x))
    df_ys.set_index('datetime', inplace=True)
    ls_cols = df_ys.columns.tolist()
    str_Close = [i for i in ls_cols if i[-6:] == '.close'][0]
    ys = df_ys.loc[:, str_Close]

    ys = ys[-300:]

    from technical_indicators import BB
    signal = BB(ys)['signal']
