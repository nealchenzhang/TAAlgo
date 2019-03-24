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


# TODO: check the normality of rtn

# Assessing the Performance of Trading Signals


def pair_test(ys, signal, hp=5, rtn_type='log'):
    """
    
    :param ys: original data Series
    :param signal: signal Series generated with time index
    :param hp: predefined fixed holding periods
    :param rtn_type: definition of return types
                    'log': logarithmic returns
                    'mean': arithmetic returns
    :return: h
    """
    ls_ix_signal = signal.index.tolist()
    ls_ix_signal_nan = [i for i in ls_ix_signal if i not in signal.dropna().index.tolist()]
    rtn = (ys.shift(hp) / ys).apply(np.log)
    rtn.loc[ls_ix_signal_nan] = np.nan

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
    df = n_signal + n_market - 2

    sed = np.sqrt(((n_signal-1)*std_signal**2 + (n_market-1)*std_market**2)/df)
    
    t_stat = (mean_signal - mean_market) / (sed * np.sqrt(1/n_signal + 1/n_market))

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

    return hp


def Bernoulli_trials(x, N, p=0.5):
    """
    p: probability of success: 50%
    N: number of trials (different assets)
    x: number of successful cases 
        (where trading rule generates positive profits)
    """
#    mean: p*N
#    sigma: (N*p*(1-p))**(1/2)
    z_stat = (x-p*N) / (N*p*(1-p))**(1/2)
    
    print('Null Hypothesis: x = p*N')
    print('Alternative Hypothesis: x < p*N')
        
    # Critical z-value: one-tailed
    one_tailed_alpha = [0.1, 0.05, 0.01]
    print('-' * 40)
    print('Calculated z_stats is {}.'.format(z_stat))
    for alpha in one_tailed_alpha:
        c_t = stats.norm.ppf(1 - alpha)
        if z_stat > c_t:
            print('Reject the null hypothesis at the {:.2%} level of significance'.format(alpha))
            print('Good to go with the strategy.')
        else:
            print('We failed to reject the null hypothesis at the {:.2%} level of significance'.format(alpha))


def Bootstrap_Approach(ys):
    log_return = (ys / ys.shift(1)).apply(np.log)
    mean = log_return.mean()
    std_dev = log_return.std()

    # Calculate sample bias-corrected skewness
    N = log_return.size
    g1 = (((log_return - mean) ** 3).sum() / N) / ((((log_return - mean) ** 2).sum() / N) ** 3 / 2)
    G1 = (N * (N - 1)) ** 0.5 * g1 / (N - 2)
    # Significance test of skewness
    SES = (6*N*(N-1) / ((N-2)*(N+1)*(N+3)))**0.5
    # H0: G1 = 0
    # H1: G1 != 0
    ZG1 = G1 / SES
    
    print('Null Hypothesis: G1 = 0')
    print('Alternative Hypothesis: G1 != 0')
        
    # Critical z-value: two-tailed
    two_tailed_alpha = [0.05, 0.01]
    print('-' * 40)
    print('Calculated z_stats is {}.'.format(ZG1))
    skewness_significance = 0
    for alpha in two_tailed_alpha:
        c_t = stats.norm.ppf(1 - alpha / 2)
        if ZG1 > c_t:
            print('Reject the null hypothesis at the {:.2%} level of significance'.format(alpha))
            skewness_significance = alpha
        else:
            print('We failed to reject the null hypothesis at the {:.2%} level of significance'.format(alpha))
    
    # Calculate sample bias-corrected kurtosis
    N = log_return.size
    g2 = (((log_return - mean) ** 4).sum() / N) / ((((log_return - mean) ** 2).sum() / N) ** 2)
    G2 = (N - 1)/((N-2)*(N-3)) *((N+1)*g2-3(N-1))+3
    # Significance test of kurtosis
    SEK = 2*SES*((N**2-1)/((N-3)*(N+5)))**0.5
    # H0: G2 = 3
    # H1: G2 != 3
    ZG2 = G2 / SEK
    
    print('Null Hypothesis: G2 = 3')
    print('Alternative Hypothesis: G2 != 3')
        
    # Critical z-value: two-tailed
    two_tailed_alpha = [0.05, 0.01]
    print('-' * 40)
    print('Calculated z_stats is {}.'.format(ZG2))
    kurtosis_significance = 0
    for alpha in two_tailed_alpha:
        c_t = stats.norm.ppf(1 - alpha / 2)
        if ZG2 > c_t:
            print('Reject the null hypothesis at the {:.2%} level of significance'.format(alpha))
            kurtosis_significance = alpha
        else:
            print('We failed to reject the null hypothesis at the {:.2%} level of significance'.format(alpha))


    dict_stats = {
        'Mean': mean,
        'Std': std_dev,
        'Skewness': {
            'value': G1,
            'significance': skewness_significance
        },
        'Kurtosis': {
            'value': G2,
            'significance': kurtosis_significance
        },
        'KS_stat': {
            'value': None,
            'significance': None
        }
    }
        
    return dict_stats
    
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

