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
def pair_tests(ys, h, signal, rtn_type='log'):
    """
    
    :param ys: original data Series
    :param h: predefined fixed holding periods
    :param signal: signal Series generated with time index
    :param rtn_type: definition of return types
                    'log': logarithmic returns
                    'mean': arithmetic returns
    :return:
    """
    rtn = (ys.shift(h) / ys).apply(np.log)

    # TODO: refer to null hypothesis test
    print('# 2. Population Correlation Coefficient Test')
    print('-' * 40)
    print('H0: rho = 0')
    number = self.data_set.loc[:, x].count()
    sample_corr = self.data_set.loc[:, [x, y]].corr(method='pearson')[x][y]
    cal_ttest = (sample_corr * np.sqrt(number - 2)) / np.sqrt(1 - sample_corr ** 2)
    # Critical t-value: 2-tailed
    two_tailed_alpha = [0.1, 0.05, 0.01]
    from scipy import stats
    print('-' * 40)
    print('Calculated t_value is {}.\nWith df = {}'.format(cal_ttest, number - 2))
    for i in two_tailed_alpha:
        c_t = stats.t.ppf(1 - i / 2, df=number - 2)
        if (cal_ttest < -c_t) or (cal_ttest > c_t):
            print('Reject the null hypothesis at the {:.2%} level of significance'.format(i))
        else:
            print('We failed to reject the null hypothesis at the {:.2%} level of significance'.format(i))


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
