
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style('white')

from horizontal_patterns import HSAR
from horizontal_patterns import HSAR_plotting

from preprocessing import RW
from preprocessing import TP
#from preprocessing import PIPs

from preprocessing import PB_plotting


def HSARsim(ys, pflag, x,, indata, method='RW', **kwargs):
    """
    
    :param ys: 
    :param pflag: 
    :param x: 
    :param indata: an integer number defining the size of the initial subsample used for the 
    identification of the first HSARs
    :param method: 
    :param kwargs: 
    :return: 
    """

    if method == 'RW':
        Peaks, Bottoms = RW(ys, w=kwargs['w'], iteration=kwargs['iteration'])
    elif method == 'TP':
        Peaks, Bottoms = TP(ys, iteration=kwargs['iteration'])

    l = len(ys)

    if l>250:
        pflag = 0

    if pflag == 1:
        PB_plotting(ys, Peaks, Bottoms)

    dict_Results = {
        'NofSARs': [np.nan]*l,
        'x_act': [np.nan]*l,
        # 'ActSARs': ,
        # 'PofSARs':
    }

    indata = 500
    for i in range(indata, l-1):
        print(i)
        SAR, Bounds, Freq, x_act, Peaks, Bottoms = HSAR(
            ys[:(i-1)], pflag=0, x=10, method='RW', w=1, iteration = 1
        )
        NofSARs = len(SAR)
        if NofSARs > 0:
            PofSARs = [i for i in Freq if i >= 2]
        else:
            SAR = np.nan
            PofSARs = np.nan
        dict_Results['NofSARs'][i] = NofSARs
        dict_Results['x_act'][i] = x_act
        a = 100 - NofSARs
        if NofSARs > 0:
            a = [np.nan] * a
        else:
            a = [np.nan] *
