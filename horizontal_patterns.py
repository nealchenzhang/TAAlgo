# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 17:34:07 2019

@author: chen zhang
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style('white')

from preprocessing import RW
from preprocessing import TP

# TODO fibonacci

def HSAR(ys, w, x, iteration):
    """
    The HSAR(*) identifies HSARs at time t conditioning information up to time t-1.
    Function RW from processing is embedded.

    :param ys: column vector of price series with time index
    :param w: width of the rolling window (total 2w+1)
    :param x: desired percentage that will give the bounds for the HSARs (e.g., 5%)
    warning: for commodity prices which base price is large, I consider calculate x
    based on the minimum price change, where x = minimum price change / ys.mean()
    :param pflag: plot a graph if 1
    :return:
        SAR: horizontal support and resistance levels
        Bounds: bounds of bins used to classify the peaks and bottoms
        Freq: frequencies for each bin
        x_act: actual percentage of the bins' distance
    """
    Peaks, Bottoms = RW(ys, w, iteration)
    # TODO: commodity price is not continuously changing, should be careful with
    # TODO: data preprocessing
    L = Peaks.append(Bottoms)
    x = 10/ys.mean()
    L1 = L.min() / (1+x/2)
    Ln = L.max() * (1+x/2)
    # My modification to calculate number of bins, for the problem that
    # for commodity prices, the base price is large, so pct bandwidth is hard
    # to find
    n = np.log(Ln/L1) / np.log(1+x)
    N_act = np.int(np.round(n))
    x_act = (Ln/L1)**(1/N_act)-1
    
    Bounds = L1 * (1+x_act)**(pd.Series(np.arange(0, N_act+1)))
    Freq = [0]*N_act
    for i in range(0, N_act):
        Freq[i] = np.count_nonzero((L>=Bounds[i])&(L<Bounds[i+1]))
    ls_idx = [i for i,x in enumerate(Freq) if x>=2]
    
    if len(ls_idx) == 0:
        print('No HSARs identified')
        SAR = []
    else:
        m = len(ls_idx)
        SAR = (Bounds[ls_idx].values + Bounds[[i+1 for i in ls_idx]].values)/2
        # Plotting
        if pflag == 1:
            ls_x = ys.index.tolist()
            num_x = len(ls_x)
            ls_time_ix = np.linspace(0,num_x-1,num_x)
            
            ls_p = Peaks.index.tolist()
            ls_b = Bottoms.index.tolist()
            ls_peaks_time = [ys.index.get_loc(x) for x in ls_p]
            ls_bottoms_time = [ys.index.get_loc(x) for x in ls_b]
            
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            ax.plot(ls_time_ix, ys.values)
            ax.scatter(x=ls_peaks_time, y=Peaks.values, marker='o', color='r', alpha=0.5)
            ax.scatter(x=ls_bottoms_time, y=Bottoms.values, marker='o', color='g', alpha=0.5)
            
            Pxy = pd.Series()
            Bxy = pd.Series()
            for i in range(0, m):
                ls_Pc = (Peaks.index.where((Peaks>=Bounds[ls_idx[i]]) & \
                                           (Peaks<=Bounds[ls_idx[i]+1]))).tolist()
                ls_Pc = [x for x in ls_Pc if str(str(x).lower()) != 'nan']
                Pxy = Pxy.append(Peaks.loc[ls_Pc])
                ls_Bc = (Bottoms.index.where((Bottoms>=Bounds[ls_idx[i]]) & \
                                             (Bottoms<=Bounds[ls_idx[i]+1]))).tolist()
                ls_Bc = [x for x in ls_Bc if str(str(x).lower()) != 'nan']
                Bxy = Bxy.append(Bottoms.loc[ls_Bc])
                
                ls_pxy = Peaks.loc[ls_Pc].index.tolist()
                ls_bxy = Bottoms.loc[ls_Bc].index.tolist()
                ls_pHSAR_time = [ys.index.get_loc(x) for x in ls_pxy]
                ls_bHSAR_time = [ys.index.get_loc(x) for x in ls_bxy]
                for ls_time in [ls_pHSAR_time, ls_bHSAR_time]:
                    if len(ls_pHSAR_time) == 0:
                        pass
                    else:
                        for time in ls_time:
                            ax.hlines(y=SAR[i], xmin=time-num_x/10, xmax=time+num_x/10, colors='r', alpha=0.4, linestyles='dotted')
            plt.show()               

    return SAR, Bounds, Freq, x_act


