# -*- coding: utf-8 -*-
"""
Created on Tue Feb 2 13:04:07 2019

@author: chen zhang
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style('white')

###############################################################################
# This preprocessing module is to identify reginal locals
# Different methods are presented here
###############################################################################


def PB_plotting(ys, Peaks, Bottoms, savefig=False):
    # TODO: xaxis optimization for display
    ls_x = ys.index.tolist()
    num_x = len(ls_x)
    ls_time_ix = np.linspace(0,num_x-1,num_x)
    ls_p = Peaks.index.tolist()
    ls_b = Bottoms.index.tolist()
    ls_peaks_time = [ys.index.get_loc(x) for x in ls_p]
    ls_bottoms_time = [ys.index.get_loc(x) for x in ls_b]
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.plot(ls_time_ix, ys.values)
#        plt.show()
    ax.scatter(x=ls_peaks_time, y=Peaks.values, marker='o', color='r', alpha=0.5)
    ax.scatter(x=ls_bottoms_time, y=Bottoms.values, marker='o', color='g', alpha=0.5)
    
    # for i in ls_peaks_time:
    #     ax.text(x=i, y=Peaks.loc[ls_x[i]],
    #             s=ls_x[i], withdash=True,
    #             )
    new_xticklabels = [ls_x[np.int(i)] for i in list(ax.get_xticks()) if i in ls_time_ix]
    new_xticklabels = [ls_x[0]] + new_xticklabels
    new_xticklabels.append(ls_x[-1])
    ax.set_xticklabels(new_xticklabels)
    for tick in ax.get_xticklabels():
        tick.set_rotation(15)
#        plt.savefig('1.jpg')
    plt.show()

###############################################################################
# Useful tools
# elimination function to filter peaks and bottoms
###############################################################################


def elimination(Peaks, Bottoms):
    ls_ix_peaks = Peaks.index.tolist()
    ls_ix_bottoms = Bottoms.index.tolist()
    ds_TPs = Peaks.append(Bottoms)
    ds_TPs.sort_index(ascending=True, inplace=True)

    l_TPs = len(ds_TPs)
    ls_ix_TPs = ds_TPs.index.tolist()
    ls_ix_TPs_new = []
    i = 0
    while i < (l_TPs-3):
        ls_ix_TPs_new.append(ls_ix_TPs[i])
#        print(ls_ix_TPs_new)
        if ((ds_TPs.iloc[i]<ds_TPs.iloc[i+1]) and (ds_TPs.iloc[i]<ds_TPs.iloc[i+2]) \
                and (ds_TPs.iloc[i+1]<ds_TPs.iloc[i+3]) and (ds_TPs.iloc[i+2]<ds_TPs.iloc[i+3]) \
                and (np.abs(ds_TPs.iloc[i+1]-ds_TPs.iloc[i+2])<(np.abs(ds_TPs.iloc[i]-ds_TPs.iloc[i+2])+ \
                                                                np.abs(ds_TPs.iloc[i+1]-ds_TPs.iloc[i+3])))) \
            or ((ds_TPs.iloc[i]>ds_TPs.iloc[i+1]) and (ds_TPs.iloc[i]>ds_TPs.iloc[i+2]) \
                and (ds_TPs.iloc[i+1]>ds_TPs.iloc[i+3]) and (ds_TPs.iloc[i+2]>ds_TPs.iloc[i+3]) \
                and (np.abs(ds_TPs.iloc[i+2]-ds_TPs.iloc[i+1])<(np.abs(ds_TPs.iloc[i]-ds_TPs.iloc[i+2])+ \
                                                                np.abs(ds_TPs.iloc[i+1]-ds_TPs.iloc[i+3])))) \
            or ((np.abs(ds_TPs.iloc[i]/ds_TPs.iloc[i+2]-1)<0.0002) and\
                (np.abs(ds_TPs.iloc[i+1]/ds_TPs.iloc[i+3]-1)<0.0002)):
            # TODO: approximately equal in price (hard to define, have to consider the minimum variation)
            # Here I used the ratio less than a threshold = 0.0002
            # The threshold could be minimum variation / the previous day settlement? maybe an improvement
            ls_ix_TPs_new.append(ls_ix_TPs[i+3])
            i += 3
        else:
            i += 1
    if i == l_TPs-3:
        ls_ix_TPs_new.append(ls_ix_TPs[i])
        ls_ix_TPs_new.append(ls_ix_TPs[i+1])
        ls_ix_TPs_new.append(ls_ix_TPs[i+2])
    elif i == l_TPs-2:
        ls_ix_TPs_new.append(ls_ix_TPs[i])
        ls_ix_TPs_new.append(ls_ix_TPs[i+1])
    elif i == l_TPs-1:
        ls_ix_TPs_new.append(ls_ix_TPs[i])
                
    ls_ix_peaks_new = [peak for peak in ls_ix_peaks if peak in ls_ix_TPs_new]
    ls_ix_bottoms_new = [bottom for bottom in ls_ix_bottoms if bottom in ls_ix_TPs_new]

    Peaks_new = pd.Series(index=ls_ix_peaks_new, data=ds_TPs.loc[ls_ix_peaks_new])
    Bottoms_new = pd.Series(index=ls_ix_bottoms_new, data=ds_TPs.loc[ls_ix_bottoms_new])

    return Peaks_new, Bottoms_new

###############################################################################
# Method 1: Rolling Window
###############################################################################


def RW(ys, w, iteration=0):
    """
    Rolling window method
    
    ys: column vector of price series with str datetime index
    w: width of the rolling window (total 2w+1)
    when w = 1 Rolling Window == Turning Points Methods
    iteration: 0 means no iteration
    
    returns: 
        Peaks: data Series with datetime index and price
        Bottoms: data Series with datetime index and price
    """
    # if l < w how to correct and in real trading environment how to detect
    l = len(ys)
    ls_ix = ys.index.tolist()
    ls_ix_peaks = []
    ls_ix_bottoms = []
    for i in range(w+1, l-w+1):
        if (ys.iloc[i-1] > np.max(ys.iloc[i-w-1: i-1])) and \
            (ys.iloc[i-1] > np.max(ys.iloc[i: i+w])):
                ls_ix_peaks.append(ls_ix[i-1])
        if (ys.iloc[i-1] < np.min(ys.iloc[i-w-1: i-1])) and \
            (ys.iloc[i-1] < np.min(ys.iloc[i: i+w])):
                ls_ix_bottoms.append(ls_ix[i-1])
    Peaks = pd.Series(index=ls_ix_peaks, data=ys.loc[ls_ix_peaks])
    Bottoms = pd.Series(index=ls_ix_bottoms, data=ys.loc[ls_ix_bottoms])
    
    itero = 0
    while itero < iteration:
        Peaks_new, Bottoms_new = elimination(Peaks, Bottoms)
        Peaks = Peaks_new
        Bottoms = Bottoms_new
        itero += 1
    else:
        pass
    return Peaks, Bottoms

###############################################################################
# Method 2: Turning Points
###############################################################################


def TP(ys, iteration=0):
    """
    This method is based on the paper after Jiangling Yin, Yain-Whar Si, Zhiguo Gong
        Financial Time Series Segmentation Based on Turning Points
    ys: column vector of price series
    iteration: 0 means no iteration

    returns:
        Peaks: data Series with datetime index and price
        Bottoms: data Series with datetime index and price
    """
    l = len(ys)
    ls_ix = ys.index.tolist()
    ls_ix_peaks = []
    ls_ix_bottoms = []
    for i in range(1, l-1):
        # print(i)
        if (ys.iloc[i - 1] > ys.iloc[i]) and (ys.iloc[i + 1] > ys.iloc[i]):
            ls_ix_bottoms.append(ls_ix[i])
        if (ys.iloc[i - 1] < ys.iloc[i]) and (ys.iloc[i + 1] < ys.iloc[i]):
            ls_ix_peaks.append(ls_ix[i])
    Peaks = pd.Series(index=ls_ix_peaks, data=ys.loc[ls_ix_peaks])
    Bottoms = pd.Series(index=ls_ix_bottoms, data=ys.loc[ls_ix_bottoms])
    
    itero = 0
    while itero < iteration:
        Peaks_new, Bottoms_new = elimination(Peaks, Bottoms)
        Peaks = Peaks_new
        Bottoms = Bottoms_new
        itero += 1
    else:
        pass

    return Peaks, Bottoms

###############################################################################
# Method 3: Perceptually Important Points
# Output of this method should be used with caution
###############################################################################


def EDist(ys, xs, Adjx, Adjy):
    ED = ((Adjx.iloc[:, 1].values - xs.values)**2 + (Adjy.iloc[:, 1].values - ys.values)**2)**(1/2)+\
    ((Adjx.iloc[:, 0].values - xs.values)**2 + (Adjy.iloc[:, 0].values - ys.values)**2)**(1/2)
    return ED


def PDist(ys, xs, Adjx, Adjy):
    slopes = (Adjy.iloc[:, 1].values - Adjy.iloc[:, 0].values) / (Adjx.iloc[:, 1].values - Adjx.iloc[:, 0].values)
    constants = Adjy.iloc[:, 1].values - slopes * Adjx.iloc[:, 1].values
    PD = np.abs(slopes * xs.values - ys.values + constants) / ((slopes**2 + 1)**(1/2))
    return PD


def VDist(ys, xs, Adjx, Adjy):
    slopes = (Adjy.iloc[:, 1].values - Adjy.iloc[:, 0].values) / (Adjx.iloc[:, 1].values - Adjx.iloc[:, 0].values)
    constants = Adjy.iloc[:, 1].values - slopes * Adjx.iloc[:, 1].values
    Yshat = slopes * xs.values + constants
    VD = np.abs(Yshat.values - ys.values)
    return VD


def PIPs(ys, n_PIPs, type_dist=1, pflag=0):
    """

    :param ys: column vector of price series with time index
    :param n_PIPs: number of requested PIPs
    :param type_dist: 1 = Euclidean Distance ED,
                      2 = Perpendicular Distance PD,
                      3 = Vertical Distance VD
    :param pflag: 1 = plot graph
    :return PIPxy: pandas Series with time index and
                   PIPs' price in column named y
                   indicating coordinates of PIPs
    """
    l = len(ys)
    xs = pd.Series(np.arange(0, l))

    ds_PIP_points = pd.Series(np.arange(0, l)) * 0
    ds_PIP_points.iloc[[0, l-1]] = 1

    df_Adjacents = pd.DataFrame(np.zeros((l, 2)))
    for i in df_Adjacents.columns.tolist():
        df_Adjacents.iloc[:, i] = df_Adjacents.iloc[:, i].apply(np.int)
    currentstate = 2 # initial PIPs: the first and the last observation
    
    while currentstate <= n_PIPs:
        Existed_PIPs = ((ds_PIP_points.where(ds_PIP_points==1).dropna()).apply(np.int)).index.tolist()
        currentstate = len(Existed_PIPs)
        locator = pd.DataFrame(np.ones((l, currentstate))* np.NaN)
        for j in range(0, currentstate):
            locator.iloc[:, j] = np.abs(xs - Existed_PIPs[j])
        b1 = [0]*l
        b2 = b1.copy()

        for i in range(0, l):
            b1[i] = locator.iloc[i,:].idxmin()
#            print('b1',b1)
            locator.iloc[i, np.int(b1[i])] = np.NaN
#            print(locator)
            b2[i] = locator.iloc[i,:].idxmin()
#            print('b2',b2)
            df_Adjacents.iloc[i, 0] = Existed_PIPs[np.int(b1[i])]
            df_Adjacents.iloc[i, 1] = Existed_PIPs[np.int(b2[i])]

        Adjx = df_Adjacents
        Adjy = pd.DataFrame(data=[ys.iloc[df_Adjacents.iloc[:,0]].values,
                                  ys.iloc[df_Adjacents.iloc[:,1]].values]).T
        Adjx.iloc[Existed_PIPs, :] = np.NaN
        Adjy.iloc[Existed_PIPs, :] = np.NaN
        
        if type_dist == 1:
            D = EDist(ys, xs, Adjx, Adjy)
        elif type_dist == 2:
            D = PDist(ys, xs, Adjx, Adjy)
        else:
            D = VDist(ys, xs, Adjx, Adjy)
        
        D[Existed_PIPs]=0
        Dmax = D.argmax()
        ds_PIP_points.iloc[Dmax] = 1
        currentstate += 1
    PIPxy = pd.Series(index=ys.index[Existed_PIPs], data=ys.iloc[Existed_PIPs])
#    TODO: ways to divide PIPs into Peaks and Bottoms?
    if pflag == 1:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.plot(ys)
        ax.plot(PIPxy, color='b', alpha=0.7)
        ax.scatter(x=PIPxy.index, y=PIPxy, marker='o', color='g', alpha=0.5)
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
#        plt.tight_layout()
        plt.show()
    return PIPxy


if __name__ == '__main__':

    df_data = pd.read_csv('./Data/my_data.csv')
    # 注意 这里datetime 是 str 不是datetime64
    df_data.set_index('datetime', inplace=True)
    df_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    ys = df_data.Close[:120]
    Peaks, Bottoms = RW(ys, w=4, iteration=0)
    PB_plotting(ys, Peaks, Bottoms, savefig=False)
#    Peaks, Bottoms=TP(ys, iteration=1)
#    PB_plotting(ys, Peaks, Bottoms, savefig=False)

    PIPs(ys, 5, )

