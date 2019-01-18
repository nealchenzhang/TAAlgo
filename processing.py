# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 14:17:07 2019

@author: chen zhang
"""
import datetime as dt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.dates import DateFormatter, WeekdayLocator, DayLocator, MONDAY,YEARLY
from mpl_finance import candlestick_ohlc
from matplotlib.pylab import date2num

import seaborn as sns
sns.set_style('white')

df_data = pd.read_csv('my_data.csv')
# 注意 这里datetime 是 str 不是datetime64
#df_data.datetime = df_data.datetime.apply(pd.to_datetime)
df_data.set_index('datetime', inplace=True)
df_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

ds_ys = df_data.Close[:120]
# 需要对数据源进行处理 行情中断 10：15-10：29 以及不连续的处理

def RW(ds_ys, w, pflag=0):
    """
    ds_ys: column vector of price series
    w: width of the rolling window
    pflag: plot a graph equals 1
    
    returns: 
        Peaks: data Series with datetime index and price
        Bottoms: data Series with datetime index and price
    """
    # TODO: if l < w how to correct and in real trading enviromnent how to detect
    l = len(ds_ys)
    ls_ix = ds_ys.index.tolist()
    ls_ix_peaks = []
    ls_ix_bottoms = []
    for i in range(w+1, l-w+1):
        print(i)
        if (ds_ys.iloc[i-1] > np.max(ds_ys.iloc[i-w-1: i-1])) and \
            (ds_ys.iloc[i-1] > np.max(ds_ys.iloc[i: i+w])):
#                print(i)
                ls_ix_peaks.append(ls_ix[i-1])
        if (ds_ys.iloc[i-1] < np.min(ds_ys.iloc[i-w-1: i-1])) and \
            (ds_ys.iloc[i-1] < np.min(ds_ys.iloc[i: i+w])):
#                print(i)
                ls_ix_bottoms.append(ls_ix[i-1])
    ds_peaks = pd.Series(index=ls_ix_peaks, data=ds_ys.loc[ls_ix_peaks])
    ds_bottoms = pd.Series(index=ls_ix_bottoms, data=ds_ys.loc[ls_ix_bottoms])
            
    if pflag == 1:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.plot(ds_ys)
        ax.scatter(x=ds_peaks.index, y=ds_peaks, marker='o', color='r', alpha=0.5)
        ax.scatter(x=ds_bottoms.index, y=ds_bottoms, marker='o', color='g', alpha=0.5)
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
        plt.show()
        
    return ds_peaks, ds_bottoms

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

def PIPs(ds_ys, n_PIPs, type_dist, pflag=0):
    """

    :param ds_ys: column vector of price series with time index
    :param n_PIPs: number of requested PIPs
    :param type_dist: 1 = Euclidean Distance ED,
                      2 = Perpendicular Distance PD,
                      3 = Vertical Distance VD
    :param pflag: 1 = plot graph
    :return PIPxy: pandas Series with time index and
                   PIPs' price in column named y
                   indicating coordinates of PIPs
    """
    l = len(ds_ys)
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
#             print(j)
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
        Adjy = pd.DataFrame(data=[ds_ys.iloc[df_Adjacents.iloc[:,0]].values,
                                  ds_ys.iloc[df_Adjacents.iloc[:,1]].values]).T
        Adjx.iloc[Existed_PIPs, :] = np.NaN
        Adjy.iloc[Existed_PIPs, :] = np.NaN
        
        if type_dist == 1:
            D = EDist(ds_ys, xs, Adjx, Adjy)
        elif type_dist == 2:
            D = PDist(ds_ys, xs, Adjx, Adjy)
        else:
            D = VDist(ds_ys, xs, Adjx, Adjy)
        
        D[Existed_PIPs]=0
        Dmax = D.argmax()
        ds_PIP_points.iloc[Dmax] = 1
        currentstate += 1
    PIPxy = pd.Series(index=ds_ys.index[Existed_PIPs], data=ds_ys.iloc[Existed_PIPs])
    
    if pflag == 1:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.plot(ds_ys)
        ax.plot(PIPxy, color='b', alpha=0.7)
        ax.scatter(x=PIPxy.index, y=PIPxy, marker='o', color='g', alpha=0.5)
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
#        plt.tight_layout()
        plt.show()
    return PIPxy

if __name__ == '__main__':
    df_data = pd.read_csv('my_data.csv')
    # 注意 这里datetime 是 str 不是datetime64
    #df_data.datetime = df_data.datetime.apply(pd.to_datetime)
    df_data.set_index('datetime', inplace=True)
    df_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    ds_ys = df_data.Close[:120]
    # 需要对数据源进行处理 行情中断 10：15-10：29 以及不连续的处理
    PIPs(ds_ys, 6, type_dist=1, pflag=1)
