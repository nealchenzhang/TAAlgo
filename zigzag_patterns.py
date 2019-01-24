# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 16:13:07 2019

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

ys = df_data.Close[:300]
# 需要对数据源进行处理 行情中断 10：15-10：29 以及不连续的处理
from processing import RW

#class 



def HS(ys, w, pflag):
    """
    The HS(*) identifies Head and Shoulders Pattern on a price series by
    adopting the conditions presented in (Lucke 2003).
    e.g., HS tops pattern
        1) Head recognition: P2 > max(P1, P3)
        2) Trend preexistence: 
                    P1 > P0 & T1 > T0 (trend reversal pattern)
                    P1 < P0 & T1 < T0 (trend continuation pattern)
        3) Balance:
                    P1>=0.5(P3+T2) & P3>=0.5(P1+T1)
        4) Symmetry:
                    tP2 - tP1 < 2.5(tP3 - tP2) &
                    tP3 - tP2 < 2.5(tP2 - tP1)
        5) Penetration:
                    B < ((T2-T1)/(tT2-tT1))*(tB-tT1)+T1
                    tB < tP3 + (tP3 - tP1)

    :param ys: column vector of price series with time index
    :param w: width of the rolling window (total 2w+1)
    :param pflag: plot a graph if 1
    :return:
        TODO:
        ##################################################################
        SAR: horizontal support and resistance levels
        Bounds: bounds of bins used to classify the peaks and bottoms
        Freq: frequencies for each bin
        x_act: actual percentage of the bins' distance
        ##################################################################
    """
    l = len(ys)
    Peaks, Bottoms = RW(ys, w, pflag=0)
    
    ls_x = ys.index.tolist()
    ls_p = Peaks.index.tolist()
    ls_b = Bottoms.index.tolist()
    P_idx = [ys.index.get_loc(x) for x in ls_p]
    B_idx = [ys.index.get_loc(x) for x in ls_b]
    
    P_idx = pd.Series(index=ls_p, data=[1]*len(ls_p))
    B_idx = pd.Series(index=ls_b, data=[2]*len(ls_b))
    PB_idx = P_idx.append(B_idx)
    PB_idx.sort_index(inplace=True)
    m = len(PB_idx.index)
    
    Pot_Normal = [1,2,1,2,1]
    Pot_Inverse = [2,1,2,1,2]
    Pot_Index = [0]*m
    
    for i in range(0, m-4):
        if PB_idx.iloc[i:i+5].values.tolist() == Pot_Normal:
            Pot_Index[i+4] = 1
        elif PB_idx.iloc[i:i+5].values.tolist() == Pot_Inverse:
            Pot_Index[i+4] = 2
    
    PNidx = [i for i,x in enumerate(Pot_Index) if x==1]
    PIidx = [i for i,x in enumerate(Pot_Index) if x==2]
    
    ## HS Tops (Normal Form)
    # Definition of outputs
    Patterns_Normal_Points = []
    Patterns_Normal_Necklines = []
    Patterns_Normal_Breakpoints = []
    
    
    mn = len(PNidx)
    Pot_Normalcases_Percases = [0] * mn# bug??
    if mn != 0:
        Pot_Normalcases_Idx = [0] * mn
        for i in range(0, mn):
            # Conditions check
            PerCase = pd.DataFrame(columns=['P/B', 'Price'])
            PerCase.loc[:, 'P/B'] = PB_idx.iloc[PNidx[i]-4: PNidx[i]+1]
            PerCase.loc[:, 'Price'] = ys.loc[PerCase.index.tolist()].values
            # Condition 1 3 4
            if ((PerCase.iloc[2]['Price'] > np.max([PerCase.iloc[0]['Price'],
                                                    PerCase.iloc[4]['Price']])) &
                (PerCase.iloc[0]['Price'] >= 0.5 * np.sum([PerCase.iloc[3]['Price'],
                                                           PerCase.iloc[4]['Price']])) &
                (PerCase.iloc[4]['Price'] >= 0.5 * np.sum([PerCase.iloc[0]['Price'],
                                                          PerCase.iloc[1]['Price']])) &
                ((ls_x.index(PerCase.index.tolist()[2]) - ls_x.index(PerCase.index.tolist()[0])) <
                 2.5*(ls_x.index(PerCase.index.tolist()[4]) - ls_x.index(PerCase.index.tolist()[2]))) &
                ((ls_x.index(PerCase.index.tolist()[4]) - ls_x.index(PerCase.index.tolist()[2])) <
                 2.5 * (ls_x.index(PerCase.index.tolist()[2]) - ls_x.index(PerCase.index.tolist()[0])))):
                Pot_Normalcases_Idx[i] = 1
                Pot_Normalcases_Percases[i] = PerCase # bug??
    else:
        Pot_Normalcases_Idx = 0

    mnn = np.sum(Pot_Normalcases_Idx)
    Pot_Normalcases_Idx2 = [i for i,x in enumerate(Pot_Normalcases_Idx) if x==1]
    j = 0
    if mnn!=0:
        for i in range(0, mnn):
#            print(i)
            NPerCase = Pot_Normalcases_Percases[Pot_Normalcases_Idx2[i]]
            Timelimit = ls_x.index(NPerCase.index.tolist()[4]) + ls_x.index(NPerCase.index.tolist()[4]) -\
            ls_x.index(NPerCase.index.tolist()[0])
            Neckline_Beta = (NPerCase.iloc[3]['Price'] - NPerCase.iloc[1]['Price']) /\
            (ls_x.index(NPerCase.index.tolist()[3]) - ls_x.index(NPerCase.index.tolist()[1]))
            Neckline_Alpha = NPerCase.iloc[1]['Price'] - Neckline_Beta * ls_x.index(NPerCase.index.tolist()[1])
            if Timelimit <=l:
                Neckline_OU = list(range(ls_x.index(NPerCase.index.tolist()[4])+1, Timelimit))
            else:
                Neckline_OU = list(range(ls_x.index(NPerCase.index.tolist()[4])+1, l))
            
            Neckline_OU = pd.DataFrame(index=[ls_x[i] for i in Neckline_OU], data=ys.iloc[Neckline_OU])
            Neckline_OU.columns = ['Price']
            Neckline_OU.loc[:, 'Line'] = [ls_x.index(i) for i in Neckline_OU.index.tolist()]
            Neckline_OU.loc[:, 'Line'] = Neckline_OU.loc[:, 'Line'] * Neckline_Beta + Neckline_Alpha
            Neckline_OU.loc[:, 'Diff'] = Neckline_OU.loc[:, 'Price'] - Neckline_OU.loc[:, 'Line']
            # TODO: optimize codes
            if len((Neckline_OU.where(Neckline_OU.loc[:, 'Diff']<0).dropna()).index.tolist()) != 0:
                Neckline_Breakpoint = ls_x.index((Neckline_OU.where(Neckline_OU.loc[:, 'Diff']<0).dropna()).index.tolist()[0])
                j += 1
                Patterns_Normal_Points.append(NPerCase)
                Patterns_Normal_Necklines.append([Neckline_Alpha, Neckline_Beta])
                Patterns_Normal_Breakpoints.append([])
     ## HS Bottoms (Inverse Form)
    
    return Patterns
