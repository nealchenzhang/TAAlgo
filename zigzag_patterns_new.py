# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 16:13:07 2019

@author: chen zhang
"""

import datetime as dt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#from matplotlib.dates import DateFormatter, WeekdayLocator, DayLocator, MONDAY,YEARLY
#from mpl_finance import candlestick_ohlc
#from matplotlib.pylab import date2num

import seaborn as sns
sns.set_style('white')

from preprocessing import RW
from preprocessing import TP
#from preprocessing import PIPs

from preprocessing import PB_plotting


def HS_plotting(ys, Peaks, Bottoms, dict_Patterns, figname, savefig=False):

    Patterns_Normal_Points = dict_Patterns['Normal']['Points']
    Patterns_Normal_Breakpoints = dict_Patterns['Normal']['Breakpoints']
    Patterns_Normal_Necklines = dict_Patterns['Normal']['Necklines']
    Patterns_Normal_Numberofnormals = dict_Patterns['Normal']['Number']
    Patterns_Normal_PT = dict_Patterns['Normal']['PriceTarget']

    Patterns_Inverse_Points = dict_Patterns['Inverse']['Points']
    Patterns_Inverse_Breakpoints = dict_Patterns['Inverse']['Breakpoints']
    Patterns_Inverse_Necklines = dict_Patterns['Inverse']['Necklines']
    Patterns_Inverse_Numberofinverses = dict_Patterns['Inverse']['Number']
    Patterns_Inverse_PT = dict_Patterns['Inverse']['PriceTarget']

    j = Patterns_Normal_Numberofnormals
    jj = Patterns_Inverse_Numberofinverses
    
    ls_x = ys.index.tolist()
    num_x = len(ls_x)
    ls_time_ix = np.linspace(0, num_x-1, num_x)
    ls_p = Peaks.index.tolist()
    ls_b = Bottoms.index.tolist()
    ls_peaks_time = [ys.index.get_loc(x) for x in ls_p]
    ls_bottoms_time = [ys.index.get_loc(x) for x in ls_b]

    if j > 0 or jj > 0:
        fig, ax = plt.subplots(1,1, figsize=(12, 8))
        num_x = len(ls_x)
        ls_time_ix = np.linspace(0, num_x-1, num_x)

        ls_peaks_time = [ys.index.get_loc(x) for x in ls_p]
        ls_bottoms_time = [ys.index.get_loc(x) for x in ls_b]

        ax.plot(ls_time_ix, ys.values)
        ax.scatter(x=ls_peaks_time, y=Peaks.values, marker='o', color='r', alpha=0.5)
        ax.scatter(x=ls_bottoms_time, y=Bottoms.values, marker='o', color='g', alpha=0.5)

        if j > 0:
            clr = 'darkred'
            for i in range(0, Patterns_Normal_Numberofnormals):
                ls_xline = [ls_x.index(ix) for ix in Patterns_Normal_Points[i].index.tolist()]

                ax.scatter(x=ls_xline, y=Patterns_Normal_Points[i]['Price'].values, color=clr)
                ax.scatter(x=ls_x.index(Patterns_Normal_Breakpoints[i][0]), y=Patterns_Normal_Breakpoints[i][1],
                           marker='x', color='black')
                # Plotting Neckline
                ls_neckline_x = np.linspace(start=ls_xline[0]-15, stop=ls_x.index(Patterns_Normal_Breakpoints[i][0])+15)
                ls_neckline_y = ls_neckline_x*Patterns_Normal_Necklines[i][1] + Patterns_Normal_Necklines[i][0]
                ax.plot(ls_neckline_x, ls_neckline_y, linestyle='-.', color='green')
                ax.scatter(x=ls_x.index(Patterns_Normal_Breakpoints[i][0]), y=Patterns_Normal_PT[i], marker='o', color='yellow')
        if jj > 0:
            clr = 'darkgray'
            for i in range(0, Patterns_Inverse_Numberofinverses):
                ls_xline = [ls_x.index(ix) for ix in Patterns_Inverse_Points[i].index.tolist()]
                ax.scatter(x=ls_xline, y=Patterns_Inverse_Points[i]['Price'].values, color=clr)
                ax.scatter(x=ls_x.index(Patterns_Inverse_Breakpoints[i][0]), y=Patterns_Inverse_Breakpoints[i][1],
                           marker='v', color='black')
                # Plotting Neckline
                ls_neckline_x = np.linspace(start=ls_xline[0]-15, stop=ls_x.index(Patterns_Inverse_Breakpoints[i][0])+15)
                ls_neckline_y = ls_neckline_x*Patterns_Inverse_Necklines[i][1] + Patterns_Inverse_Necklines[i][0]
                ax.plot(ls_neckline_x, ls_neckline_y, linestyle='-.', color='red')
                ax.scatter(x=ls_x.index(Patterns_Inverse_Breakpoints[i][0]), y=Patterns_Inverse_PT[i], marker='o',
                           color='yellow')

        plt.title(figname)
        new_xticklabels = [ls_x[np.int(i)] for i in list(ax.get_xticks()) if i in ls_time_ix]
        new_xticklabels = [ls_x[0]] + new_xticklabels
        new_xticklabels.append(ls_x[-1])
        ax.set_xticklabels(new_xticklabels)
        for tick in ax.get_xticklabels():
            tick.set_rotation(15)

        if savefig:
            plt.savefig('.//Data//figs//'+ figname + '.png')

        plt.show()


def line_inter(A, B):
    """
    A: ls of coordinates [[x1,y1],[x2,y2]]
    B: ls
    """
    S1 = (A[1][1] - A[0][1]) / (A[1][0] - A[0][0]) # slope1
    S2 = (B[1][1] - B[0][1]) / (B[1][0] - B[0][0]) # slope2
    C1 = A[0][1] - S1*A[0][0]                      # constant1
    C2 = B[0][1] - S2*B[0][0]                      # constant2
    tstar = (C2 - C1) / (S1 - S2)
    ystar = S1 * tstar + C1
    LU = [S1, C1]
    LD = [S2, C2]
    
    return tstar, ystar, LU, LD


def HS(ys, pflag, method='RW', **kwargs):
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
    :param pflag: plot a graph if 1
    :param method:
        1) RW: rolling window method to find turning points
                kwargs['w']
        2) TP: turning points with filter methods (iter=2)
                kwargs['iteration']
    :param kwargs: {'w': width of the rolling window (total 2w+1)
                    'iteration': iteration number for TP method,
                    'savefig': True,
                    'figname': str(figname)}
    :return:
    """
    l = len(ys)
    if method == 'RW':
        Peaks, Bottoms = RW(ys, w=kwargs['w'], iteration=kwargs['iteration'])
    elif method == 'TP':
        Peaks, Bottoms = TP(ys, iteration=kwargs['iteration'])

    if l > 250:
        pflag = 0

    if pflag == 1:
        PB_plotting(ys, Peaks, Bottoms)
    
    ls_x = ys.index.tolist()
    ls_p = Peaks.index.tolist()
    ls_b = Bottoms.index.tolist()

    P_idx = pd.Series(index=ls_p, data=[1]*len(ls_p))
    B_idx = pd.Series(index=ls_b, data=[2]*len(ls_b))
    PB_idx = P_idx.append(B_idx)
    PB_idx.sort_index(inplace=True)
    m = len(PB_idx.index)
    
    Pot_Normal = [1,2,1,2,1]
    Pot_Inverse = [2,1,2,1,2]
    Pot_Index = [0] * m
    
    for i in range(0, m-4):
        if PB_idx.iloc[i:i+5].values.tolist() == Pot_Normal:
            Pot_Index[i+4] = 1
        elif PB_idx.iloc[i:i+5].values.tolist() == Pot_Inverse:
            Pot_Index[i+4] = 2
    
    PNidx = [i for i, x in enumerate(Pot_Index) if x == 1]
    PIidx = [i for i, x in enumerate(Pot_Index) if x == 2]
    
    ## HS Tops (Normal Form)
    # Definition of outputs
    Patterns_Normal_Points = []
    Patterns_Normal_Necklines = []
    Patterns_Normal_Breakpoints = []
    Patterns_Normal_Widths = []
    Patterns_Normal_Heights = []
    Patterns_Normal_TL = []
    Patterns_Normal_PT = []
    
    mn = len(PNidx)
    Pot_Normalcases_Percases = [0] * mn
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
                Pot_Normalcases_Percases[i] = PerCase
    else:
        Pot_Normalcases_Idx = 0

    mnn = np.sum(Pot_Normalcases_Idx)
    if mnn == 0:
        pass
    else:
        Pot_Normalcases_Idx2 = [i for i, x in enumerate(Pot_Normalcases_Idx) if x == 1]
    j = 0
    if mnn!=0:
        for i in range(0, mnn):
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
                Patterns_Normal_Breakpoints.append([ls_x[Neckline_Breakpoint], ys.loc[ls_x[Neckline_Breakpoint]]])
                Patterns_Normal_Widths.append(ls_x.index(NPerCase.index.tolist()[3])-ls_x.index(NPerCase.index.tolist()[1]))
                Patterns_Normal_Heights.append(NPerCase.iloc[2]['Price']-(Neckline_Beta*ls_x.index(NPerCase.index.tolist()[2])+Neckline_Alpha))
                Patterns_Normal_TL.append(ls_x.index(Patterns_Normal_Breakpoints[-1][0])+Patterns_Normal_Widths[-1])
                tstar, ystar, LU, LD = line_inter([[ls_x.index(NPerCase.index.tolist()[1]), NPerCase.iloc[1]['Price']],
                                                     [ls_x.index(NPerCase.index.tolist()[3]), NPerCase.iloc[3]['Price']]],
                                                   [[ls_x.index(Patterns_Normal_Breakpoints[-1][0])-1,
                                                        ys.iloc[ls_x.index(Patterns_Normal_Breakpoints[-1][0])-1]],
                                                    [ls_x.index(Patterns_Normal_Breakpoints[-1][0]), Patterns_Normal_Breakpoints[-1][1]]])
                Patterns_Normal_PT.append(ystar - Patterns_Normal_Heights[-1])
    Patterns_Normal_Numberofnormals = j

    ## HS Bottoms (Inverse Form)
    # Definition of outputs
    Patterns_Inverse_Points = []
    Patterns_Inverse_Necklines = []
    Patterns_Inverse_Breakpoints = []
    Patterns_Inverse_Widths = []
    Patterns_Inverse_Heights = []
    Patterns_Inverse_TL = []
    Patterns_Inverse_PT = []

    mi = len(PIidx)
    Pot_Inversecases_Percases = [0] * mi  # bug??
    if mi != 0:
        Pot_Inversecases_Idx = [0] * mi
        for i in range(0, mi):
            # Conditions check
            PerCase = pd.DataFrame(columns=['P/B', 'Price'])
            PerCase.loc[:, 'P/B'] = PB_idx.iloc[PIidx[i] - 4: PIidx[i] + 1]
            PerCase.loc[:, 'Price'] = ys.loc[PerCase.index.tolist()].values
            # Condition 1 3 4
            if ((PerCase.iloc[2]['Price'] < np.min([PerCase.iloc[0]['Price'],
                                                    PerCase.iloc[4]['Price']])) &
                    (PerCase.iloc[0]['Price'] <= 0.5 * np.sum([PerCase.iloc[3]['Price'],
                                                               PerCase.iloc[4]['Price']])) &
                    (PerCase.iloc[4]['Price'] <= 0.5 * np.sum([PerCase.iloc[0]['Price'],
                                                               PerCase.iloc[1]['Price']])) &
                    ((ls_x.index(PerCase.index.tolist()[2]) - ls_x.index(PerCase.index.tolist()[0])) <
                     2.5 * (ls_x.index(PerCase.index.tolist()[4]) - ls_x.index(PerCase.index.tolist()[2]))) &
                    ((ls_x.index(PerCase.index.tolist()[4]) - ls_x.index(PerCase.index.tolist()[2])) <
                     2.5 * (ls_x.index(PerCase.index.tolist()[2]) - ls_x.index(PerCase.index.tolist()[0])))):
                Pot_Inversecases_Idx[i] = 1
                Pot_Inversecases_Percases[i] = PerCase  # bug??
    else:
        Pot_Inversecases_Idx = 0

    mii = np.sum(Pot_Inversecases_Idx)
    if mii == 0:
        pass
    else:
        Pot_Inversecases_Idx2 = [i for i, x in enumerate(Pot_Inversecases_Idx) if x == 1]
    jj = 0
    if mii != 0:
        for i in range(0, mii):
            NPerCase = Pot_Inversecases_Percases[Pot_Inversecases_Idx2[i]]
            Timelimit = ls_x.index(NPerCase.index.tolist()[4]) + ls_x.index(NPerCase.index.tolist()[4]) - \
                        ls_x.index(NPerCase.index.tolist()[0])
            Neckline_Beta = (NPerCase.iloc[3]['Price'] - NPerCase.iloc[1]['Price']) / \
                            (ls_x.index(NPerCase.index.tolist()[3]) - ls_x.index(NPerCase.index.tolist()[1]))
            Neckline_Alpha = NPerCase.iloc[1]['Price'] - Neckline_Beta * ls_x.index(NPerCase.index.tolist()[1])
            if Timelimit <= l:
                Neckline_OU = list(range(ls_x.index(NPerCase.index.tolist()[4]) + 1, Timelimit))
            else:
                Neckline_OU = list(range(ls_x.index(NPerCase.index.tolist()[4]) + 1, l))

            Neckline_OU = pd.DataFrame(index=[ls_x[i] for i in Neckline_OU], data=ys.iloc[Neckline_OU])
            Neckline_OU.columns = ['Price']
            Neckline_OU.loc[:, 'Line'] = [ls_x.index(i) for i in Neckline_OU.index.tolist()]
            Neckline_OU.loc[:, 'Line'] = Neckline_OU.loc[:, 'Line'] * Neckline_Beta + Neckline_Alpha
            Neckline_OU.loc[:, 'Diff'] = Neckline_OU.loc[:, 'Price'] - Neckline_OU.loc[:, 'Line']
            # TODO: optimize codes
            if len((Neckline_OU.where(Neckline_OU.loc[:, 'Diff'] > 0).dropna()).index.tolist()) != 0:
                Neckline_Breakpoint = ls_x.index(
                    (Neckline_OU.where(Neckline_OU.loc[:, 'Diff'] > 0).dropna()).index.tolist()[0])
                jj += 1
                Patterns_Inverse_Points.append(NPerCase)
                Patterns_Inverse_Necklines.append([Neckline_Alpha, Neckline_Beta])
                Patterns_Inverse_Breakpoints.append([ls_x[Neckline_Breakpoint], ys.loc[ls_x[Neckline_Breakpoint]]])
                Patterns_Inverse_Widths.append(
                    ls_x.index(NPerCase.index.tolist()[3]) - ls_x.index(NPerCase.index.tolist()[1]))
                Patterns_Inverse_Heights.append(np.abs(NPerCase.iloc[2]['Price'] - (
                            Neckline_Beta * ls_x.index(NPerCase.index.tolist()[2]) + Neckline_Alpha)))
                Patterns_Inverse_TL.append(ls_x.index(Patterns_Inverse_Breakpoints[-1][0]) + Patterns_Inverse_Widths[-1])
                tstar, ystar, LU, LD = line_inter([[ls_x.index(NPerCase.index.tolist()[1]), NPerCase.iloc[1]['Price']],
                                                    [ls_x.index(NPerCase.index.tolist()[3]),
                                                     NPerCase.iloc[3]['Price']]],
                                                   [[ls_x.index(Patterns_Inverse_Breakpoints[-1][0]) - 1,
                                                     ys.iloc[ls_x.index(Patterns_Inverse_Breakpoints[-1][0]) - 1]],
                                                    [ls_x.index(Patterns_Inverse_Breakpoints[-1][0]), Patterns_Inverse_Breakpoints[-1][1]]])
                Patterns_Inverse_PT.append(ystar + Patterns_Inverse_Heights[-1])
    Patterns_Inverse_Numberofinverses = jj
    
    dict_Patterns = {
        'Normal': {
            'Points': Patterns_Normal_Points,
            'Breakpoints': Patterns_Normal_Breakpoints,
            'Necklines': Patterns_Normal_Necklines,
            'Number': Patterns_Normal_Numberofnormals,
            'PriceTarget': Patterns_Normal_PT
        },
        'Inverse': {
            'Points': Patterns_Inverse_Points,
            'Breakpoints': Patterns_Inverse_Breakpoints,
            'Necklines': Patterns_Inverse_Necklines,
            'Number': Patterns_Inverse_Numberofinverses,
            'PriceTarget': Patterns_Inverse_PT
        }
    }

    return dict_Patterns, Peaks, Bottoms


def TTTB(ys, pflag, method='RW', **kwargs):
    """
    The TTTB(*) identifies Triple Tops/ Triple Bottoms Pattern on a price series by
    adopting the following conditions:

        1) Trend preexistence:
                    TT: uptrend if P1 > P0 & T1 > T0
                    TB: downtrend if P1 < P0 & T1 < T0
        2) Balance:
                    TT: (max(P1, P3) - min(P1, P3)) / min(P1, P3) <= 0.04 & P2 <= max(P1, P3)
                    TB: (max(T1, T3) - min(T1, T3)) / min(T1, T3) <= 0.04 & T2 >= min(T1, T3)
        3) Intervening locals
                    TT: T1 <= T2 <= 1.04T1
                    TB: P2 <= P1 <= 1.04P2
        4) Symmetry:
                    TT: tP2 - tP1 < 2.5(tP3 - tP2) & tP3 - tP2 < 2.5(tP2 - tP1)
                    TB: tT2 - tT1 < 2.5(tT3 - tT2) & tT3 - tT2 < 2.5(tT2 - tT1)
        5) Penetration:
                    TT: tB < tP3 + (tP3 - tP1)
                    TB: tB < tT3 + (tT3 - tT1)

    :param ys: column vector of price series with time index
    :param pflag: plot a graph if 1
    :param method:
        1) RW: rolling window method to find turning points
                kwargs['w']
        2) TP: turning points with filter methods (iter=2)
                kwargs['iteration']
    :param kwargs: {'w': width of the rolling window (total 2w+1)
                    'iteration': iteration number for TP method,
                    'savefig': True,
                    'figname': str(figname)}
    :return:
    """
    l = len(ys)
    if method == 'RW':
        Peaks, Bottoms = RW(ys, w=kwargs['w'], iteration=kwargs['iteration'])
    elif method == 'TP':
        Peaks, Bottoms = TP(ys, iteration=kwargs['iteration'])

    Peaks, Bottoms = RW(ys, w=1, iteration=0)
    pflag = 1
    if l > 250:
        pflag = 0

    if pflag == 1:
        PB_plotting(ys, Peaks, Bottoms)

    ls_x = ys.index.tolist()
    ls_p = Peaks.index.tolist()
    ls_b = Bottoms.index.tolist()

    P_idx = pd.Series(index=ls_p, data=[1] * len(ls_p))
    B_idx = pd.Series(index=ls_b, data=[2] * len(ls_b))
    PB_idx = P_idx.append(B_idx)
    PB_idx.sort_index(inplace=True)
    m = len(PB_idx.index)

    Pot_Normal = [1, 2, 1, 2, 1]
    Pot_Inverse = [2, 1, 2, 1, 2]
    Pot_Index = [0] * m

    for i in range(0, m - 4):
        if PB_idx.iloc[i:i + 5].values.tolist() == Pot_Normal:
            Pot_Index[i + 4] = 1
        elif PB_idx.iloc[i:i + 5].values.tolist() == Pot_Inverse:
            Pot_Index[i + 4] = 2

    PNidx = [i for i, x in enumerate(Pot_Index) if x == 1]
    PIidx = [i for i, x in enumerate(Pot_Index) if x == 2]

    # TODO
    ## Triple Tops (Normal Form)
    # Definition of outputs
    Patterns_Normal_Points = []
    Patterns_Normal_Necklines = []
    Patterns_Normal_Breakpoints = []
    Patterns_Normal_Widths = []
    Patterns_Normal_Heights = []
    Patterns_Normal_TL = []
    Patterns_Normal_PT = []

    mn = len(PNidx)
    Pot_Normalcases_Percases = [0] * mn
    if mn != 0:
        Pot_Normalcases_Idx = [0] * mn
        for i in range(0, mn):
            # Conditions check
            PerCase = pd.DataFrame(columns=['P/B', 'Price'])
            PerCase.loc[:, 'P/B'] = PB_idx.iloc[PNidx[i] - 4: PNidx[i] + 1]
            PerCase.loc[:, 'Price'] = ys.loc[PerCase.index.tolist()].values
            # Condition 2 3 4
            ls_tmp = [PerCase.iloc[0]['Price'], PerCase.iloc[4]['Price']]
            if (PerCase.iloc[2]['Price'] <= np.max([PerCase.iloc[0]['Price'], PerCase.iloc[4]['Price']])) & \
                    ((np.max(ls_tmp) - np.min(ls_tmp))/np.min(ls_tmp) <= 0.04) & \
                    (PerCase.iloc[1]['Price'] <= PerCase.iloc[3]['Price'] <= 1.4*(PerCase.iloc[1]['Price'])) & \
                    ((ls_x.index(PerCase.index.tolist()[2]) - ls_x.index(PerCase.index.tolist()[0])) <
                     2.5 * (ls_x.index(PerCase.index.tolist()[4]) - ls_x.index(PerCase.index.tolist()[2]))) & \
                    ((ls_x.index(PerCase.index.tolist()[4]) - ls_x.index(PerCase.index.tolist()[2])) <
                     2.5 * (ls_x.index(PerCase.index.tolist()[2]) - ls_x.index(PerCase.index.tolist()[0]))):
                Pot_Normalcases_Idx[i] = 1
                Pot_Normalcases_Percases[i] = PerCase
    else:
        Pot_Normalcases_Idx = 0

    mnn = np.sum(Pot_Normalcases_Idx)
    if mnn == 0:
        pass
    else:
        Pot_Normalcases_Idx2 = [i for i, x in enumerate(Pot_Normalcases_Idx) if x == 1]
    j = 0
    if mnn != 0:
        for i in range(0, mnn):
            NPerCase = Pot_Normalcases_Percases[Pot_Normalcases_Idx2[i]]
            Timelimit = ls_x.index(NPerCase.index.tolist()[4]) + ls_x.index(NPerCase.index.tolist()[4]) - \
                        ls_x.index(NPerCase.index.tolist()[0])
            # TODO
            # Neckline_Beta = (NPerCase.iloc[3]['Price'] - NPerCase.iloc[1]['Price']) / \
            #                 (ls_x.index(NPerCase.index.tolist()[3]) - ls_x.index(NPerCase.index.tolist()[1]))
            # Neckline_Alpha = NPerCase.iloc[1]['Price'] - Neckline_Beta * ls_x.index(NPerCase.index.tolist()[1])
            if Timelimit <= l:
                Neckline_OU = list(range(ls_x.index(NPerCase.index.tolist()[4]) + 1, Timelimit))
            else:
                Neckline_OU = list(range(ls_x.index(NPerCase.index.tolist()[4]) + 1, l))

            Neckline_OU = pd.DataFrame(index=[ls_x[i] for i in Neckline_OU], data=ys.iloc[Neckline_OU])
            Neckline_OU.columns = ['Price']
            Neckline_OU.loc[:, 'Line'] = [ls_x.index(i) for i in Neckline_OU.index.tolist()]
            Neckline_OU.loc[:, 'Line'] = Neckline_OU.loc[:, 'Line'] * Neckline_Beta + Neckline_Alpha
            Neckline_OU.loc[:, 'Diff'] = Neckline_OU.loc[:, 'Price'] - Neckline_OU.loc[:, 'Line']
            # TODO: optimize codes
            if len((Neckline_OU.where(Neckline_OU.loc[:, 'Diff'] < 0).dropna()).index.tolist()) != 0:
                Neckline_Breakpoint = ls_x.index(
                    (Neckline_OU.where(Neckline_OU.loc[:, 'Diff'] < 0).dropna()).index.tolist()[0])
                j += 1
                Patterns_Normal_Points.append(NPerCase)
                Patterns_Normal_Necklines.append([Neckline_Alpha, Neckline_Beta])
                Patterns_Normal_Breakpoints.append([ls_x[Neckline_Breakpoint], ys.loc[ls_x[Neckline_Breakpoint]]])
                Patterns_Normal_Widths.append(
                    ls_x.index(NPerCase.index.tolist()[3]) - ls_x.index(NPerCase.index.tolist()[1]))
                Patterns_Normal_Heights.append(NPerCase.iloc[2]['Price'] - (
                            Neckline_Beta * ls_x.index(NPerCase.index.tolist()[2]) + Neckline_Alpha))
                Patterns_Normal_TL.append(ls_x.index(Patterns_Normal_Breakpoints[-1][0]) + Patterns_Normal_Widths[-1])
                tstar, ystar, LU, LD = line_inter([[ls_x.index(NPerCase.index.tolist()[1]), NPerCase.iloc[1]['Price']],
                                                   [ls_x.index(NPerCase.index.tolist()[3]), NPerCase.iloc[3]['Price']]],
                                                  [[ls_x.index(Patterns_Normal_Breakpoints[-1][0]) - 1,
                                                    ys.iloc[ls_x.index(Patterns_Normal_Breakpoints[-1][0]) - 1]],
                                                   [ls_x.index(Patterns_Normal_Breakpoints[-1][0]),
                                                    Patterns_Normal_Breakpoints[-1][1]]])
                Patterns_Normal_PT.append(ystar - Patterns_Normal_Heights[-1])
    Patterns_Normal_Numberofnormals = j

    ## Triple Bottoms (Inverse Form)
    # Definition of outputs
    Patterns_Inverse_Points = []
    Patterns_Inverse_Necklines = []
    Patterns_Inverse_Breakpoints = []
    Patterns_Inverse_Widths = []
    Patterns_Inverse_Heights = []
    Patterns_Inverse_TL = []
    Patterns_Inverse_PT = []

    mi = len(PIidx)
    Pot_Inversecases_Percases = [0] * mi  # bug??
    if mi != 0:
        Pot_Inversecases_Idx = [0] * mi
        for i in range(0, mi):
            # Conditions check
            PerCase = pd.DataFrame(columns=['P/B', 'Price'])
            PerCase.loc[:, 'P/B'] = PB_idx.iloc[PIidx[i] - 4: PIidx[i] + 1]
            PerCase.loc[:, 'Price'] = ys.loc[PerCase.index.tolist()].values
            # Condition 1 3 4
            if ((PerCase.iloc[2]['Price'] < np.min([PerCase.iloc[0]['Price'],
                                                    PerCase.iloc[4]['Price']])) &
                    (PerCase.iloc[0]['Price'] <= 0.5 * np.sum([PerCase.iloc[3]['Price'],
                                                               PerCase.iloc[4]['Price']])) &
                    (PerCase.iloc[4]['Price'] <= 0.5 * np.sum([PerCase.iloc[0]['Price'],
                                                               PerCase.iloc[1]['Price']])) &
                    ((ls_x.index(PerCase.index.tolist()[2]) - ls_x.index(PerCase.index.tolist()[0])) <
                     2.5 * (ls_x.index(PerCase.index.tolist()[4]) - ls_x.index(PerCase.index.tolist()[2]))) &
                    ((ls_x.index(PerCase.index.tolist()[4]) - ls_x.index(PerCase.index.tolist()[2])) <
                     2.5 * (ls_x.index(PerCase.index.tolist()[2]) - ls_x.index(PerCase.index.tolist()[0])))):
                Pot_Inversecases_Idx[i] = 1
                Pot_Inversecases_Percases[i] = PerCase  # bug??
    else:
        Pot_Inversecases_Idx = 0

    mii = np.sum(Pot_Inversecases_Idx)
    if mii == 0:
        pass
    else:
        Pot_Inversecases_Idx2 = [i for i, x in enumerate(Pot_Inversecases_Idx) if x == 1]
    jj = 0
    if mii != 0:
        for i in range(0, mii):
            NPerCase = Pot_Inversecases_Percases[Pot_Inversecases_Idx2[i]]
            Timelimit = ls_x.index(NPerCase.index.tolist()[4]) + ls_x.index(NPerCase.index.tolist()[4]) - \
                        ls_x.index(NPerCase.index.tolist()[0])
            Neckline_Beta = (NPerCase.iloc[3]['Price'] - NPerCase.iloc[1]['Price']) / \
                            (ls_x.index(NPerCase.index.tolist()[3]) - ls_x.index(NPerCase.index.tolist()[1]))
            Neckline_Alpha = NPerCase.iloc[1]['Price'] - Neckline_Beta * ls_x.index(NPerCase.index.tolist()[1])
            if Timelimit <= l:
                Neckline_OU = list(range(ls_x.index(NPerCase.index.tolist()[4]) + 1, Timelimit))
            else:
                Neckline_OU = list(range(ls_x.index(NPerCase.index.tolist()[4]) + 1, l))

            Neckline_OU = pd.DataFrame(index=[ls_x[i] for i in Neckline_OU], data=ys.iloc[Neckline_OU])
            Neckline_OU.columns = ['Price']
            Neckline_OU.loc[:, 'Line'] = [ls_x.index(i) for i in Neckline_OU.index.tolist()]
            Neckline_OU.loc[:, 'Line'] = Neckline_OU.loc[:, 'Line'] * Neckline_Beta + Neckline_Alpha
            Neckline_OU.loc[:, 'Diff'] = Neckline_OU.loc[:, 'Price'] - Neckline_OU.loc[:, 'Line']
            # TODO: optimize codes
            if len((Neckline_OU.where(Neckline_OU.loc[:, 'Diff'] > 0).dropna()).index.tolist()) != 0:
                Neckline_Breakpoint = ls_x.index(
                    (Neckline_OU.where(Neckline_OU.loc[:, 'Diff'] > 0).dropna()).index.tolist()[0])
                jj += 1
                Patterns_Inverse_Points.append(NPerCase)
                Patterns_Inverse_Necklines.append([Neckline_Alpha, Neckline_Beta])
                Patterns_Inverse_Breakpoints.append([ls_x[Neckline_Breakpoint], ys.loc[ls_x[Neckline_Breakpoint]]])
                Patterns_Inverse_Widths.append(
                    ls_x.index(NPerCase.index.tolist()[3]) - ls_x.index(NPerCase.index.tolist()[1]))
                Patterns_Inverse_Heights.append(np.abs(NPerCase.iloc[2]['Price'] - (
                        Neckline_Beta * ls_x.index(NPerCase.index.tolist()[2]) + Neckline_Alpha)))
                Patterns_Inverse_TL.append(
                    ls_x.index(Patterns_Inverse_Breakpoints[-1][0]) + Patterns_Inverse_Widths[-1])
                tstar, ystar, LU, LD = line_inter([[ls_x.index(NPerCase.index.tolist()[1]), NPerCase.iloc[1]['Price']],
                                                   [ls_x.index(NPerCase.index.tolist()[3]),
                                                    NPerCase.iloc[3]['Price']]],
                                                  [[ls_x.index(Patterns_Inverse_Breakpoints[-1][0]) - 1,
                                                    ys.iloc[ls_x.index(Patterns_Inverse_Breakpoints[-1][0]) - 1]],
                                                   [ls_x.index(Patterns_Inverse_Breakpoints[-1][0]),
                                                    Patterns_Inverse_Breakpoints[-1][1]]])
                Patterns_Inverse_PT.append(ystar + Patterns_Inverse_Heights[-1])
    Patterns_Inverse_Numberofinverses = jj

    dict_Patterns = {
        'Normal': {
            'Points': Patterns_Normal_Points,
            'Breakpoints': Patterns_Normal_Breakpoints,
            'Necklines': Patterns_Normal_Necklines,
            'Number': Patterns_Normal_Numberofnormals,
            'PriceTarget': Patterns_Normal_PT
        },
        'Inverse': {
            'Points': Patterns_Inverse_Points,
            'Breakpoints': Patterns_Inverse_Breakpoints,
            'Necklines': Patterns_Inverse_Necklines,
            'Number': Patterns_Inverse_Numberofinverses,
            'PriceTarget': Patterns_Inverse_PT
        }
    }

    return dict_Patterns, Peaks, Bottoms

if __name__ == '__main__':
    
    df_ys = pd.read_csv('./Data/m_i_1d.csv')
    df_ys.datetime = df_ys.datetime.apply(pd.to_datetime)
    df_ys.datetime = df_ys.datetime.apply(lambda x: str(x)) 
    df_ys.set_index('datetime',inplace=True)
    ls_cols = df_ys.columns.tolist()
    str_Close = [i for i in ls_cols if i[-6:] == '.close'][0]
    ys = df_ys.loc[:, str_Close]
    
    dict_Patterns, Peaks, Bottoms = HS(ys, pflag=1, method='RW', w=1, iteration=0)
    HS_plotting(ys, Peaks, Bottoms, dict_Patterns, figname='m_i_1d', savefig=True)

    dict_Patterns, Peaks, Bottoms = TTTB(ys, pflag=1, method='RW', w=1, iteration=0)
