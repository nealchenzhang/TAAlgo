import os

import numpy as np
import pandas as pd
import datetime as dt
import time

import asyncio

from contextlib import closing
from tqsdk import TqApi, TqSim, TqAccount, TqBacktest, BacktestFinished, TargetPosTask

from zigzag_patterns import HS
from zigzag_patterns import HS_plotting

ls_symbols = ['KQ.m@SHFE.ni']
# api = TqApi(TqSim(), backtest=TqBacktest(start_dt=dt.date(2018,1,2), end_dt=dt.date(2019,1,10)))
api = TqApi('SIM')

dict_klines = {}
dict_update_kline_chan = {}

for SYMBOL in ls_symbols:
    dict_klines[SYMBOL] = api.get_kline_serial(SYMBOL, duration_seconds=30 * 60)
    dict_update_kline_chan[SYMBOL] = api.register_update_notify(dict_klines[SYMBOL])


async def signal_generator_HS(SYMBOL):
    """该task应用策略在价格触发时开仓，出发平仓条件时平仓"""
    klines = dict_klines[SYMBOL]
    update_kline_chan = dict_update_kline_chan[SYMBOL]

    while True:
        async for _ in update_kline_chan:
            if api.is_changing(klines[-1], 'datetime'):
                k30 = str(dt.datetime.fromtimestamp(klines.datetime[-2] / 1e9) + pd.Timedelta(minutes=29, seconds=59))
                print(SYMBOL, '信号时间', k30)
                str_name = k30.split(' ')[0]
                str_name += k30.split(' ')[1].split(':')[0]
                str_name += k30.split(' ')[1].split(':')[1]
                str_name += k30.split(' ')[1].split(':')[2]

                ys = pd.Series(data=klines.close[-250:-1],
                               index=[str(dt.datetime.fromtimestamp(i / 1e9)) for i in klines.datetime[-250:-1]]
                               )
                dict_Patterns, Peaks, Bottoms = HS(ys, pflag=0, method='RW', w=1, iteration=0)
                HS_plotting(ys, Peaks, Bottoms, dict_Patterns, figname='hanyi//ni'+str_name, savefig=True)

    await update_kline_chan.close()


for SYMBOL in ls_symbols:
    api.create_task(signal_generator_HS(SYMBOL))

# with closing(api):
#     try:
#         while True:
#             api.wait_update()
#     except BacktestFinished:
#         print('----回测结束----')

with closing(api):
    while True:
        api.wait_update()