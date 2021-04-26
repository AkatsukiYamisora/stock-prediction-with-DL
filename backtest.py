# -*- coding: utf-8 -*-
"""
@version: 3.8.3
@time: 21/4/26 10:52
@author: Yamisora
@file: backtest.py
"""
import pandas as pd

# 起止日期
date = '2014-01-01', '2020-12-31'


class Backtest:
    def __init__(self, start_date=date[0], end_date=date[1], index=0):
        """
        设置起止时间
        index: 0:上证 50  1:沪深 300  2:中证 500
        """
        self.start_date = start_date
        self.end_date = end_date
        self.stocks = 'sz50', 'hs300', 'zz500'
        self.indexes = 'sh.000016', 'sh.000300', 'sh.000905'
        self.stocks_codes = pd.read_csv('./data/{}_stocks.csv'.format(self.stocks[index]))
        self.index = pd.read_csv('./data/stocks/{}.csv'.format(self.indexes[index]))
        self.trading_dates = self.index['date']

    def trade(self, trading_date: str):
        """
        一次交易日
        date为'yyyy-mm-dd'格式字符串
        """
        for stock_code in self.stocks_codes['code']:
            data = pd.read_csv('./data/stocks/{}.csv'.format(stock_code), index_col='date')
            print(data)
            print(data.index)
            if trading_date in data.index:
                print(data.loc[trading_date])


if __name__ == '__main__':
    bt = Backtest()
    bt.trade(bt.trading_dates[0])
