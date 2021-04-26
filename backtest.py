# -*- coding: utf-8 -*-
"""
@version: 3.8.3
@time: 21/4/26 10:52
@author: Yamisora
@file: backtest.py
"""
import pandas as pd


class Backtest:
    def __init__(self, start_date='2014-01-01', end_date='2020-12-31', index=0, start_cash=100000):
        """
        设置起止时间
        index: 0:上证 50  1:沪深 300  2:中证 500
        """
        self.start_date = start_date
        self.end_date = end_date
        self.index_name = 'sz50', 'hs300', 'zz500'
        self.indexes = 'sh.000016', 'sh.000300', 'sh.000905'
        self.data_path = './data/stocks/'
        self.stocks = pd.read_csv('./data/{}_stocks.csv'.format(self.index_name[index]))
        self.stocks_codes = self.stocks['code']
        self.index = pd.read_csv('{}{}.csv'.format(self.data_path, self.indexes[index]))
        self.trading_dates = self.index['date']
        self.cash = start_cash

    def buy(self, stock_codes, trading_date: str) -> bool:
        """
        买入trading date的所有stock codes股票
        trading date为'yyyy-mm-dd'格式字符串
        """
        if stock_codes.empty:
            return False
        stocks = pd.DataFrame()
        for stock_code in stock_codes:
            data = pd.read_csv('{}{}.csv'.format(self.data_path, stock_code), index_col='date')
            if trading_date in data.index:
                stocks = stocks.append(data.loc[trading_date], ignore_index=True)
        stocks = stocks.set_index('code')
        print(stocks)
        return True

    def sell(self, stock_codes, trading_date: str) -> bool:
        """
        买入trading date的所有stock codes股票
        trading date为'yyyy-mm-dd'格式字符串
        """
        if stock_codes.empty:
            return False
        stocks = pd.DataFrame()
        for stock_code in stock_codes:
            data = pd.read_csv('{}{}.csv'.format(self.data_path, stock_code), index_col='date')
            if trading_date in data.index:
                stocks = stocks.append(data.loc[trading_date], ignore_index=True)
        stocks = stocks.set_index('code')
        print(stocks)
        return True

    def trade(self, trading_date: str):
        """
        一次交易日
        date为'yyyy-mm-dd'格式字符串
        """
        pass


if __name__ == '__main__':
    bt = Backtest()
    bt.trade(bt.trading_dates[0])
    b = bt.buy(bt.stocks_codes[:3], bt.trading_dates[0])
    print(b)
