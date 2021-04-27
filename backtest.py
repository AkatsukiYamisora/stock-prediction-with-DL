# -*- coding: utf-8 -*-
"""
@version: 3.8.3
@time: 21/4/26 10:52
@author: Yamisora
@file: backtest.py
"""
import pandas as pd


class Backtest:
    def __init__(self, start_date='2014-01-01', end_date='2020-12-31', index=0, start_cash=10000000,
                 fee=0.0003):
        """
        设置起止时间
        index: 0:上证 50  1:沪深 300  2:中证 500
        """
        self.start_date = start_date
        self.end_date = end_date
        self.index_name = 'sz50', 'hs300', 'zz500'
        self.indexes = 'sh.000016', 'sh.000300', 'sh.000905'
        self.data_path = './data/stocks/'
        self.fee = fee  # 万三手续费
        self.stocks = pd.read_csv('./data/{}_stocks.csv'.format(self.index_name[index]))
        self.stocks_codes = self.stocks['code']
        self.index = pd.read_csv('{}{}.csv'.format(self.data_path, self.indexes[index]))
        self.trading_dates = self.index['date']
        self.start_cash = start_cash
        self.cash = start_cash
        self.position = pd.DataFrame(self.stocks_codes)
        self.position['quantity'] = 0
        self.position['buying price'] = 0
        self.position = self.position.set_index('code')

    def stocks_data(self, stock_codes, trading_date: str) -> pd.DataFrame():
        if stock_codes.empty:
            return pd.DataFrame(None)
        n = len(stock_codes)
        stocks = pd.DataFrame()
        for stock_code in stock_codes:
            data = pd.read_csv('{}{}.csv'.format(self.data_path, stock_code), index_col='date')
            if trading_date in data.index:
                stocks = stocks.append(data.loc[trading_date], ignore_index=True)
        stocks = stocks.set_index('code')
        return stocks

    def buy(self, stock_codes, trading_date: str) -> bool:
        """
        全仓买入trading date的所有stock codes股票
        trading date为'yyyy-mm-dd'格式字符串
        """
        stocks = self.stocks_data(stock_codes, trading_date)
        if stocks.empty:
            return False
        n = len(stock_codes)
        single = self.cash // n
        for stock_code in stock_codes:
            open_price = stocks.loc[stock_code]['open']
            self.position.loc[stock_code]['buying price'] = open_price
            quantity = ((single / (1+self.fee)) / open_price) // 100
            self.position.loc[stock_code]['quantity'] = quantity * 100
            self.cash -= open_price * quantity * 100
        return True

    def sell(self, stock_codes, trading_date: str) -> bool:
        """
        空仓trading date的所有stock codes股票
        trading date为'yyyy-mm-dd'格式字符串
        """
        stocks = self.stocks_data(stock_codes, trading_date)
        if stocks.empty:
            return False
        cash = 0
        for stock_code in stock_codes:
            cash += self.position.loc[stock_code]['quantity'] * stocks.loc[stock_code]['open'] * (1-self.fee)
            self.position.loc[stock_code]['quantity'] = 0
            self.position.loc[stock_code]['buying price'] = 0
        self.cash += cash
        return True


if __name__ == '__main__':
    bt = Backtest()
    bt.buy(bt.stocks_codes[:10], bt.trading_dates[0])
    bt.sell(bt.stocks_codes[:10], bt.trading_dates[300])
    print(bt.cash)
    print(str((bt.cash - bt.start_cash)*100/bt.start_cash) + '%')
