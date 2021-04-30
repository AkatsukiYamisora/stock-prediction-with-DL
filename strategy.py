# -*- coding: utf-8 -*-
"""
@version: 3.8.3
@time: 21/4/29 9:30
@author: Yamisora
@file: strategy.py
"""
import pandas as pd


class Strategy:
    def __init__(self, data_days=30, index=0):
        """
        index: 0:上证 50  1:沪深 300  2:中证 500
        """
        # 策略所需数据天数
        self.data_days = data_days
        # index选择指数组合
        self.index_name = 'sz50', 'hs300', 'zz500'
        self.indexes = 'sh.000016', 'sh.000300', 'sh.000905'
        # 不建议修改路径
        self.base_data_path = './data/'
        self.data_path = './data/stocks/'
        # 指数组合内股票名称,代码数据
        self.stocks = pd.read_csv('{}{}_stocks.csv'.format(self.base_data_path, self.index_name[index]))
        self.stocks_codes = self.stocks['code']
        # 指数日线数据
        self.index = pd.read_csv('{}{}.csv'.format(self.data_path, self.indexes[index]))
        # 交易日str序列
        self.trading_dates = self.index['date']

    def choose(self, today: tuple):
        # 第一次买入策略应大于策略所需数据天数
        if today[1] < self.data_days:
            return pd.Series(None)
