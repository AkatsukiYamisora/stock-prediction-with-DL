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
        # 存储路径
        self.base_data_path = './data/'
        self.data_path = './data/stocks/'
        # 指数组合内股票名称,代码数据
        self.stocks = pd.read_csv('{}{}_stocks.csv'.format(self.base_data_path, self.index_name[index]))
        self.stocks_codes = self.stocks['code']
        # 指数日线数据
        self.index = pd.read_csv('{}{}.csv'.format(self.data_path, self.indexes[index]))
        # 交易日str序列
        self.trading_dates = self.index['date']

    def choose_by_bm(self, today: tuple, number: int):
        """
        选择最近data_days中平均账面市值比(BM)最高的number只股票
        """
        # 第一次买入策略应大于策略所需数据天数
        if today[1] < self.data_days:
            return pd.Series(None)
        # 建立用于计算平均BM的DF
        stocks_data = pd.DataFrame(self.stocks_codes)
        stocks_data['aver_BM'] = 0
        stocks_data = stocks_data.set_index('code')
        # 到交易日前一日为止共data_days日期序号
        days = range(today[1] - self.data_days, today[1])
        for stock_code in self.stocks_codes:
            sum_BM = 0
            valid_days = self.data_days
            stock_data = pd.read_csv('{}{}.csv'.format(self.data_path, stock_code), index_col='date')
            for day in days:
                if self.trading_dates[day] in stock_data.index:
                    # 加入每日市净率倒数
                    pb = stock_data.loc[self.trading_dates[day], 'pbMRQ']
                    if pb != 0:
                        sum_BM += 1.0 / pb
                    else:
                        sum_BM += 0
                else:
                    valid_days -= 1
            if valid_days != 0:
                aver_BM = sum_BM / valid_days
            else:
                aver_BM = 0
            stocks_data.loc[stock_code, 'aver_BM'] = aver_BM
        # print(stocks_data)
        stocks_data.sort_values(by='aver_BM', ascending=False, inplace=True)
        # print(stocks_data.index)
        return stocks_data.index[0:number]
