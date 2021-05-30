# -*- coding: utf-8 -*-
"""
@version: 3.8.3
@time: 21/4/29 9:30
@author: Yamisora
@file: strategy.py
"""
from prediction import *


class Strategy:
    def __init__(self, data_days=10, index=1):
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
        self.train_data_path = './data/train_data/'
        # 指数组合内股票名称,代码数据
        self.stocks = pd.read_csv('{}{}_stocks.csv'.format(self.base_data_path, self.index_name[index]))
        self.stocks_codes = self.stocks['code']
        # 指数日线数据
        self.index = pd.read_csv('{}{}.csv'.format(self.data_path, self.indexes[index]))
        # 交易日str序列
        self.trading_dates = self.index['date']
        # 训练CNN模型
        self.dataset = StockDataset(index=index, data_days=data_days)
        self.prediction = Prediction(index=index, data_days=data_days, batch_size=50)
        self.prediction.train_cnn(self.dataset)

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

    def choose_by_cnn(self, today: tuple, number: int):
        """
        选择CNN预测未来data_days天涨幅最高的number只股票
        """
        # 第一次买入策略应大于策略所需数据天数
        if today[1] < self.data_days:
            return pd.Series(None)
        # 建立用于计算预测涨跌幅的DF
        stocks_data = pd.DataFrame(self.stocks_codes)
        stocks_data['change'] = 0
        stocks_data = stocks_data.set_index('code')
        # 预测每只股票未来data_days天涨跌幅
        for stock_code in self.stocks_codes:
            change = self.prediction.predict_cnn(stock_code, today)
            if change != 0:
                change = change.item()
            stocks_data.loc[stock_code, 'change'] = change
        # 排序
        stocks_data.sort_values(by='change', ascending=False, inplace=True)
        return stocks_data.index[0:number]

    def choose_by_mf(self, today: tuple, number: int):
        """
        选择最近data_days中动量因子(Momentum Factor)最高的number只股票
        """
        # 第一次买入策略应大于策略所需数据天数
        if today[1] < self.data_days:
            return pd.Series(None)
        # 建立用于计算平均MF的DF
        stocks_data = pd.DataFrame(self.stocks_codes)
        stocks_data['aver_MF'] = 0
        stocks_data = stocks_data.set_index('code')
        # 到交易日前一日为止共data_days日期序号
        days = range(today[1] - self.data_days, today[1])
        for stock_code in self.stocks_codes:
            sum_MF = 0
            valid_days = self.data_days
            stock_data = pd.read_csv('{}{}.csv'.format(self.data_path, stock_code), index_col='date')
            for day in days:
                if self.trading_dates[day] in stock_data.index:
                    # 加入收益率
                    pc = stock_data.loc[self.trading_dates[day], 'preclose']
                    close = stock_data.loc[self.trading_dates[day], 'close']
                    sum_MF += close / pc
                else:
                    valid_days -= 1
            if valid_days != 0:
                aver_MF = sum_MF / valid_days
            else:
                aver_MF = 0
            stocks_data.loc[stock_code, 'aver_MF'] = aver_MF
        # print(stocks_data)
        stocks_data.sort_values(by='aver_MF', ascending=False, inplace=True)
        # print(stocks_data.index)
        return stocks_data.index[0:number]

    def choose_by_tr(self, today: tuple, number: int):
        """
        选择最近data_days中换手率因子(Unusual Turnover Rate, 异常换手率)最高的number只股票
        """
        # 第一次买入策略应大于策略所需数据天数
        if today[1] < self.data_days:
            return pd.Series(None)
        # 建立用于计算平均TR的DF
        stocks_data = pd.DataFrame(self.stocks_codes)
        stocks_data['aver_TR'] = 0
        stocks_data = stocks_data.set_index('code')
        # 到交易日前一日为止共data_days日期序号
        days = range(today[1] - self.data_days, today[1])
        for stock_code in self.stocks_codes:
            sum_TR = 0
            valid_days = self.data_days
            stock_data = pd.read_csv('{}{}.csv'.format(self.data_path, stock_code), index_col='date')
            for day in days:
                if self.trading_dates[day] in stock_data.index:
                    # 加入换手率
                    tr = stock_data.loc[self.trading_dates[day], 'turn']
                    sum_TR += tr
                else:
                    valid_days -= 1
            if valid_days != 0:
                aver_TR = sum_TR / valid_days
                tr = stock_data.loc[self.trading_dates[today[1] - 1], 'turn']
                ratio = tr / aver_TR
            else:
                ratio = 0

            stocks_data.loc[stock_code, 'ratio'] = ratio
        # print(stocks_data)
        stocks_data.sort_values(by='ratio', ascending=False, inplace=True)
        # print(stocks_data.index)
        return stocks_data.index[0:number]
