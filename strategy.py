# -*- coding: utf-8 -*-
"""
@version: 3.8.3
@time: 21/4/29 9:30
@author: Yamisora
@file: strategy.py
"""
from prediction import *


class Strategy:
    def __init__(self, data_days=10):
        """
        选股策略
        @data_days: 回测选择数据日期
        """
        # 策略所需数据天数
        self.data_days = data_days
        # index选择指数组合
        self.index_name = 'hs300'
        self.index_code = 'sh.000300'
        # 存储路径
        self.base_data_path = './data/'
        self.data_path = './data/stocks/'
        self.train_data_path = './data/train_data/'
        # 指数组合内股票名称,代码数据
        self.stocks = pd.read_csv('{}{}_stocks.csv'.format(self.base_data_path, self.index_name))
        self.stocks_codes = self.stocks['code']
        # 指数日线数据
        self.index = pd.read_csv('{}{}.csv'.format(self.data_path, self.index_code))
        # 交易日str序列
        self.trading_dates = self.index['date']
        # 训练CNN模型
        self.dataset = StockDataset(data_days=data_days)
        self.prediction = Prediction(data_days=data_days, batch_size=50)
        self.prediction.train_cnn(self.dataset, retrain=False, epochs=2)
        self.prediction.train_lstm(self.dataset, retrain=False, epochs=2)
        # self.prediction.train_gru(self.dataset, retrain=False, epochs=2)
        # self.prediction.train_rnn_tanh(self.dataset, retrain=False, epochs=2)
        self.prediction.train_rnn_relu(self.dataset, retrain=False, epochs=2)
        self.prediction.train_resnet18(self.dataset, retrain=False, epochs=2)
        self.prediction.train_resnet34(self.dataset, retrain=False, epochs=2)
        self.prediction.train_resnet50(self.dataset, retrain=False, epochs=2)
        self.prediction.train_resnet101(self.dataset, retrain=False, epochs=2)
        self.prediction.train_resnet152(self.dataset, retrain=False, epochs=2)
        self.prediction.train_densenet(self.dataset, retrain=False, epochs=2)

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
            if aver_BM > 0:
                stocks_data.loc[stock_code, 'aver_BM'] = aver_BM
        # print(stocks_data)
        stocks_data.sort_values(by='aver_BM', ascending=False, inplace=True)
        # print(stocks_data.index)
        if len(stocks_data.index) > number:
            # 取0到number-1共number只股票
            return stocks_data.index[0:number]
        else:
            # 取全部股票
            return stocks_data.index[:]

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
            if aver_MF > 0:
                stocks_data.loc[stock_code, 'aver_MF'] = aver_MF
        # print(stocks_data)
        stocks_data.sort_values(by='aver_MF', ascending=False, inplace=True)
        # print(stocks_data.index)
        if len(stocks_data.index) > number:
            # 取0到number-1共number只股票
            return stocks_data.index[0:number]
        else:
            # 取全部股票
            return stocks_data.index[:]

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
            if valid_days > 2:
                aver_TR = sum_TR / valid_days
                tr1 = stock_data.loc[self.trading_dates[days[-1]], 'turn']
                tr2 = stock_data.loc[self.trading_dates[days[-1] - 1], 'turn']
                ratio = (tr1 + tr2) / (aver_TR * 2)
            else:
                ratio = 0
            if ratio > 0:
                stocks_data.loc[stock_code, 'ratio'] = ratio

        # print(stocks_data)
        stocks_data.sort_values(by='ratio', ascending=False, inplace=True)
        # print(stocks_data.index)
        if len(stocks_data.index) > number:
            # 取0到number-1共number只股票
            return stocks_data.index[0:number]
        else:
            # 取全部股票
            return stocks_data.index[:]

    def __nn_choose(self, model_type: str, today: tuple, number: int):
        """
        选择指定NN预测未来data_days天涨幅最高的number只股票
        """
        # 第一次买入策略应大于策略所需数据天数
        if today[1] < self.data_days:
            return pd.Series(None)
        # 建立用于计算预测涨跌幅的DF
        stocks_data = pd.DataFrame(self.stocks_codes)
        stocks_data['change'] = 0
        stocks_data = stocks_data.set_index('code')
        avail_num = 0
        # 预测每只股票未来data_days天涨跌幅
        for stock_code in self.stocks_codes:
            change = getattr(self.prediction, 'predict_' + model_type)(stock_code, today)
            if type(change) != int:
                # tensor直接取值
                change = change[0, 0].item()
            if change > 0:
                # 去除小于0的预测值
                stocks_data.loc[stock_code, 'change'] = change
                avail_num += 1
        # 排序
        stocks_data.sort_values(by='change', ascending=False, inplace=True)
        if avail_num > number:
            # 取0到number-1共number只股票
            return stocks_data.index[0:number]
        else:
            # 取全部有效股票
            return stocks_data.index[:avail_num]

    def choose_by_cnn(self, today: tuple, number: int):
        return self.__nn_choose('cnn', today, number)

    def choose_by_lstm(self, today: tuple, number: int):
        return self.__nn_choose('lstm', today, number)

    def choose_by_gru(self, today: tuple, number: int):
        return self.__nn_choose('gru', today, number)

    def choose_by_rnn_tanh(self, today: tuple, number: int):
        return self.__nn_choose('rnn_tanh', today, number)

    def choose_by_rnn_relu(self, today: tuple, number: int):
        return self.__nn_choose('rnn_relu', today, number)

    def choose_by_resnet18(self, today: tuple, number: int):
        return self.__nn_choose('resnet18', today, number)

    def choose_by_resnet34(self, today: tuple, number: int):
        return self.__nn_choose('resnet34', today, number)

    def choose_by_resnet50(self, today: tuple, number: int):
        return self.__nn_choose('resnet50', today, number)

    def choose_by_resnet101(self, today: tuple, number: int):
        return self.__nn_choose('resnet101', today, number)

    def choose_by_resnet152(self, today: tuple, number: int):
        return self.__nn_choose('resnet152', today, number)

    def choose_by_densenet(self, today: tuple, number: int):
        return self.__nn_choose('densenet', today, number)

    def choose_by_ensemble(self, today: tuple):
        chosen_num = pd.DataFrame()
        chosen_num['code'] = self.stocks_codes
        chosen_num['num'] = 0
        chosen_num.set_index('code', inplace=True)
        number = 300
        for chosen in (self.__nn_choose('cnn', today, number),
                       self.__nn_choose('lstm', today, number),
                       self.__nn_choose('rnn_relu', today, number),
                       self.__nn_choose('resnet18', today, number),
                       self.__nn_choose('resnet34', today, number),
                       self.__nn_choose('resnet50', today, number),
                       self.__nn_choose('densenet', today, number)):
            for stock_code in self.stocks_codes:
                if stock_code in chosen:
                    chosen_num.loc[stock_code, 'num'] += 1
        return chosen_num[chosen_num['num'] >= 3].index
