# -*- coding: utf-8 -*-
"""
@version: 3.8.3
@time: 21/5/11 8:33
@author: Yamisora
@file: prediction.py
"""
import torch
import torch.nn as nn
# import torch.nn.functional as F
import pandas as pd


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class RNNModel(torch.nn.Module):
    def __init__(self, rnn_type, input_size, hidden_size, n_layers, dropout=0.5):
        super(RNNModel, self).__init__()
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(input_size, hidden_size, n_layers, dropout=dropout)
        else:
            try:
                non_linearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError("""非可选RNN类型,可选参数:['LSTM', 'GRU', 'RNN_TANH', 'RNN_RELU']""")
            self.rnn = nn.RNN(input_size, hidden_size, n_layers, nonlinearity=non_linearity, dropout=dropout)

        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.n_layers = n_layers

    def forward(self, x, hidden):
        output, hidden = self.rnn(x, hidden)
        return output, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.n_layers, batch_size, self.hidden_size),
                    weight.new_zeros(self.n_layers, batch_size, self.hidden_size))
        else:
            return weight.new_zeros(self.n_layers, batch_size, self.hidden_size)


class Prediction:
    def __init__(self, data_days=30, index=0, batch_size=50):
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
        # 一次喂入数据批次
        self.batch_size = batch_size
        # 输入列
        self.input_columns = 'open', 'high', 'low', 'close', 'preclose', 'volume', 'amount', \
                             'turn', 'pctChg', 'peTTM', 'psTTM', 'pcfNcfTTM', 'pbMRQ'
        # RNN类型 输入大小 隐层大小 隐层数
        rnn_type = 'LSTM'
        input_size = len(self.input_columns)
        hidden_size = 10
        n_layers = data_days
        # 初始化模型
        self.model = RNNModel(rnn_type, input_size, hidden_size, n_layers).to(device)
        # 初始化训练数据
        self.train_data = pd.DataFrame()
        for stock_code in self.stocks_codes:
            stock_data = pd.read_csv('{}{}'.format(self.data_path, stock_code))
            batches = len(stock_data.index) - 2 * self.data_days
            if batches <= 0:
                continue
            for i in range(batches):
                predict_high = stock_data.loc[data_days+i, 'high']
                predict_low = stock_data.loc[data_days+i, 'low']
                predict_change = stock_data.loc[2*data_days+i, 'close']/stock_data.loc[data_days+i, 'close']
                # TODO

    def get_batch(self):
        # TODO
        pass

    def train(self):
        self.model.train()
        hidden = self.model.init_hidden(self.batch_size)
        # TODO

    def predict(self, stock_code: str, today: tuple):
        self.model.eval()
        # TODO
