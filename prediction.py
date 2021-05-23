# -*- coding: utf-8 -*-
"""
@version: 3.8.3
@time: 21/5/11 8:33
@author: Yamisora
@file: prediction.py
"""
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

# 运行设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class StockDataset(Dataset):
    def __init__(self, data_days=30, index=0, remake_data=False):
        super(StockDataset, self).__init__()
        # 存储路径
        self.base_data_path = './data/'
        self.data_path = './data/stocks/'
        # 策略所需数据天数
        self.data_days = data_days
        # index选择指数组合
        self.index_name = 'sz50', 'hs300', 'zz500'
        self.indexes = 'sh.000016', 'sh.000300', 'sh.000905'
        # 指数组合内股票名称,代码数据
        self.stocks = pd.read_csv('{}{}_stocks.csv'.format(self.base_data_path, self.index_name[index]))
        self.stocks_codes = self.stocks['code']
        # 输入列
        self.input_columns = ('open', 'high', 'low', 'close', 'preclose', 'volume', 'amount',
                              'turn', 'pctChg', 'peTTM', 'psTTM', 'pcfNcfTTM', 'pbMRQ')
        # 数据集
        if not os.path.exists('{}{}.pkl'.format(self.base_data_path, self.index_name[index])):
            remake_data = True
        if remake_data:
            self.data = pd.DataFrame()
            for stock_code in tqdm(self.stocks_codes):
                stock_data = pd.read_csv('{}{}.csv'.format(self.data_path, stock_code))
                stock_data = pd.DataFrame(stock_data, columns=self.input_columns)
                batches = len(stock_data.index) - 2 * self.data_days
                if batches <= 0:
                    continue
                # 数据集存入个股训练数据
                for i in range(batches):
                    # predict_high = stock_data.loc[data_days+i, 'high']
                    # predict_low = stock_data.loc[data_days+i, 'low']
                    predict_change = stock_data.loc[2 * data_days + i, 'close'] / stock_data.loc[data_days + i, 'close']
                    self.data = self.data.append({'data': stock_data[i:i+self.data_days].values,
                                                  'label': predict_change}, ignore_index=True)
            self.data.to_pickle('{}{}.pkl'.format(self.base_data_path, self.index_name[index]))
        else:
            self.data = pd.read_pickle('{}{}.pkl'.format(self.base_data_path, self.index_name[index]))

    def __len__(self):
        """返回整个数据集的大小"""
        return len(self.data)

    def __getitem__(self, idx):
        """根据索引index返回dataset[index]"""
        data = torch.tensor(self.data.loc[idx, 'data'], dtype=torch.float32, device=device)
        label = torch.tensor(self.data.loc[idx, 'label'], dtype=torch.float32, device=device)
        return data, label


class CNNModel(nn.Module):
    def __init__(self, data_days=30, input_size=13):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)  # 输入通道数为1，输出通道数为6
        self.conv2 = nn.Conv2d(6, 16, 3)  # 输入通道数为6，输出通道数为16
        self.fc1 = nn.Linear((data_days-4) * (input_size-4) * 16, 120)
        self.fc2 = nn.Linear(120, 60)
        self.fc3 = nn.Linear(60, 1)

    def forward(self, x):
        x = x.view(x.size()[0], 1, x.size()[1], x.size()[2])
        # 输入x (50, 1, 30, 13) -> conv1 (50, 6, 28, 11) -> relu
        x = self.conv1(x)
        x = F.relu(x)
        # 输入x (50, 6, 28, 11) -> conv2 (50, 16, 26, 9) -> relu
        x = self.conv2(x)
        x = F.relu(x)
        # view函数将张量x变形成一维向量形式，总特征数不变，为全连接层做准备
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class RNNModel(nn.Module):
    def __init__(self, rnn_type, input_size, hidden_size, n_layers, dropout=0):
        super(RNNModel, self).__init__()
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(input_size, hidden_size, n_layers,
                                             dropout=dropout, batch_first=True)
        else:
            try:
                non_linearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError("""非可选RNN类型,可选参数:['LSTM', 'GRU', 'RNN_TANH', 'RNN_RELU']""")
            self.rnn = nn.RNN(input_size, hidden_size, n_layers, nonlinearity=non_linearity,
                              dropout=dropout, batch_first=True)
        self.fc = nn.Linear(in_features=hidden_size, out_features=1)

        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.n_layers = n_layers

    def forward(self, x):
        x, _ = self.rnn(x)
        x = F.relu(x)
        x = self.fc(x[:, -1])
        return x

    # def forward(self, x, hidden):
    #     output, _ = self.rnn(x, hidden)
    #     output = self.fc(output)
    #     return output

    # def init_hidden(self, batch_size):
    #     weight = next(self.parameters())
    #     if self.rnn_type == 'LSTM':
    #         return (weight.new_zeros(batch_size, self.n_layers, self.hidden_size),
    #                 weight.new_zeros(batch_size, self.n_layers, self.hidden_size))
    #     else:
    #         return weight.new_zeros(batch_size, self.n_layers, self.hidden_size)


class Prediction:
    def __init__(self, data_days=30, index=0, batch_size=50):
        # 策略所需数据天数
        self.data_days = data_days
        # index选择指数组合
        self.index_name = 'sz50', 'hs300', 'zz500'
        self.indexes = 'sh.000016', 'sh.000300', 'sh.000905'
        self.idx = index
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
        self.input_columns = ('open', 'high', 'low', 'close', 'preclose', 'volume', 'amount',
                              'turn', 'pctChg', 'peTTM', 'psTTM', 'pcfNcfTTM', 'pbMRQ')
        input_size = len(self.input_columns)

        # 选择cnn模型
        self.cnn_model = CNNModel(data_days, input_size).to(device)

        # RNN类型 输入大小 隐层大小 隐层数
        rnn_types = 'LSTM', 'GRU', 'RNN_TANH', 'RNN_RELU'
        rnn_type = rnn_types[2]
        hidden_size = 20
        n_layers = 1
        # 初始化模型
        self.rnn_model = RNNModel(rnn_type, input_size, hidden_size, n_layers).to(device)

        # 使用MSE误差
        self.criterion = nn.MSELoss()
        # 使用Adam优化器 默认参数
        self.cnn_optimizer = torch.optim.Adam(self.cnn_model.parameters())
        self.rnn_optimizer = torch.optim.Adam(self.rnn_model.parameters())

    def train_cnn(self, train_dataset, epochs=2, retrain=False):
        if not os.path.exists('{}{}_CNN_model.pkl'.format(self.base_data_path, self.index_name[self.idx])):
            retrain = True
        if not retrain:
            # 读取训练好的模型
            self.cnn_model = torch.load('{}{}_CNN_model.pkl'.format(self.base_data_path, self.index_name[self.idx]))
            return
        # 生成训练数据
        train_data = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        # 设置为训练模式
        self.cnn_model.train()
        for epoch in range(epochs):
            for data, label in tqdm(train_data):
                # 前向传播 计算结果
                output = self.cnn_model.forward(data)
                # 使label与输出维度一致
                label = label.view(label.size()[0], 1)
                # 计算误差
                loss = self.criterion(output, label)
                # 清除梯度记录
                self.cnn_optimizer.zero_grad()
                # 误差反向传播
                loss.backward()
                # 优化器更新参数
                self.cnn_optimizer.step()
                # print('Train loss: ', loss.item())
        # 保存训练好的模型
        torch.save(self.cnn_model, '{}{}_CNN_model.pkl'.format(self.base_data_path, self.index_name[self.idx]))

    def train_rnn(self, train_dataset, epochs=2, retrain=False):
        if not os.path.exists('{}{}_RNN_model.pkl'.format(self.base_data_path, self.index_name[self.idx])):
            retrain = True
        if not retrain:
            # 读取训练好的模型
            self.rnn_model = torch.load('{}{}_RNN_model.pkl'.format(self.base_data_path, self.index_name[self.idx]))
            return
        # 生成训练数据
        train_data = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        # 设置为训练模式
        self.rnn_model.train()
        # hidden = self.model.init_hidden(self.batch_size)
        for epoch in range(epochs):
            for data, label in tqdm(train_data):
                # 前向传播 计算结果
                output = self.rnn_model.forward(data)
                # 使label与输出维度一致
                label = label.view(label.size()[0], 1)
                # 计算误差
                loss = self.criterion(output, label)
                # 清除梯度记录
                self.cnn_optimizer.zero_grad()
                # 误差反向传播
                loss.backward()
                # 优化器更新参数
                self.cnn_optimizer.step()
                print('Train loss: ', loss.item())
        # 保存训练好的模型
        torch.save(self.rnn_model, '{}{}_RNN_model.pkl'.format(self.base_data_path, self.index_name[self.idx]))

    def predict_data(self, stock_code: str, today: tuple):
        stock_data = pd.read_csv('{}{}.csv'.format(self.data_path, stock_code))
        # 当前日期在数据集中序号
        date_index = len(stock_data) - len(self.trading_dates) + today[1]
        if date_index < self.data_days:
            return 0
        # 生成预测数据
        stock_data = pd.DataFrame(stock_data, columns=self.input_columns)
        stock_data = stock_data[date_index - self.data_days:date_index]
        stock_data = np.reshape(stock_data.values, (1, self.data_days, len(self.input_columns)))
        stock_data = torch.tensor(stock_data, dtype=torch.float32, device=device)
        return stock_data

    def predict_cnn(self, stock_code: str, today: tuple):
        # 设置为预测模式
        self.cnn_model.eval()
        stock_data = self.predict_data(stock_code, today)
        with torch.no_grad():
            # 前向传播 输出结果
            output = self.cnn_model.forward(stock_data)
            return output

    def predict_rnn(self, stock_code: str, today: tuple):
        # 设置为预测模式
        self.rnn_model.eval()
        stock_data = self.predict_data(stock_code, today)
        with torch.no_grad():
            # 前向传播 输出结果
            output = self.rnn_model.forward(stock_data)
            return output


if __name__ == '__main__':
    dataset = StockDataset(index=1)
    print(len(dataset))
    prediction = Prediction(batch_size=50)
    # prediction.train_cnn(dataset)
    # out = prediction.predict_cnn(dataset.stocks_codes[0], (prediction.trading_dates[30], 30))
    prediction.train_rnn(dataset, retrain=True)
    out = prediction.predict_rnn(dataset.stocks_codes[0], (prediction.trading_dates[30], 30))
    print(out)
