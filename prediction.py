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
    def __init__(self, data_days=10, remake_data=False):
        super(StockDataset, self).__init__()
        # 存储路径
        self.base_data_path = './data/'
        self.data_path = './data/stocks/'
        self.train_data_path = './data/train_data/'
        # 策略所需数据天数
        self.data_days = data_days
        # 指数组合
        self.index_name = 'hs300'
        self.index_code = 'sh.000300'
        # 指数组合内股票名称,代码数据
        self.stocks = pd.read_csv('{}{}_stocks.csv'.format(self.base_data_path, self.index_name))
        self.stocks_codes = self.stocks['code']
        # 输入列
        self.input_columns = ('open', 'high', 'low', 'close', 'preclose', 'volume', 'amount',
                              'turn', 'pctChg', 'peTTM', 'psTTM', 'pcfNcfTTM', 'pbMRQ',
                              'total_share', 'float_share', 'free_share', 'total_mv', 'circ_mv')
        # 数据集
        if not os.path.exists('{}{}.pkl'.format(self.base_data_path, self.index_name)):
            remake_data = True
        if remake_data:
            data = []
            for stock_code in tqdm(self.stocks_codes):
                # 读取数据
                stock_data = pd.read_csv('{}{}.csv'.format(self.train_data_path, stock_code))
                # 选择指定列
                stock_data = pd.DataFrame(stock_data, columns=self.input_columns)
                # 归一化数据
                minimum = stock_data.min()
                r = stock_data.max() - minimum

                batches = len(stock_data.index) - 2 * self.data_days
                if batches <= 0:
                    continue
                # 数据集存入个股训练数据
                for i in range(batches):
                    # 清除无效数据
                    if 0 in stock_data[i:i+self.data_days].values:
                        continue
                    predict_high = stock_data.loc[data_days+i, 'high']
                    predict_low = stock_data.loc[data_days+i, 'low']
                    predict_change = stock_data.loc[2 * data_days + i, 'close'] / stock_data.loc[data_days + i, 'close']
                    # 归一化后存入
                    data.append({'data': ((stock_data[i:i+self.data_days]-minimum)/r).values,
                                 'label': [predict_change, predict_low, predict_high]})
            self.data = pd.DataFrame(data)
            self.data.to_pickle('{}{}.pkl'.format(self.base_data_path, self.index_name))
        else:
            self.data = pd.read_pickle('{}{}.pkl'.format(self.base_data_path, self.index_name))

    def __len__(self):
        """返回整个数据集的大小"""
        return len(self.data)

    def __getitem__(self, idx):
        """根据索引index返回dataset[index]"""
        data = torch.tensor(self.data.loc[idx, 'data'], dtype=torch.float32, device=device)
        label = torch.tensor(self.data.loc[idx, 'label'], dtype=torch.float32, device=device)
        return data, label


class CNNModel(nn.Module):
    """类LeNet结构CNN模型"""
    def __init__(self, data_days=10, input_size=18):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)  # 输入通道数为1，输出通道数为6
        self.conv2 = nn.Conv2d(6, 16, 3)  # 输入通道数为6，输出通道数为16
        self.fc1 = nn.Linear((data_days-4) * (input_size-4) * 16, 120)
        self.fc2 = nn.Linear(120, 60)
        self.fc3 = nn.Linear(60, 3)

    def forward(self, x):
        x = x.view(x.size()[0], 1, x.size()[1], x.size()[2])
        # 输入x (50, 1, 10, 18) -> conv1 (50, 6, 8, 16) -> relu
        x = self.conv1(x)
        x = F.relu(x)
        # 输入x (50, 6, 8, 16) -> conv2 (50, 16, 6, 14) -> relu
        x = self.conv2(x)
        x = F.relu(x)
        # view函数将张量x变形成一维向量形式，总特征数不变，为全连接层做准备
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class RNNModel(nn.Module):
    def __init__(self, rnn_type, input_size, hidden_size, n_layers):
        super(RNNModel, self).__init__()
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(input_size, hidden_size, n_layers, batch_first=True)
        else:
            try:
                non_linearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError("""非可选RNN类型,可选参数:['LSTM', 'GRU', 'RNN_TANH', 'RNN_RELU']""")
            self.rnn = nn.RNN(input_size, hidden_size, n_layers, nonlinearity=non_linearity, batch_first=True)
        # self.fc1 = nn.Linear(hidden_size, 120)
        # self.fc2 = nn.Linear(120, 60)
        # self.fc3 = nn.Linear(60, 1)
        # self.norm = nn.BatchNorm1d(10)
        self.fc = nn.Linear(hidden_size, 3)

        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.n_layers = n_layers

    def forward(self, x):
        # h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size).to(device)
        # c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size).to(device)
        # if self.rnn_type == 'LSTM':
        #     x, _ = self.rnn(x, (h0, c0))
        # else:
        #     x, _ = self.rnn(x, h0)
        # x = self.norm(x)
        x, _ = self.rnn(x)
        # x = F.relu(self.fc1(x[:, -1, :]))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        print(x)
        x = torch.sigmoid(x[:, -1, :])
        print(x)
        x = self.fc(x)
        return x


# 用于ResNet18和34的残差块，用的是2个3x3的卷积
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        # 经过处理后的x要与x的维度相同(尺寸和深度)
        # 如果不相同，需要添加卷积+BN来变换为同一维度
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# 用于ResNet50,101和152的残差块，用的是1x1+3x3+1x1的卷积
class Bottleneck(nn.Module):
    # 前面1x1和3x3卷积的filter个数相等，最后1x1卷积是其expansion倍
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=3):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(4 * 512 * block.expansion, num_classes)  # 去掉池化变成4倍

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size()[0], 1, x.size()[1], x.size()[2])
        print(x)
        out = F.relu(self.bn1(self.conv1(x)))
        print(out)
        out = self.layer1(out)
        print(out)
        out = self.layer2(out)
        print(out)
        out = self.layer3(out)
        print(out)
        out = self.layer4(out)
        print(out)
        # out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


class Prediction:
    def __init__(self, data_days=10, batch_size=50):
        # 策略所需数据天数
        self.data_days = data_days
        # 指数组合
        self.index_name = 'hs300'
        self.index_code = 'sh.000300'
        # 存储路径
        self.base_data_path = './data/'
        self.data_path = './data/stocks/'
        # 指数组合内股票名称,代码数据
        self.stocks = pd.read_csv('{}{}_stocks.csv'.format(self.base_data_path, self.index_name))
        self.stocks_codes = self.stocks['code']
        # 指数日线数据
        self.index = pd.read_csv('{}{}.csv'.format(self.data_path, self.index_code))
        # 交易日str序列
        self.trading_dates = self.index['date']
        # 一次喂入数据批次
        self.batch_size = batch_size
        # 输入列
        self.input_columns = ('open', 'high', 'low', 'close', 'preclose', 'volume', 'amount',
                              'turn', 'pctChg', 'peTTM', 'psTTM', 'pcfNcfTTM', 'pbMRQ',
                              'total_share', 'float_share', 'free_share', 'total_mv', 'circ_mv')
        input_size = len(self.input_columns)

        # 选择cnn模型
        self.cnn_model = CNNModel(data_days, input_size).to(device)

        # RNN类型 输入大小 隐层大小 隐层数
        rnn_types = 'LSTM', 'GRU', 'RNN_TANH', 'RNN_RELU'
        rnn_type = rnn_types[3]
        hidden_size = 10
        n_layers = 1
        # 初始化模型
        self.rnn_model = RNNModel(rnn_type, input_size, hidden_size, n_layers).to(device)

        # ResNet模型
        self.resnet18 = ResNet18().to(device)
        self.resnet34 = ResNet34().to(device)
        self.resnet50 = ResNet50().to(device)
        self.resnet101 = ResNet101().to(device)
        self.resnet152 = ResNet152().to(device)

        # 使用MSE误差
        self.criterion = nn.MSELoss()
        # 使用Adam优化器 默认参数
        self.cnn_optimizer = torch.optim.Adam(self.cnn_model.parameters())
        self.rnn_optimizer = torch.optim.Adam(self.rnn_model.parameters())
        self.rn18_optimizer = torch.optim.Adam(self.resnet18.parameters())
        self.rn34_optimizer = torch.optim.Adam(self.resnet34.parameters())
        self.rn50_optimizer = torch.optim.Adam(self.resnet50.parameters())
        self.rn101_optimizer = torch.optim.Adam(self.resnet101.parameters())
        self.rn152_optimizer = torch.optim.Adam(self.resnet152.parameters())

    def train_cnn(self, train_dataset, epochs=2, retrain=False):
        if not os.path.exists('{}{}_CNN_model.pt'.format(self.base_data_path, self.index_name)):
            retrain = True
        if not retrain:
            # 读取训练好的模型
            self.cnn_model = torch.load('{}{}_CNN_model.pt'.format(self.base_data_path, self.index_name))
            return
        # 生成训练数据
        train_data = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        # 设置为训练模式
        self.cnn_model.train()
        for epoch in range(epochs):
            for data, label in tqdm(train_data):
                # 前向传播 计算结果
                output = self.cnn_model.forward(data)
                # # 使label与输出维度一致
                # label = label.view(label.size()[0], 1)
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
        torch.save(self.cnn_model, '{}{}_CNN_model.pt'.format(self.base_data_path, self.index_name))

    def train_rnn(self, train_dataset, epochs=2, retrain=False):
        if not os.path.exists('{}{}_RNN_model.pt'.format(self.base_data_path, self.index_name)):
            retrain = True
        if not retrain:
            # 读取训练好的模型
            self.rnn_model = torch.load('{}{}_RNN_model.pt'.format(self.base_data_path, self.index_name))
            return
        # 生成训练数据
        train_data = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        # 设置为训练模式
        self.rnn_model.train()
        for epoch in range(epochs):
            for data, label in tqdm(train_data):
                # 前向传播 计算结果
                output = self.rnn_model.forward(data)
                # print(output)
                # # 使label与输出维度一致
                # label = label.view(label.size()[0], 1)
                # print(label)
                # 计算误差
                loss = self.criterion(output, label)
                # 清除梯度记录
                self.rnn_optimizer.zero_grad()
                # 误差反向传播
                loss.backward()
                # # 梯度裁剪
                # for p in self.rnn_model.parameters():
                #     print(p.grad.norm())                 # 查看参数p的梯度
                #     torch.nn.utils.clip_grad_norm_(self.rnn_model.parameters(), max_norm=20, norm_type=2)
                # 优化器更新参数
                self.rnn_optimizer.step()
                print('Train loss: ', loss.item())
        # 保存训练好的模型
        torch.save(self.rnn_model, '{}{}_RNN_model.pt'.format(self.base_data_path, self.index_name))

    def train_resnet18(self, train_dataset, epochs=2, retrain=False):
        if not os.path.exists('{}{}_rn18_model.pt'.format(self.base_data_path, self.index_name)):
            retrain = True
        if not retrain:
            # 读取训练好的模型
            self.resnet18 = torch.load('{}{}_rn18_model.pt'.format(self.base_data_path, self.index_name))
            return
        # 生成训练数据
        train_data = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        # 设置为训练模式
        self.resnet18.train()
        for epoch in range(epochs):
            # for data, label in tqdm(train_data):
            for data, label in train_data:
                # 前向传播 计算结果
                output = self.resnet18.forward(data)
                print(output)
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
        torch.save(self.resnet18, '{}{}_rn18_model.pt'.format(self.base_data_path, self.index_name))

    def predict_data(self, stock_code: str, today: tuple):
        stock_data = pd.read_csv('{}{}.csv'.format(self.data_path, stock_code))
        # 当前日期在数据集中序号
        date_index = len(stock_data) - len(self.trading_dates) + today[1]
        # 数据不足时返回0
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
        if type(stock_data) == int:
            return 0
        with torch.no_grad():
            # 前向传播 输出结果
            output = self.cnn_model.forward(stock_data)
            return output

    def predict_rnn(self, stock_code: str, today: tuple):
        # 设置为预测模式
        self.rnn_model.eval()
        stock_data = self.predict_data(stock_code, today)
        if type(stock_data) == int:
            return 0
        with torch.no_grad():
            # 前向传播 输出结果
            output = self.rnn_model.forward(stock_data)
            return output


if __name__ == '__main__':
    dataset = StockDataset(data_days=10, remake_data=False)
    print(len(dataset))
    prediction = Prediction(data_days=10, batch_size=50)
    prediction.train_cnn(dataset, retrain=False, epochs=2)
    out1 = prediction.predict_cnn(dataset.stocks_codes[0], (prediction.trading_dates[1], 1))
    out2 = prediction.predict_cnn(dataset.stocks_codes[0], (prediction.trading_dates[30], 30))
    # print(out1, out2)
    # prediction.train_rnn(dataset, retrain=True, epochs=1)
    # out = prediction.predict_rnn(dataset.stocks_codes[0], (prediction.trading_dates[30], 30))
    # print(out)
    # prediction.train_resnet18(dataset, epochs=1, retrain=True)
