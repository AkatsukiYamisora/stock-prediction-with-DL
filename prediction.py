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
import matplotlib.pyplot as plt
from collections import OrderedDict

# 运行设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def return_rate_transform(return_rate):
    """按收益率分布转换为数值分类模型"""
    if return_rate < -0.093:
        return -1.0
    elif return_rate < -0.053:
        return -0.8
    elif return_rate < -0.030:
        return -0.6
    elif return_rate < -0.014:
        return -0.4
    elif return_rate < 0.000:
        return -0.2
    elif return_rate < 0.016:
        return 0.2
    elif return_rate < 0.034:
        return 0.4
    elif return_rate < 0.058:
        return 0.6
    elif return_rate < 0.100:
        return 0.8
    elif return_rate >= 0.100:
        return 1.0


class StockDataset(Dataset):
    """沪深300股票训练数据集"""
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
        self.input_columns = ('open', 'high', 'low', 'close', 'preclose',
                              'turn', 'peTTM', 'psTTM', 'pcfNcfTTM', 'pbMRQ',)
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

                batches = len(stock_data.index) - 2 * self.data_days
                if batches <= 0:
                    continue
                # 数据集存入个股训练数据
                for i in range(batches):
                    # 清除无效数据(0)
                    if 0 in stock_data[i:i + self.data_days].values:
                        continue
                    # 当前日期为data_days + i
                    # data_days后收盘价
                    next_price = stock_data.loc[2 * data_days + i, 'close']
                    # 当前日期收盘价
                    this_price = stock_data.loc[data_days + i, 'close']
                    # high_change = stock_data.loc[data_days + i, 'high'] / stock_data.loc[data_days + i - 1, 'high']-1
                    # low_change = stock_data.loc[data_days + i, 'low'] / stock_data.loc[data_days + i - 1, 'low'] - 1
                    close_change = this_price / stock_data.loc[data_days + i - 1, 'close'] - 1
                    predict_change = (next_price / this_price - 1)
                    # 当前日期前一天到前data_days天 共data_days天数据
                    data.append({'data': stock_data[i:i + self.data_days].values,
                                 'label': [predict_change, close_change]})
                    #            'label': [predict_change, low_change, high_change, close_change]})
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
    def __init__(self, input_size, data_days=10):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)  # 输入通道数为1，输出通道数为6
        self.conv2 = nn.Conv2d(6, 16, 3)  # 输入通道数为6，输出通道数为16
        self.fc1 = nn.Linear((data_days - 4) * (input_size - 4) * 16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = x.view(x.size()[0], 1, x.size()[1], x.size()[2])
        # 输入x (50, 1, 10, 10) -> conv1 (50, 6, 8, 8) -> relu
        x = self.conv1(x)
        x = F.relu(x)
        # 输入x (50, 6, 8, 8) -> conv2 (50, 16, 6, 6) -> relu
        x = self.conv2(x)
        x = F.relu(x)
        # view函数将张量x变形成一维向量形式，总特征数不变，为全连接层做准备
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class RNNModel(nn.Module):
    """LSTM,GRU,使用tanh与relu激活的RNN四种结构模型"""
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
        self.fc1 = nn.Linear(hidden_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)
        # self.norm = nn.BatchNorm1d(10)
        # self.fc = nn.Linear(hidden_size, 3)

        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.n_layers = n_layers

    def forward(self, x):
        # x = self.norm(x)
        x, _ = self.rnn(x)
        x = F.relu(self.fc1(x[:, -1, :]))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # print(x)
        # x = F.relu(x[:, -1, :])
        # print(x)
        # x = self.fc(x)
        return x


class BasicBlock(nn.Module):
    """用于ResNet18和34的残差块，用的是2个3x3的卷积"""
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


class Bottleneck(nn.Module):
    """用于ResNet50,101和152的残差块，用的是1x1+3x3+1x1的卷积"""
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
    """实现将ResNet迁移应用于股票预测"""
    def __init__(self, block, num_blocks, num_classes=2):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(4 * 512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size()[0], 1, x.size()[1], x.size()[2])
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        # print(out.size())
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


class DenseLayer(nn.Sequential):
    def __init__(self, in_channels, growth_rate, bn_size):
        super(DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(in_channels))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv2d(in_channels, bn_size * growth_rate,
                                           kernel_size=1,
                                           stride=1, bias=False))
        self.add_module('norm2', nn.BatchNorm2d(bn_size*growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv2d(bn_size*growth_rate, growth_rate,
                                           kernel_size=3,
                                           stride=1, padding=1, bias=False))

    # 重载forward函数
    def forward(self, x):
        new_features = super(DenseLayer, self).forward(x)
        return torch.cat([x, new_features], 1)


class DenseBlock(nn.Sequential):
    def __init__(self, num_layers, in_channels, bn_size, growth_rate):
        super(DenseBlock, self).__init__()
        for i in range(num_layers):
            self.add_module('denselayer%d' % (i+1),
                            DenseLayer(in_channels + growth_rate * i,
                                       growth_rate, bn_size))


class Transition(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(in_channels, out_channels,
                                          kernel_size=1,
                                          stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNetBC(nn.Module):
    def __init__(self, growth_rate=12, block_config=(6, 12, 24, 16),
                 bn_size=4, theta=0.5, num_classes=2):
        super(DenseNetBC, self).__init__()

        # 初始的卷积为filter:2倍的growth_rate
        num_init_feature = 2 * growth_rate

        # 原DenseNet对cifar-10与ImageNet的分别初始化
        # if num_classes == 10:
        #     self.features = nn.Sequential(OrderedDict([
        #         ('conv0', nn.Conv2d(3, num_init_feature,
        #                             kernel_size=3, stride=1,
        #                             padding=1, bias=False)),
        #     ]))
        # else:
        #     self.features = nn.Sequential(OrderedDict([
        #         ('conv0', nn.Conv2d(3, num_init_feature,
        #                             kernel_size=7, stride=2,
        #                             padding=3, bias=False)),
        #         ('norm0', nn.BatchNorm2d(num_init_feature)),
        #         ('relu0', nn.ReLU(inplace=True)),
        #         ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        #     ]))
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(1, num_init_feature,
                                kernel_size=3, stride=1,
                                padding=1, bias=False)),
        ]))

        num_feature = num_init_feature
        for i, num_layers in enumerate(block_config):
            self.features.add_module('denseblock%d' % (i+1),
                                     DenseBlock(num_layers, num_feature,
                                                bn_size, growth_rate))
            num_feature = num_feature + growth_rate * num_layers
            if i != len(block_config)-1:
                self.features.add_module('transition%d' % (i + 1),
                                         Transition(num_feature,
                                                    int(num_feature * theta)))
                num_feature = int(num_feature * theta)

        self.features.add_module('norm5', nn.BatchNorm2d(num_feature))
        self.features.add_module('relu5', nn.ReLU(inplace=True))
        self.features.add_module('avg_pool', nn.AdaptiveAvgPool2d((1, 1)))

        self.linear = nn.Linear(num_feature, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 增加一个1的维度(图像处理中为RGB维度)
        x = x.view(x.size()[0], 1, x.size()[1], x.size()[2])
        features = self.features(x)
        out = features.view(features.size(0), -1)
        out = self.linear(out)
        return out


def dense_net_BC_100():
    return DenseNetBC(growth_rate=12, block_config=(16, 16, 16))


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
        self.input_columns = ('open', 'high', 'low', 'close', 'preclose',
                              'turn', 'peTTM', 'psTTM', 'pcfNcfTTM', 'pbMRQ',)
        input_size = len(self.input_columns)

        # cnn模型
        self.cnn = CNNModel(data_days, input_size).to(device)

        # RNN类型 输入大小 隐层大小 隐层数
        # rnn_types = 'LSTM', 'GRU', 'RNN_TANH', 'RNN_RELU'
        hidden_size = 20
        n_layers = 2
        # 初始化模型
        self.lstm = RNNModel('LSTM', input_size, hidden_size, n_layers).to(device)
        self.gru = RNNModel('GRU', input_size, hidden_size, n_layers).to(device)
        self.rnn_tanh = RNNModel('RNN_TANH', input_size, hidden_size, n_layers).to(device)
        self.rnn_relu = RNNModel('RNN_RELU', input_size, hidden_size, n_layers).to(device)

        # ResNet模型
        self.resnet18 = ResNet18().to(device)
        self.resnet34 = ResNet34().to(device)
        self.resnet50 = ResNet50().to(device)
        self.resnet101 = ResNet101().to(device)
        self.resnet152 = ResNet152().to(device)

        # DenseNet模型
        self.densenet = dense_net_BC_100().to(device)

        # 使用MSE误差
        self.criterion = nn.MSELoss()
        # 使用AdamW优化器 默认参数
        self.cnn_optimizer = torch.optim.AdamW(self.cnn.parameters())
        self.lstm_optimizer = torch.optim.AdamW(self.lstm.parameters())
        self.gru_optimizer = torch.optim.AdamW(self.gru.parameters())
        self.rnn_tanh_optimizer = torch.optim.AdamW(self.rnn_tanh.parameters())
        self.rnn_relu_optimizer = torch.optim.AdamW(self.rnn_relu.parameters())
        self.rn18_optimizer = torch.optim.AdamW(self.resnet18.parameters())
        self.rn34_optimizer = torch.optim.AdamW(self.resnet34.parameters())
        self.rn50_optimizer = torch.optim.AdamW(self.resnet50.parameters())
        self.rn101_optimizer = torch.optim.AdamW(self.resnet101.parameters())
        self.rn152_optimizer = torch.optim.AdamW(self.resnet152.parameters())
        self.densenet_optimizer = torch.optim.AdamW(self.densenet.parameters())

    def __train(self, model_name, model, optim, train_dataset, epochs=2):
        # 生成训练数据
        train_data = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        # 设置为训练模式
        model.train()
        print('*' * 20, '\n', model_name, '模型训练中')
        for epoch in range(epochs):
            for data, label in train_data:
                # 前向传播 计算结果
                output = model.forward(data)
                # label = label.view(label.size()[0], 1)
                # 计算误差
                loss = self.criterion(output, label)
                # 清除梯度记录
                optim.zero_grad()
                # 误差反向传播
                loss.backward()
                # 优化器更新参数
                optim.step()
                print('Train_loss:', loss.item(), end='\r')
                if loss.item() < 1e-3:
                    break
        # 保存训练好的模型
        torch.save(model, '{}{}.pt'.format(self.base_data_path, model_name))
        print('\n', model_name, '模型训练完成')

    def train_cnn(self, train_dataset, epochs=2, retrain=False):
        if not os.path.exists('{}CNN.pt'.format(self.base_data_path)):
            retrain = True
        if not retrain:
            # 读取训练好的模型
            self.cnn = torch.load('{}CNN.pt'.format(self.base_data_path, self.index_name))
            return
        self.__train('CNN', self.cnn, self.cnn_optimizer, train_dataset, epochs)

    def train_lstm(self, train_dataset, epochs=2, retrain=False):
        if not os.path.exists('{}LSTM.pt'.format(self.base_data_path)):
            retrain = True
        if not retrain:
            # 读取训练好的模型
            self.lstm = torch.load('{}LSTM.pt'.format(self.base_data_path))
            return
        self.__train('LSTM', self.lstm, self.lstm_optimizer, train_dataset, epochs)

    def train_gru(self, train_dataset, epochs=2, retrain=False):
        if not os.path.exists('{}GRU.pt'.format(self.base_data_path)):
            retrain = True
        if not retrain:
            # 读取训练好的模型
            self.rnn_relu = torch.load('{}GRU.pt'.format(self.base_data_path))
            return
        self.__train('GRU', self.gru, self.gru_optimizer, train_dataset, epochs)

    def train_rnn_tanh(self, train_dataset, epochs=2, retrain=False):
        if not os.path.exists('{}RNN_tanh.pt'.format(self.base_data_path)):
            retrain = True
        if not retrain:
            # 读取训练好的模型
            self.rnn_relu = torch.load('{}RNN_tanh.pt'.format(self.base_data_path))
            return
        self.__train('RNN_tanh', self.rnn_tanh, self.rnn_tanh_optimizer, train_dataset, epochs)

    def train_rnn_relu(self, train_dataset, epochs=2, retrain=False):
        if not os.path.exists('{}RNN_relu.pt'.format(self.base_data_path)):
            retrain = True
        if not retrain:
            # 读取训练好的模型
            self.rnn_relu = torch.load('{}RNN_relu.pt'.format(self.base_data_path))
            return
        self.__train('RNN_relu', self.rnn_relu, self.rnn_relu_optimizer, train_dataset, epochs)

    def train_resnet18(self, train_dataset, epochs=2, retrain=False):
        if not os.path.exists('{}resnet18.pt'.format(self.base_data_path)):
            retrain = True
        if not retrain:
            # 读取训练好的模型
            self.resnet18 = torch.load('{}resnet18.pt'.format(self.base_data_path))
            return
        self.__train('resnet18', self.resnet18, self.rn18_optimizer, train_dataset, epochs)

    def train_resnet34(self, train_dataset, epochs=2, retrain=False):
        if not os.path.exists('{}resnet34.pt'.format(self.base_data_path)):
            retrain = True
        if not retrain:
            # 读取训练好的模型
            self.resnet34 = torch.load('{}resnet34.pt'.format(self.base_data_path))
            return
        self.__train('resnet34', self.resnet34, self.rn34_optimizer, train_dataset, epochs)

    def train_resnet50(self, train_dataset, epochs=2, retrain=False):
        if not os.path.exists('{}resnet50.pt'.format(self.base_data_path)):
            retrain = True
        if not retrain:
            # 读取训练好的模型
            self.resnet50 = torch.load('{}resnet50.pt'.format(self.base_data_path))
            return
        self.__train('resnet50', self.resnet50, self.rn50_optimizer, train_dataset, epochs)

    def train_resnet101(self, train_dataset, epochs=2, retrain=False):
        if not os.path.exists('{}resnet101.pt'.format(self.base_data_path)):
            retrain = True
        if not retrain:
            # 读取训练好的模型
            self.resnet101 = torch.load('{}resnet101.pt'.format(self.base_data_path))
            return
        self.__train('resnet101', self.resnet101, self.rn101_optimizer, train_dataset, epochs)

    def train_resnet152(self, train_dataset, epochs=2, retrain=False):
        if not os.path.exists('{}resnet152.pt'.format(self.base_data_path)):
            retrain = True
        if not retrain:
            # 读取训练好的模型
            self.resnet152 = torch.load('{}resnet152.pt'.format(self.base_data_path))
            return
        self.__train('resnet152', self.resnet152, self.rn152_optimizer, train_dataset, epochs)

    def train_densenet(self, train_dataset, epochs=2, retrain=False):
        if not os.path.exists('{}densenet.pt'.format(self.base_data_path)):
            retrain = True
        if not retrain:
            # 读取训练好的模型
            self.densenet = torch.load('{}densenet.pt'.format(self.base_data_path))
            return
        self.__train('densenet', self.densenet, self.densenet_optimizer, train_dataset, epochs)

    def __predict_data(self, stock_code: str, today: tuple, abs_date=False):
        stock_data = pd.read_csv('{}{}.csv'.format(self.data_path, stock_code))
        # 当前日期在数据集中序号
        date_index = today[1] if abs_date else len(stock_data) - len(self.trading_dates) + today[1]
        # 数据不足时返回0
        if date_index < self.data_days:
            return 0
        # 生成预测数据
        stock_data = pd.DataFrame(stock_data, columns=self.input_columns)
        # 将0替换为上一行数据
        stock_data = stock_data.replace(0, None)
        stock_data = stock_data[date_index - self.data_days:date_index]
        stock_data = np.reshape(stock_data.values, (1, self.data_days, len(self.input_columns)))
        stock_data = torch.tensor(stock_data, dtype=torch.float32, device=device)
        return stock_data

    def __predict(self, model, stock_code: str, today: tuple):
        # 设置为预测模式
        model.eval()
        stock_data = self.__predict_data(stock_code, today)
        if type(stock_data) == int:
            return 0
        with torch.no_grad():
            # 前向传播 输出结果
            output = model.forward(stock_data)
            return output

    def predict_cnn(self, stock_code: str, today: tuple):
        return self.__predict(self.cnn, stock_code, today)

    def predict_lstm(self, stock_code: str, today: tuple):
        return self.__predict(self.lstm, stock_code, today)

    def predict_gru(self, stock_code: str, today: tuple):
        return self.__predict(self.gru, stock_code, today)

    def predict_rnn_tanh(self, stock_code: str, today: tuple):
        return self.__predict(self.rnn_tanh, stock_code, today)

    def predict_rnn_relu(self, stock_code: str, today: tuple):
        return self.__predict(self.rnn_relu, stock_code, today)

    def predict_resnet18(self, stock_code: str, today: tuple):
        return self.__predict(self.resnet18, stock_code, today)

    def predict_resnet34(self, stock_code: str, today: tuple):
        return self.__predict(self.resnet34, stock_code, today)

    def predict_resnet50(self, stock_code: str, today: tuple):
        return self.__predict(self.resnet50, stock_code, today)

    def predict_resnet101(self, stock_code: str, today: tuple):
        return self.__predict(self.resnet101, stock_code, today)

    def predict_resnet152(self, stock_code: str, today: tuple):
        return self.__predict(self.resnet152, stock_code, today)

    def predict_densenet(self, stock_code: str, today: tuple):
        return self.__predict(self.densenet, stock_code, today)


if __name__ == '__main__':
    dataset = StockDataset(data_days=10, remake_data=False)
    print('训练集大小:', len(dataset))

    prediction = Prediction(data_days=10, batch_size=200)

    # p2 = t_data.loc[trading_day1[1], 'high'] / t_data.loc[trading_day1[1] - 1, 'high'] - 1
    # p3 = t_data.loc[trading_day1[1], 'low'] / t_data.loc[trading_day1[1] - 1, 'low'] - 1
    # p4 = t_data.loc[trading_day1[1], 'close'] / t_data.loc[trading_day1[1] - 1, 'close'] - 1
    # print(return_rate_transform(p1), p2, p3, p4)

    prediction.train_cnn(dataset, retrain=False, epochs=1)
    prediction.train_lstm(dataset, retrain=False, epochs=1)
    # GRU与tanhRNN效果不佳 抛弃
    # prediction.train_gru(dataset, retrain=False, epochs=1)
    # prediction.train_rnn_tanh(dataset, retrain=False, epochs=1)
    prediction.train_rnn_relu(dataset, retrain=False, epochs=1)
    prediction.train_resnet18(dataset, retrain=False, epochs=1)
    prediction.train_resnet34(dataset, retrain=False, epochs=1)
    prediction.train_resnet50(dataset, retrain=False, epochs=1)
    prediction.train_resnet101(dataset, retrain=False, epochs=1)
    prediction.train_resnet152(dataset, retrain=False, epochs=1)
    prediction.train_densenet(dataset, retrain=False, epochs=1)

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=[30, 15], dpi=160)
    for code in dataset.stocks_codes[:5]:
        print('正在绘制'+code+'预测图像')
        plt.clf()
        df = pd.read_csv('./data/stocks/' + code + '.csv')
        trading_dates = df['date']
        x_r = range(0, len(trading_dates))
        x_ticks = list(x_r[::100])
        x_ticks.append(x_r[-1])
        x_labels = [trading_dates[i] for i in x_ticks]
        true_close = df['close'].values

        def close_p(x):
            if type(x) == int:
                return x
            x = x[0, 1].item()
            return x if 0.2 > x > -0.2 else 0.0

        print('计算CNN')
        cnn_close = [true_close[j]*(1+close_p(prediction.predict_cnn(code, (0, j))))
                     for j in range(len(trading_dates))]
        print('计算LSTM')
        lstm_close = [true_close[j]*(1+close_p(prediction.predict_lstm(code, (0, j))))
                      for j in range(len(trading_dates))]
        # print('计算GRU')
        # gru_close = [true_close[j]*(1+close_p(prediction.predict_gru(code, (0, j))))
        #              for j in range(len(trading_dates))]
        # print('计算RNN_tanh')
        # rnn_tanh_close = [true_close[j] * (1 + close_p(prediction.predict_rnn_tanh(code, (0, j))))
        #                   for j in range(len(trading_dates))]
        print('计算RNN_relu')
        rnn_relu_close = [true_close[j] * (1 + close_p(prediction.predict_rnn_relu(code, (0, j))))
                          for j in range(len(trading_dates))]
        print('计算ResNet18')
        rn18_close = [true_close[j]*(1+close_p(prediction.predict_resnet18(code, (0, j))))
                      for j in range(len(trading_dates))]
        print('计算ResNet34')
        rn34_close = [true_close[j]*(1+close_p(prediction.predict_resnet34(code, (0, j))))
                      for j in range(len(trading_dates))]
        print('计算ResNet50')
        rn50_close = [true_close[j]*(1+close_p(prediction.predict_resnet50(code, (0, j))))
                      for j in range(len(trading_dates))]
        print('计算ResNet101')
        rn101_close = [true_close[j]*(1+close_p(prediction.predict_resnet101(code, (0, j))))
                       for j in range(len(trading_dates))]
        print('计算ResNet152')
        rn152_close = [true_close[j]*(1+close_p(prediction.predict_resnet152(code, (0, j))))
                       for j in range(len(trading_dates))]
        print('计算DenseNet')
        densenet_close = [true_close[j]*(1+close_p(prediction.predict_densenet(code, (0, j))))
                          for j in range(len(trading_dates))]

        def sp(i, predict_close, label_name):
            plt.subplot(3, 3, i)
            plt.plot(x_r, true_close, label='真实值')
            plt.plot(x_r, predict_close, label=label_name)
            plt.ylabel('收盘价')
            plt.xticks(x_ticks, x_labels)
            plt.legend()

        sp(1, cnn_close, 'CNN模型预测值')
        sp(2, lstm_close, 'LSTM模型预测值')
        # sp(3, gru_close, 'GRU模型预测值')
        # sp(4, rnn_tanh_close, 'RNN_tanh模型预测值')
        sp(3, rnn_relu_close, 'RNN_relu模型预测值')
        sp(4, rn18_close, 'ResNet18模型预测值')
        sp(5, rn34_close, 'ResNet34模型预测值')
        sp(6, rn34_close, 'ResNet50模型预测值')
        sp(7, rn101_close, 'ResNet101模型预测值')
        sp(8, rn101_close, 'ResNet152模型预测值')
        sp(9, densenet_close, 'DenseNet模型预测值')

        plt.savefig(code+'_predict.jpg')
