# stock-prediction-with-DL/深度学习与股票分析预测

![GitHub](https://img.shields.io/github/license/AkatsukiYamisora/stock-prediction-with-DL)
![Python Version](https://img.shields.io/badge/python-3.8+-blue)
![Pytorch Version](https://img.shields.io/badge/torch-1.7.0+-blue)
![GitHub watchers](https://img.shields.io/github/watchers/AkatsukiYamisora/stock-prediction-with-DL?style=social)
![GitHub Repo stars](https://img.shields.io/github/stars/AkatsukiYamisora/stock-prediction-with-DL?style=social)

先开一个坑，慢慢做

## data_downloader.py

通过baostock下载上证50，沪深300，中证500的日线数据

## backtest.py

回测模块，根据调仓周期，通过模型选股，按开盘价买入卖出（暂定），计算收益，与指数比较并绘图

## strategy.py

按价值因子（选择账面市值比BM）进行选股（效果差，由于BM在大市值公司的效果不佳）

用CNN预测模型进行选股

用动量因子(MF)进行选股

用换手率因子(TR)进行选股（思路参考自Liu et al.(2019) Size and value in China, 修改了数据选取)

其余策略待更新

## prediction.py

使用CNN对股票数据进行预测

使用多种RNN对股票数据进行预测（待更新）

## /data/sz50_stocks.csv & hs300_stocks.csv & zz500_stocks.csv

存放指数股票列表数据

## /data/stocks/sh.000016.csv & sh.000300.csv & sh.000905.csv

存放指数日线数据

## /data/stocks/*.csv

存放回测个股日线数据

## /data/train_data/*.csv

存放训练集个股日线数据

## /data/*.pkl

存放训练集数据与训练完成的模型数据

## result.jpg

回测结果展示图片
