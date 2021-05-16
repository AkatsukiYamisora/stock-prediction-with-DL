# stock-prediction-with-DL

## 深度学习与股票分析预测

### 先开一个坑，慢慢做

data_downloader.py

通过baostock下载上证50，沪深300，中证500的日线数据

backtest.py

回测模块，根据调仓周期按开盘价买入卖出（暂定），计算收益，与指数比较并绘图

strategy.py

按Fama-French三因子模型中的账面市值比因子进行选股（暂定）

其余策略待更新

prediction.py

使用多种RNN对股票数据进行预测（待更新）