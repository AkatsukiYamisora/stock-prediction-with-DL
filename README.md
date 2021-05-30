# stock-prediction-with-DL

## 深度学习与股票分析预测

### 先开一个坑，慢慢做

data_downloader.py

通过baostock下载上证50，沪深300，中证500的日线数据

backtest.py

回测模块，根据调仓周期按开盘价买入卖出（暂定），计算收益，与指数比较并绘图

strategy.py

按价值因子（选择账面市值比BM）进行选股（效果差，由于BM在大市值公司的效果不佳）

用CNN预测模型进行选股

用动量因子(MF)进行选股

用换手率因子(TR)进行选股（思路参考自Liu et al.(2019) Size an value in China, 修改了数据选取)

其余策略待更新

prediction.py

使用CNN对股票数据进行预测

使用多种RNN对股票数据进行预测（待更新）
