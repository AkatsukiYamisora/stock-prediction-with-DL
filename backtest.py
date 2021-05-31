# -*- coding: utf-8 -*-
"""
@version: 3.8.3
@time: 21/4/26 10:52
@author: Yamisora
@file: backtest.py
"""
import matplotlib.pyplot as plt
import time

from strategy import *


class Backtest:
    def __init__(self, index=1, start_cash=300000, fee=0.0003):
        """
        index: 0:上证 50  1:沪深 300  2:中证 500
        """
        # index选择指数组合
        self.index_name = 'sz50', 'hs300', 'zz500'
        self.indexes = 'sh.000016', 'sh.000300', 'sh.000905'
        # 存储路径
        self.base_data_path = './data/'
        self.data_path = './data/stocks/'
        # 默认万三手续费
        self.fee = fee
        # 指数组合内股票名称,代码数据
        self.stocks = pd.read_csv('{}{}_stocks.csv'.format(self.base_data_path, self.index_name[index]))
        self.stocks_codes = self.stocks['code']
        # 指数日线数据
        self.index = pd.read_csv('{}{}.csv'.format(self.data_path, self.indexes[index]))
        # 交易日str序列
        self.trading_dates = self.index['date']
        # 今日名称str与序列序号int
        self.today = (self.trading_dates[0], 0)
        # 初始资金
        self.start_cash = start_cash
        # 每日可用资金
        self.cash = pd.DataFrame(self.trading_dates)
        self.cash['cash'] = start_cash
        self.cash = self.cash.set_index('date')
        # 每日持股
        self.position = pd.DataFrame()
        self.position['date'] = self.trading_dates
        for stock_code in self.stocks_codes:
            self.position[stock_code] = 0
        self.position = self.position.set_index('date')

    def stocks_data(self, stock_codes, trading_date: str) -> pd.DataFrame():
        """
        读取指定交易日内，一系列股票日线数据
        """
        # 空输入情况
        if stock_codes.empty:
            return pd.DataFrame(None)
        stocks = pd.DataFrame()
        for stock_code in stock_codes:
            # 读取每只股票数据
            data = pd.read_csv('{}{}.csv'.format(self.data_path, stock_code), index_col='date')
            if trading_date in data.index:
                # 读入指定日期数据
                stocks = stocks.append(data.loc[trading_date], ignore_index=True)
        stocks = stocks.set_index('code')
        return stocks

    def buy(self, stocks_codes) -> bool:
        """
        全仓买入trading date的所有stock codes股票
        """
        trading_date = self.today[0]
        # 读入当前交易日内所有待购买股票日线数据
        stocks = self.stocks_data(stocks_codes, trading_date)
        # 空输入情况
        if stocks.empty:
            return False
        # 预计购买股票数
        n = len(stocks_codes)
        # 每只股票可用购买资金
        single = self.cash.loc[trading_date, 'cash'] // n
        for stock_code in stocks_codes:
            # 每只股票开盘价作为买入价
            open_price = stocks.loc[stock_code, 'open']
            # 计算除去手续费后可购买份额(整百)
            quantity = ((single / (1+self.fee)) / open_price) // 100
            # 买入并修改当日现金余额
            self.position.loc[trading_date, stock_code] += quantity * 100
            self.cash.loc[trading_date, 'cash'] -= open_price * quantity * 100
        del stocks
        return True

    def sell(self, stocks_codes):
        """
        空仓trading date的除去stock codes外股票
        """
        trading_date = self.today[0]
        # 读入当前交易日内所有股票日线数据
        stocks = self.stocks_data(self.stocks_codes, trading_date)
        # 初始化卖出金额
        cash = 0
        # 卖出股票数量
        sell_num = 0
        # 卖出所有不继续持仓的股票并删除欲购买列表中已持仓的股票
        for stock_code in self.stocks_codes:
            position = self.position.loc[trading_date, stock_code]
            if position != 0:
                if stock_code not in stocks_codes:
                    # 卖出
                    cash += position * stocks.loc[stock_code, 'open'] * (1-self.fee)
                    self.position.loc[trading_date, stock_code] = 0
                    sell_num += 1
                else:
                    # 去除已选股票中已持仓股票
                    stocks_codes = stocks_codes.drop(stock_code)
        # 当日现金余额增加卖出金额
        self.cash.loc[trading_date, 'cash'] += cash
        del stocks
        return stocks_codes, sell_num

    def next_day(self) -> str:
        """
        返回下一个交易日str
        延续前一天cash数量
        """
        # 获取当日现金余额、持仓、日期序号
        cash = self.cash.loc[self.today[0], 'cash']
        position = self.position.loc[self.today[0]]
        today = self.today[1]
        if today < len(self.trading_dates) - 1:
            # 更新下一日现金余额、持仓、日期元组
            self.today = (self.trading_dates[today+1], today+1)
            self.cash.loc[self.today[0], 'cash'] = cash
            self.position.loc[self.today[0]] = position
            return self.today[0]

    def calculate(self):
        """
        计算收益折线
        """
        # 总收益百分比
        basic_position = self.start_cash
        position = []
        for i, trading_date in enumerate(self.trading_dates):
            # 每隔5交易日计算持仓
            if i % 5 != 0:
                continue
            print('当前计算交易日: '+trading_date, end='\r')
            # 读入当前交易日内所有股票日线信息
            stock_data = self.stocks_data(self.stocks_codes, trading_date)
            # 初始化当前拥有总金额为现金金额
            daily_position = self.cash.loc[trading_date, 'cash']
            for stock_code in self.stocks_codes:
                # 确认每只股票持股数
                quantity = self.position.loc[trading_date, stock_code]
                if quantity != 0:
                    daily_position += quantity * stock_data.loc[stock_code, 'close']
            position.append(daily_position)
        position = pd.Series(position)
        position /= basic_position
        return position


if __name__ == '__main__':
    start_time = time.time()
    sz50, hs300, zz500 = 0, 1, 2
    bt1 = Backtest(index=hs300, start_cash=10000000, fee=0.0003)
    bt2 = Backtest(index=hs300, start_cash=10000000, fee=0.0003)
    bt3 = Backtest(index=hs300, start_cash=10000000, fee=0.0003)
    bt4 = Backtest(index=hs300, start_cash=10000000, fee=0.0003)
    strategy = Strategy(index=hs300)
    for date_key, date in bt1.trading_dates.items():
        # 每10交易日调仓一次
        if date_key % 10 == 0:
            print('当前交易日: '+date)
            chosen1 = strategy.choose_by_bm(bt1.today, 90)
            chosen2 = strategy.choose_by_cnn(bt2.today, 90)
            chosen3 = strategy.choose_by_mf(bt3.today, 90)
            chosen4 = strategy.choose_by_tr(bt4.today, 90)
            to_buy1, sell1 = bt1.sell(chosen1)
            to_buy2, sell2 = bt2.sell(chosen2)
            to_buy3, sell3 = bt3.sell(chosen3)
            to_buy4, sell4 = bt4.sell(chosen4)
            print('价值因子(BM)选股模型卖出', sell1, '只股票')
            print('价值因子(BM)选股模型买入', len(to_buy1), '只股票')
            print('CNN选股模型卖出', sell2, '只股票')
            print('CNN选股模型买入', len(to_buy2), '只股票')
            print('动量因子(MF)选股模型卖出', sell3, '只股票')
            print('动量因子(MF)选股模型买入', len(to_buy3), '只股票')
            print('换手率因子(TR)选股模型卖出', sell4, '只股票')
            print('换手率因子(TR)选股模型买入', len(to_buy4), '只股票')
            bt1.buy(to_buy1)
            bt2.buy(to_buy2)
            bt3.buy(to_buy3)
            bt4.buy(to_buy4)
        bt1.next_day()
        bt2.next_day()
        bt3.next_day()
        bt4.next_day()
    mid_time = time.time()
    span = mid_time - start_time
    print('回测模拟交易用时 {} 分 {} 秒'.format(int(span // 60), span % 60))
    print('\n计算价值因子(BM)选股模型收益中')
    bm_position = bt1.calculate()
    print('\n计算CNN选股模型收益中')
    cnn_position = bt2.calculate()
    print('\n计算动量因子(MF)选股模型收益中')
    mf_position = bt3.calculate()
    print('\n计算换手率因子(TR)选股模型收益中')
    tr_position = bt4.calculate()
    # 指数收益百分比
    basic_index_price = bt1.index['close'][0]
    index_price = bt1.index['close'][::5] / basic_index_price
    x = range(0, len(bt1.trading_dates), 5)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=[10, 7], dpi=300)
    plt.subplot(211)
    plt.plot(x, index_price, label='沪深300指数收益率')
    plt.plot(x, bm_position, label='价值因子(BM)选股模型持仓收益率')
    plt.plot(x, cnn_position, label='CNN选股模型持仓收益率')
    plt.plot(x, mf_position, label='动量因子(MF)选股模型持仓收益率')
    plt.plot(x, tr_position, label='换手率因子(TR)选股模型持仓收益率')
    plt.ylabel('收益率/%')
    x_ticks = list(x[::len(x)//9])
    x_ticks.append(x[-1])
    x_labels = [bt1.trading_dates[i] for i in x_ticks]
    plt.xticks(x_ticks, x_labels)
    plt.legend()
    plt.subplot(212)
    plt.plot(x, [0]*len(x), label='基准市场收益率(沪深300)', )
    plt.plot(x, bm_position.values-index_price.values, label='价值因子(BM)选股模型持仓超额收益率')
    plt.plot(x, cnn_position.values-index_price.values, label='CNN选股模型持仓超额收益率')
    plt.plot(x, mf_position.values-index_price.values, label='动量因子(MF)选股模型持仓超额收益率')
    plt.plot(x, tr_position.values-index_price.values, label='换手率因子(TR)选股模型持仓超额收益率')
    plt.ylabel('超额收益率/%')
    plt.xticks(x_ticks, x_labels)
    plt.legend()
    plt.savefig('result.jpg')
    end_time = time.time()
    span = end_time - mid_time
    print('计算持仓收益用时 {} 分 {} 秒'.format(int(span // 60), span % 60))
    span = end_time - start_time
    print('总计用时 {} 分 {:.2f} 秒'.format(int(span // 60), span % 60))
    plt.show()
