# -*- coding: utf-8 -*-
"""
@version: 3.8.3
@time: 21/4/25 23:49
@author: Yamisora
@file: data_downloader.py
"""
import os
import baostock as bs
import tushare as ts
import pandas as pd
import numpy as np
from tqdm import tqdm
import time

# 设置数据下载时间段
data_start_date = '2019-01-01'
data_end_date = '2021-06-02'
train_data_start_date = '2015-01-01'
train_data_end_date = '2018-12-31'
# 存储路径
base_data_path = './data/'
data_path = './data/stocks/'
train_data_path = './data/train_data/'

# 使用tushare数据api
apikey = ''
if os.path.exists('tushare_apikey.txt'):
    with open('tushare_apikey.txt', 'r') as f:
        apikey = f.read()
pro = ts.pro_api(apikey)

# 设置是否下载数据
download_stocks = True
download_indexes = True
# 下载个股参数列表
fields = "date,code,open,high,low,close,preclose,volume," \
         "amount,turn,peTTM,psTTM,pcfNcfTTM,pbMRQ"
ts_basic_fields = "trade_date,total_share,float_share,free_share,total_mv,circ_mv"

# baostock数据下载
lg = bs.login()
# 显示登陆返回信息
if lg.error_code != '0':
    print('login respond error_code:' + lg.error_code)
    print('login respond  error_msg:' + lg.error_msg)


def status(raw):
    if raw.error_code != '0':
        print('query_history_k_data_plus respond error_code:' + rs.error_code)
        print('query_history_k_data_plus respond  error_msg:' + rs.error_msg)


def data_load(raw):
    """读取baostock数据"""
    data = []
    while (raw.error_code == '0') & raw.next():
        # 获取一条记录，将记录合并在一起
        data.append(raw.get_row_data())
    df = pd.DataFrame(data, columns=raw.fields)
    return df


def ts_c(bs_stock_code: str) -> str:
    """转换baostock股票代码为tushare股票代码"""
    return bs_stock_code[3:]+'.'+bs_stock_code[0:2].upper()


def ts_d(date):
    """转换baostock日期为tushare日期"""
    return date.replace('-', '')


def bs_d(ts_date_series):
    """转换tushare日期序列为baostock日期序列"""
    date_series = []
    for ts_date in ts_date_series:
        date = str(ts_date)
        date_series.append(date[:4] + '-' + date[4:6] + '-' + date[6:])
    return date_series


if download_stocks:
    rs = bs.query_hs300_stocks()
    # 获取股票名称与代码
    result = data_load(rs)
    result.to_csv(base_data_path+'hs300_stocks.csv')
    for code in tqdm(result['code']):
        for start_date, end_date, path in ((data_start_date, data_end_date, data_path),
                                           (train_data_start_date, train_data_end_date, train_data_path)):
            rs = bs.query_history_k_data_plus(code, fields,
                                              start_date=start_date, end_date=end_date,
                                              frequency="d", adjustflag="3")
            status(rs)

            # baostock日线数据
            df1 = data_load(rs)
            df1.set_index('date', inplace=True)

            # tushare指标数据
            df2 = pro.daily_basic(ts_code=ts_c(code), start_date=ts_d(start_date),
                                  end_date=ts_d(end_date), fields=ts_basic_fields)
            df2['trade_date'] = bs_d(df2['trade_date'])
            df2.set_index('trade_date', inplace=True)

            # 按index横向拼接
            df1 = df1.join(df2)
            # 去除无效空值,补0
            df1.replace(to_replace=r'^\s*$', value=np.nan, regex=True, inplace=True)
            df1.fillna(0, inplace=True)

            df1.to_csv(path+code+'.csv')
            del df1, df2
        time.sleep(0.2)  # 防止tushare调用到分钟上限


if download_indexes:
    rs = bs.query_history_k_data_plus('sh.000300',
                                      "date,code,open,high,low,close,preclose,volume,amount,pctChg",
                                      start_date=data_start_date, end_date=data_end_date, frequency="d")
    status(rs)
    data_load(rs).to_csv(data_path+'sh.000300.csv')

# 登出系统
bs.logout()
