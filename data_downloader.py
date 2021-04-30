# -*- coding: utf-8 -*-
"""
@version: 3.8.3
@time: 21/4/25 23:49
@author: Yamisora
@file: data_downloader.py
"""
import baostock as bs
import pandas as pd

# 设置数据下载时间段
start_date = '2014-01-01'
end_date = '2020-12-31'
stocks = 'sz50', 'hs300', 'zz500'
indexes = 'sh.000016', 'sh.000300', 'sh.000905'
# 不建议修改路径
base_data_path = './data/'
data_path = './data/stocks/'
# 设置是否下载数据
download_stocks = True
download_indexes = True
# 设置单只股票下载 填入股票代码
download_special = ''

# 登陆系统
lg = bs.login()
# 显示登陆返回信息
if lg.error_code != '0':
    print('login respond error_code:' + lg.error_code)
    print('login respond  error_msg:' + lg.error_msg)
# 获取上证50;沪深300;中证500成分股
sz = bs.query_sz50_stocks()
hs = bs.query_hs300_stocks()
zz = bs.query_zz500_stocks()


def status(raw):
    if raw.error_code != '0':
        print('query_history_k_data_plus respond error_code:' + rs.error_code)
        print('query_history_k_data_plus respond  error_msg:' + rs.error_msg)


def data_save(raw, csv_name, path=data_path):
    data = []
    while (raw.error_code == '0') & raw.next():
        # 获取一条记录，将记录合并在一起
        data.append(raw.get_row_data())
    df = pd.DataFrame(data, columns=raw.fields)
    # 结果集输出到csv文件
    df.to_csv("{}{}.csv".format(path, csv_name), index=False)
    print(df)
    return df


if download_stocks:
    for rs, name in zip((sz, hs, zz), stocks):
        # 获取股票名称与代码
        result = data_save(rs, name+'_stocks', base_data_path)

        for code in result['code']:
            stock_data = []
            print('Downloading ' + code)
            rs = bs.query_history_k_data_plus(code,
                                              "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,"
                                              "tradestatus,pctChg,isST",
                                              start_date=start_date, end_date=end_date,
                                              frequency="d", adjustflag="3")
            status(rs)
            data_save(rs, code)

if download_indexes:
    for index in indexes:
        rs = bs.query_history_k_data_plus(index,
                                          "date,code,open,high,low,close,preclose,volume,amount,pctChg",
                                          start_date=start_date, end_date=end_date, frequency="d")
        status(rs)
        data_save(rs, index)

if download_special:
    rs = bs.query_history_k_data_plus(download_special,
                                      "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,"
                                      "tradestatus,pctChg,isST",
                                      start_date=start_date, end_date=end_date, frequency="d")
    status(rs)
    data_save(rs, download_special)

# 登出系统
bs.logout()
