# -*- coding: utf-8 -*-
"""
@version: 3.8.3
@time: 21/4/25 23:49
@author: Yamisora
@file: data_downloader.py
"""
import baostock as bs
import pandas as pd

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

for rs, name in zip([sz, hs, zz], ['sz50', 'hs300', 'zz500']):
    rs_stocks = []
    while (rs.error_code == '0') & rs.next():
        # 获取一条记录，将记录合并在一起
        rs_stocks.append(rs.get_row_data())
    result = pd.DataFrame(rs_stocks, columns=rs.fields)
    # 结果集输出到csv文件
    result.to_csv("./data/{}_stocks.csv".format(name), encoding="gbk", index=False)
    print(result)

    data_list = []
    for code in result['code']:
        print('Downloading ' + code)
        rs = bs.query_history_k_data_plus(code,
                                          "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,"
                                          "tradestatus,pctChg,isST",
                                          start_date='2015-01-01', end_date='2020-12-31',
                                          frequency="d", adjustflag="3")
        if rs.error_code != '0':
            print('query_history_k_data_plus respond error_code:' + rs.error_code)
            print('query_history_k_data_plus respond  error_msg:' + rs.error_msg)

        # 打印结果集
        while (rs.error_code == '0') & rs.next():
            # 获取一条记录，将记录合并在一起
            data_list.append(rs.get_row_data())
    result2 = pd.DataFrame(data_list, columns=rs.fields)

    # 结果集输出到csv文件
    result2.to_csv("./data/{}_history_data.csv".format(name), index=False)
    print(result2)

# 登出系统
bs.logout()
