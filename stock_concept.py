# -*- coding: utf-8 -*-
"""
@version: 3.8.3
@author: Yamisora
@file: stock_concept.py
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd


base_data_path = './data/'
url = 'http://stockpage.10jqka.com.cn/'
header = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/96.0.4664.55 Safari/537.36 Edg/96.0.1054.34 '
}
hs300 = pd.read_csv(base_data_path + 'hs300_stocks.csv')
codes = [a[3:]+'/' for a in hs300['code']]
hs300['concept'] = ''

for i in range(len(hs300)):
    print('正在爬取'+codes[i]+'概念数据')
    response = requests.get(url+codes[i], headers=header)
    soup = BeautifulSoup(response.text, 'lxml')
    company_details = soup.find('dl', class_='company_details')
    details = company_details.find_all('dd')
    concept = details[1]['title']
    hs300.loc[i, 'concept'] = concept

hs300.to_csv(base_data_path+'hs300_stocks.csv')
