import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tushare as ts

from pylab import mpl
mpl.rcParams['font.sans-serif']=['SimHei']
mpl.rcParams['axes.unicode_minus']=False

sh=ts.get_k_data(code='sh',ktype='D',autype='qfq',start='1990-12-20')
print(sh.head(10))
sh.index=pd.to_datetime(sh.date)#强制转换，将date转换成日期#
sh['close'].plot(figsize=(12,6))
plt.title('Trend chart for SH stocks from 1990 to NOW')
plt.xlabel('date')
plt.show()
print(sh.describe().round(2))#小数部分保留两位#
print(sh.count())#样本的个数#
print(sh.close.mean())#均值#
print(sh.open.std())#标准差

sh.loc["2007-01-01":]['volume'].plot(figsize=(12,6))#索引是日期所以用日期切片
plt.title('from 01/01/2016')
plt.show()
ma_day=[20,52,252]
for ma in ma_day:
    column_name="%sday mean"%(str(ma))#用%(str(ma))代替%s
    sh[column_name]=sh['close'].rolling(ma).mean()#把close取出来，当ma=20时，每20一段取平均值
sh.tail(3)
sh.loc['2007-01-01':][["close","20day mean","52day mean","252day mean"]].plot(figsize=(12,6))
plt.title('2007 to now the trend of CHN Stock Market')
plt.xlabel('Date')
plt.show()

sh["Daily profit"]=sh["close"].pct_change()
sh["Daily profit"].loc['2006-01-01':].plot(figsize=(12,6))
plt.xlabel('Date')
plt.ylabel('Daily profit')
plt.title('From 2006 to now daily profit')
plt.show()

sh["Daily profit"].loc['2006-01-01':].plot(figsize=(12,4),marker="o",linestyle="--",color="b")
plt.xlabel('Date')
plt.show()

stocks={'上证指数':'sh','深证指数':'sz','沪深300':'hs300','上证50':'sz50','中小指数':'zxb','创业板':'cyb'}
stock_index=pd.DataFrame()
for stock in stocks.values():
    stock_index[stock]=ts.get_k_data(stock,ktype='D',autype='qfq',start='2005-01-01')['close']





















