import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tushare as ts

from pylab import mpl
mpl.rcParams['font.sans-serif']=['SimHei']
mpl.rcParams['axes.unicode_minus']=False
stocks={'上证指数':'sh','深证指数':'sz','沪深300':'hs300','上证50':'sz50','中小指数':'zxb','创业板':'cyb'}
stock_index=pd.DataFrame()
for stock in stocks.values():
    stock_index[stock]=ts.get_k_data(stock,ktype='D',autype='qfq',start='2005-01-01')['close']
print(stock_index.head())

tech_rets=stock_index.pct_change()[1:]
print(tech_rets.head())
print(tech_rets.describe())


print(tech_rets.describe())
print(tech_rets.mean()*100) #*100是为了算百分比

sns.jointplot('sh','sz',data=tech_rets)#算相关系数，然后以相关系数的图像输出
plt.show()

sns.jointplot('sh','sz',data=tech_rets)#算相关系数，然后以相关系数的图像输出
plt.show()
sns.pairplot(tech_rets.iloc[:,3:].dropna())#每两个相关性量进行比较，一行放三个。dropna删除空值
plt.show()
returns_fig=sns.PairGrid(tech_rets.iloc[:,3:].dropna())
returns_fig.map_upper(plt.scatter,color="purple")#upper是右上角的意思,scatter画散点图
returns_fig.map_lower(sns.kdeplot,color="cool_d")#核密度图
returns_fig.map_diag(plt.hist,bins=30)#diag是对角线，画直方图
plt.show()
