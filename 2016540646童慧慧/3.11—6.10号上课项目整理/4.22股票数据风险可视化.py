import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tushare as ts
from pylab import mpl
mpl.rcParams['font.sans-serif']=['SimHei']
mpl.rcParams['axes.unicode_minus']=False

stocks={'上证指数':'sh','深证指数':'sz','沪深300':'hs300','上证50':'sz50','中小指数版':'zxb','创业指数板':'cyb'}
def return_risk(stocks,startdate='2006-1-1'):#返回风险运算,stocks是股票
    close=pd.DataFrame()
    for stock in stocks.values():#进行处理，把stock里的值取出来
        close[stock]=ts.get_k_data(stock,ktype='D',autype='qfq',start=startdate)['close']#把每个股票取出来，
        # 每取一个作为一列,用tushare取，按天取,后面加这个['close']是表明只要close收盘价放到close里面，不然close放不下，autype=‘qfq'表拼音
    tech_rets=close.pct_change()[1:]#计算收益
    rets=tech_rets.dropna()#删除空值不然下面计算平均值会报错
    ret_mean =rets.mean()*100#百分化
    ret_std=rets.std()*100
    return ret_mean,ret_std

def plot_return_risk():#定义一个模块，可视化数据,进行风险可视化
    ret,vol=return_risk(stocks)#之前返回了两个变量，所以这里面就要接受两个变量
    color=np.array([0.18,0.96,0.75,0.3,0.9,0.5])#定义一个颜色数组
    plt.scatter(ret,vol,marker='o',c=color,s=500,cmap=plt.get_cmap('Spectral'))
    plt.xlabel("日收益率均值%")
    plt.ylabel("标准差%")
    for label,x,y in zip(stocks.keys(),ret,vol):
        plt.annotate(label,xy=(x,y),xytext=(20,20),textcoords="offset points",ha="right",va="bottom",
                     bbox=dict(boxstyle='round,pad=0.5',fc='yellow',alpha=0.5),arrowprops=dict(arrowstyle="->",connectionstyle="arc3,rad=0"))
        #annotate为标注文字，s 为注释文本内容 xy 为被注释的坐标点 xytext 为注释文字的坐标位置 arrowprops  #箭头参数,参数类型为字典dict，bbox给标题增加外框boxstyle方框外形
plot_return_risk()

stocks={'中国平安':'601318','格力电器':'000651','徐工机械':'000425','招商银行':'600036','恒生电子':'600570','贵州茅台':'600519'}
startdate='2018-1-1'
plot_return_risk()
plt.show()
df=ts.get_k_data('sh',ktype='D',autype='qfq',start='2006-1-1')
df.index=pd.to_datetime(df.data)
tech_rets=df.close.pct_change()[1:]
rets=tech_rets.dropna()
print(rets.head(100))
print(rets.quantile(0.05))