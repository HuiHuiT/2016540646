import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tushare as ts
from pylab import mpl#matplotlib中，frontend就是我们写的python代码，而backend就是负责显示我们代码所写图形的底层代码。
#这里使用微软雅黑字体
mpl.rcParams['font.sans-serif']=['SimHei']#画图时显示负号
mpl.rcParams['axes.unicode_minus']=False

df=ts.get_k_data('sh',ktype='D',autype='qfq',start='2006-1-1')#ktype:'D':日数据；‘m’：月数据，‘Y’:年数据
#autype:复权选择，默认‘qfq’前复权
df.index=pd.to_datetime(df.date)#把date这一列提出来作为索引
tech_rets=df.close.pct_change()[1:]#计算闭市收盘价收益率

rets=tech_rets.dropna()
print(rets.head(100))
print(rets.quantile(0.05))#置信区间,有95%的可能性一天的收益或损失不超过多少

def monte_carlo(start_price,days,mu,sigma): #monte_carlo曲线，里面要有四个变量
    dt=1/days
    price=np.zeros(days)
    price[0]=start_price#start_price是一个传过来的起始价格
    shock=np.zeros(days)
    drift=np.zeros(days)

    for x in range (1,days):#有多少天就循环多少次,从第一天开始
        shock[x]=np.random.normal(loc=mu*dt,scale=sigma*np.sqrt(dt))
        drift[x]=mu*dt
        price[x]=price[x-1]+(price[x-1]*(drift[x]+shock[x]))
    return price

#模拟次数
runs =10000#模拟10000次,出来的结果是365*10000
start_price=2641.34 #今日收盘价
days=365
mu=rets.mean()
sigma=rets.std()
simulations=np.zeros(runs)#模拟

for run in range(runs):
    simulations[run]=monte_carlo(start_price,days,mu,sigma)[days-1]
q=np.percentile(simulations,1)#以百分比形式

plt.figure(figsize=(10,6))
plt.hist(simulations,bins=100,color='grey')#模拟的有10000，每一百个做一个平均
plt.figtext(0.6,0.8,s='初始价格： %.2f'%start_price)
plt.figtext(0.6,0.7,'预期价格均值：%.2f'%simulations.mean())
plt.figtext(0.15,0.6,'q(0.99:%.2f)'%q)
plt.axvline(x=q,linewidth=6,color='r')
plt.title('经过%s天上证指数的蒙特卡洛模拟后价格分布图'%days,weight='bold')
plt.show()

#我们借用期权定价里对未来股票走势的假定来进行蒙特卡洛模拟。
from time import time
np.random.seed(2018)
t0=time()
s0=2641.34
T=1.0;
r=0.05;
sigma=rets.std()
M=50;
dt=T/M;
I=250000
s= np.zeros(I)
s[0]=s0
for t in range(1,M+1):
    z=np.random.standard_normal(I)
    s[t]=s[t-1]*np.exp((r-0.5*sigma**2)*dt+sigma*np.sqrt(dt)*z)
s_m=np.sum(s[-1])/I
tnp1=time()-t0
print('经过250000次模拟，得出1年后上证指数的预期平均收盘价为： %.2f'%s_m)# %2.f表保留两位小数，浮点型
#经过250000次模拟，得出1年以后上证指数的预期平均收盘价为：2776.85
