import tushare as ts
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=ts.get_hist_data('000001',start='2007-01-01',end='2019-06-03')
print(df.head(10))

sz=df.sort_index(axis=0,ascending=True)
sz_return=sz[['p_change']]
train=sz_return[0:255]
test=sz_return[255:]
plt.figure(figsize=(10,5))
train['p_change'].plot()
plt.legend(loc='best')
plt.show()

plt.figure(figsize=(10,5))
test['p_change'].plot(c='r')
plt.legend(loc='best')
plt.show()
train.index=pd.to_datetime(train.index)
test.index=pd.to_datetime(test.index)
dd=np.asarray(train.p_change)#np.asarray,把它转换成向量
y_hat=test.copy()
y_hat['naive']=dd[len(dd)-1]#长度-1即把最后一个分量赋给naive，把训练集里的最后一个值最为预测值
plt.figure(figsize=(12,8))
plt.plot(train.index,train['p_change'],label='Train')#横坐标是时间，纵坐标是收益率
plt.plot(test.index,test['p_change'],label='Test')
plt.plot(y_hat.index,y_hat['naive'],label='Naive Forcast')#用别人提出来的方法做预测
plt.legend(loc='best')
plt.title('Naive Forcast')
plt.show()

#改进预测方法
from sklearn.metrics import mean_squared_error#方差计算
from math import sqrt#开平方运算
rms=sqrt(mean_squared_error(test.p_change,y_hat.naive))#均方差，衡量两个数之间的差异程度，越小越好
print(rms)

y_hat_avg=test.copy()
y_hat_avg['avg_forcast']=train['p_change'].mean()
plt.figure(figsize=(12,8))
plt.plot(train.index,train['p_change'],label='Train')#横坐标是时间，纵坐标是收益率
plt.plot(test.index,test['p_change'],label='Test')
plt.plot(y_hat_avg['avg_forcast'],label='Average Forcast')#用别人提出来的方法做预测
plt.legend(loc='best')
plt.show()
rms=sqrt(mean_squared_error(test.p_change,y_hat_avg.avg_forcast))#均方差，衡量两个数之间的差异程度，越小越好
print(rms)

y_hat_avg=test.copy()
y_hat_avg['moving_avg_forcast']=train['p_change'].rolling(7).mean().iloc[-1]#用前面30个值做预测，是一个动态的预测,相对位置的-1
plt.figure(figsize=(12,8))
plt.plot(train.index,train['p_change'],label='Train')#横坐标是时间，纵坐标是收益率
plt.plot(test.index,test['p_change'],label='Test')
plt.plot(y_hat_avg['moving_avg_forcast'],label='Moving Average Forcast')#用别人提出来的方法做预测
plt.legend(loc='best')
plt.show()#出来的线还是一个直线，说明他前几天的动荡较小
rms=sqrt(mean_squared_error(test.p_change,y_hat_avg.moving_avg_forcast))#均方差，衡量两个数之间的差异程度，越小越好
print(rms)#不断改rolling值，发现rolling=7是准确率最高的