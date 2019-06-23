#1.时间序列有什么特别之处(时间序列的间隔是固定的）
#2.
# 3.如何评价时间序列它的稳定性(稳定的才能做预测）
#4.如果序列是不稳定的怎么让他平稳
#5.预测（预测有一个常用的方法，回归分析，时间序列的预测存在季节性,
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams#这个包可以让我们去设图形大小
rcParams['figure.figsize']=15,6

# data=pd.read_csv('AirPassengers.csv')
# print(data.head())
# print('\n Data Type:')
# print(data.dtypes)
dateparse=lambda dates:pd.datetime.strptime(dates,'%Y-%m')#利用pandas里的这个包转换数据类型
data=pd.read_csv('AirPassengers.csv',parse_dates=['Month'],index_col='Month',date_parser=dateparse)#根据月进行解析，设月为索引,具体怎么转换用之前定义的dateparse方法
print(data.head())
data.index

ts=data['#Passengers']
print(ts.head(10))
print(ts['1949'])#这样会把所有1949年的数据显示出来

#判断时间序列的稳定性
#如果这个序列的平均值、方差不变，就是稳定的，时间序列随着时间的推移，他的行为是固定的，就猜测他的未来会有相同的行为
#画图看规律
plt.plot(ts)
plt.show()

from statsmodels.tsa.stattools import adfuller#引入一个模型，用他的工具来帮我们判断
def test_stationarity(timeseries):
    rolmean=pd.Series(timeseries).rolling(window=12).mean()#每12个算一个平均值
    rolstd=pd.Series(timeseries).rolling(window=12).std()#每12个算一个fang差
    #保存在这两个变量里，用这两个量去衡量标准性
    orig=plt.plot(timeseries,color='blue',label='Original')#用蓝色表原始数据
    mean=plt.plot(rolmean,color='red',label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    print('Results of Dickey_Fuller Test:')
    dftest=adfuller(timeseries,autolag='AIC')
    dfoutput=pd.Series(dftest[0:4],index=['Test Statisic','p-value','#LagsUsed','Number of Obersercation Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value(%s)'%key]=value
    print(dfoutput)
test_stationarity(ts)
#红线是平均值，黑线是标准差       1% 5%10%指的是置信区间，只要有趋势就是不稳定的

ts_log=np.log(ts)