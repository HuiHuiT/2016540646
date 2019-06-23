import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tushare as ts
from pylab import mpl
mpl.rcParams['font.sans-serif']=['SimHei']
mpl.rcParams['axes.unicode_minus']=False

s_1='000157'# 全聚德
s_2='600031'# 光明乳业

sdate='2010-01-01'# 起止日期
edate='2018-12-31'

df_s1=ts.get_k_data(s_1,start=sdate,end=edate).sort_index(axis=0,ascending=True)#排序好之后的相关性更精确 # 获取历史数据
df_s2=ts.get_k_data(s_2,start=sdate,end=edate).sort_index(axis=0,ascending=True)
df=pd.concat([df_s1.open,df_s2.open],axis=1,keys=['s1_open','s2_open'])#取出两列并命名,axis=1是要让两列对齐
df.ffill(axis=0,inplace=True)#填充缺失值，数据清洗
df.to_csv('s12.csv')

corr=df.corr(method='pearson',min_periods=1)#间隔为1，方法为皮尔森
print(corr)

df.s1_open.plot(figsize=(20,12))
df.s2_open.plot(figsize=(20,12))
plt.show()
data=pd.read_csv('directory.csv')
print(data.head())
print(data.describe())
print(data.info())
print(data.isnull().sum())
print(data[data['City'].isnull()])

def fill_na(x):
    return x
data['City']=data['City'].fillna(fill_na(data['State/Province']))#State/Province意思是联邦制的国家的洲，非联邦制的国家的省
print(data[data['City'].isnull()])
data['Country'][data['Country']=='TW']='CN'#发现数据中把台湾做为一个国家，所以把国家为台湾的赋值成中国
print(data['Country']=='TW')



