import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
plt.style.use('ggplot')###改变其画图风格,ggplot是R语言的风格
columns=['用户ID','购买日期','订单数','订单金额']###添加列名
df=pd.read_csv('CDNOW_master.txt',names=columns,sep='\s+')##read_csv  文档是txt，但文本比较排列工整，用read_csv转换效率和效果更好,用names等于数组，sep加分隔符对正
print(df.head(10))
print(df.describe())
print(df.info())
###购买日期不需要算均值
df['购买日期']=pd.to_datetime(df.购买日期,format='%Y%m%d')###转换类型，直接to_datatime(),%Y%m%d指年月日
df['月份']=df.购买日期.values.astype('datetime64[M]')
print(df.head())
print(df.info())
plt.rcParams['font.sans-serif']=['simHei']
plt.rcParams['axes.unicode_minus']=False      ##中文显示和避免负号错误显示
###设置图片大小
plt.figure(figsize=(15,12))
###设置子图，221是指两行两列每个位置放一个
plt.subplot(221)
df.groupby('月份')['购买日期'].count().plot(fontsize=24)
plt.title('消费次数',fontsize=24)
plt.subplot(222)
df.groupby('月份')['订单金额'].sum().plot(fontsize=24)
plt.title('消费次数',fontsize=24)
plt.subplot(223)
df.groupby('月份')['订单数'].sum().plot(fontsize=24)
plt.title('总销量',fontsize=24)
plt.subplot(224)
df.groupby('月份')['用户ID'].apply(lambda x:len(x.unique())).plot(fontsize=24)
plt.title('消费人数',fontsize=24)
plt.tight_layout()          ###布局排列
plt.show()

group_user=df.groupby('用户ID').sum()
print(group_user.describe())

group_user.query('订单金额<4000').plot.scatter(x='订单金额',y='订单数')
plt.show()
group_user.订单金额.plot.hist(bins=20)
plt.show()
group_user.query('订单金额<800')['订单金额'].plot.hist(bins=20) ###直方图只显示20条
plt.show()
##查看用户购买间隔，知道用户购买得是否频繁
order_diff=df.groupby('用户ID').apply(lambda x:x['购买日期']-x['购买日期'].shift())###shift把最短的告诉他
print(order_diff.head(10))
print(order_diff.describe())