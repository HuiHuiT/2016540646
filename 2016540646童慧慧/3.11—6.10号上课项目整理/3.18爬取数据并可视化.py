import tushare as ts
import matplotlib.pyplot as plt
import seaborn as sns

stock=ts.get_hist_data('000425')
print(stock)
stock.to_csv('c:/AAAAA/try/1.csv')
stock1=ts.get_k_data('000425',autype='hfq',ktype='M',start='2012-1-1',end='2018-12-31')
stock.to_csv('c:/AAAAA/try/2.csv')
print(stock1)

sns.set_style("whitegrid")
stock['open'].plot(legend=True,figsize=(15,9))
stock['close'].plot(legend=True,figsize=(15,9))
stock['high'].plot(legend=True,figsize=(15,9))
stock['low'].plot(legend=True,figsize=(15,9))
plt.show()

y=stock1(["open"],label='open')
x=stock1(["date"],label='date')
plt.plot(x,y)
plt.title('Open stock price sets off data')
plt.show()