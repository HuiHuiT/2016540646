import pandas_datareader as pdr
import datetime
import numpy as np
import pandas as pd
aapl = pdr.get_data_yahoo('AAPL', start=datetime.datetime(2006, 10,
1),end=datetime.datetime(2012, 1, 1))
print(aapl.head())
# 将aapl数据框中`Adj Close`列数据赋值给变量`daily_close`
daily_close = aapl[['Adj Close']]
# 计算每日收益率
daily_pct_change = daily_close.pct_change()
# 用0填补缺失值NA
daily_pct_change.fillna(0, inplace=True)
# 查看每日收益率的前几行
print(daily_pct_change.head())
# 计算每日对数收益率
daily_log_returns = np.log(daily_close.pct_change()+1)
# 查看每日对数收益率的前几行
print(daily_log_returns.head())
# 按营业月对 `aapl` 数据进行重采样，取每月最后一项
monthly = aapl.resample('BM').apply(lambda x: x[-1])
# 计算每月的百分比变化，并输出前几行
print(monthly.pct_change().head())
# 按季度对`aapl`数据进行重采样，将均值最为每季度的数值
quarter = aapl.resample("3M").mean()
# 计算每季度的百分比变化，并输出前几行
print(quarter.pct_change().head())
'''使用 pct_change() 相当方便，但也让人困惑到底每日的百分比是如何计算的。这也是为
什么人们会使用 Pandas 的 shift() 函数来代替 pct_change()。用daily_close 除以
daily_close.shift(1)，然后再减 1，就得到了每日百分比变化。然而，使用这一函数会使计
算得到的数据框的开头存在缺失值 NA。
提示：将以下代码的结果和之前计算的每日百分比变化相比较，查看这两种方法的差异。'''
# 每日收益率
daily_pct_change = daily_close / daily_close.shift(1) - 1
# 输出 `daily_pct_change`的前几行
print(daily_pct_change.head())
'''提示：在IPython控制台中尝试使用 Pandas 的 shift() 函数计算每日对数收益率。（如果
你找不到答案，试一下这行代码：daily_log_returns_shift = np.log(daily_close /
daily_close.shift(1))）。
做为参考，每日百分比变化的计算公式为：r_t=\frac{p_t}{p_{t−1}}−1，其中p是价格，t是
时间（这里是天），r是收益率。
此外，让我们来绘制每日百分比变化 daily_pct_change 的分布。'''
# 导入 matplotlib
import matplotlib.pyplot as plt
# 绘制直方图
daily_pct_change.hist(bins=50)
# 显示图
plt.show()
# 输出daily_pct_change的统计摘要
print(daily_pct_change.describe())
'''以上分布看起来非常对称且像中心在0.00附近的正态分布。尽管如此，你还是需要对
daily_pct_change使用describe()函数，以确保正确解读了直方图。从统计摘要的结果中知
道，其均值非常接近0.00，标准差是0.02。同时，查看百分位数，了解有多少数据点落在
-0.010672, 0.001677 和 0.014306之间。
累积日收益率有助于定期确定投资价值。可以使用每日百分比变化的数值来计算累积日收益
率，只需将其加上1并计算累积的乘积。'''
# 计算累积日收益率
cum_daily_return = (1 + daily_pct_change).cumprod()
# 输出 `cum_daily_return` 的前几行
print(cum_daily_return.head())
'''注意仍旧可以使用 Matplotlib 快速绘制 cum_daily_return 的曲线；只需对其加上 plot()
函数即可，可选择使用参数 figsize 设置图片大小。'''
# 绘制累积日收益率曲线
cum_daily_return.plot(figsize=(12,8))
# 显示绘图
plt.show()
'''非常简单，不是吗？现在，如果你不想使用日回报率，而是用月回报率，那么对
cum_daily_return 使用 resample() 函数就可轻松实现月度水平的统计：'''
# 将累积日回报率转换成累积月回报率
cum_monthly_return = cum_daily_return.resample("M").mean()
# 输出 `cum_monthly_return` 的前几行
print(cum_monthly_return.head())
'''知道如何计算回报率是一项非常有用的技能，但是如果没有将其与其他股票进行比较，就
没有太大的意义。这就是为什么案例中经常会比较多只股票。在本节接下来的内容中，我们
将从雅虎财经中获取更多的数据以便能比较不同股票的日收益率。
注意，接下来的工作需要你对Pandas有更深入的理解以及知道如何使用Pandas操作数据。
让我们开始吧！首先从雅虎财经中获取更多的数据。通过创建一个 get() 函数可以轻松地实
现这一点。该函数将股票代码列表 tickers 以及开始和结束日期作为输入参数。第二个函数
data() 将 ticker 作为输入，用于获取 startdate 和 enddate 日期之间的股票数据并将其返
回。将 tickers 列表中的元素通过 map() 函数映射，获取所有股票的数据并将它们合并在一
个数据框中。
以下代码获取了 Apple、Microsoft、IBM 和 Google 的股票数据，并将它们合并在一个大
的数据框中。'''
 def get(tickers, startdate, enddate):
    def data(ticker):
        return (pdr.get_data_yahoo(ticker, start=startdate, end=enddate))
datas = map (data, tickers)
return(pd.concat(datas, keys=tickers, names=['Ticker', 'Date']))
tickers = ['AAPL', 'MSFT', 'IBM', 'GOOG']
all_data = get(tickers, datetime.datetime(2006, 10, 1), datetime.datetime(2012, 1, 1))
'''注意这一代码源自 “Mastering Pandas for Finance” 一书，并且在本教程中根据新的
标准进行了升级。还是要注意，因为开发人员仍在研究从雅虎财经API中获取数据的更持久
的修复方案，你可能还是需要导入 fix_yahoo_finance 包。可以从此处找到安装说明或者查
看这篇教程的Jupyter notebook。
现在，查看以上获取的数据：'''
print(all_data.head())
'''接下来让我们使用这个大的数据框做一些有趣的图表：'''
# 选取 `Adj Close` 这一列并变换数据框
daily_close_px = all_data[['Adj Close']].reset_index().pivot('Date', 'Ticker', 'Adj Close')
# 对`daily_close_px` 计算每日百分比变化
daily_pct_change = daily_close_px.pct_change()
# 绘制分布直方图
daily_pct_change.hist(bins=50, sharex=True, figsize=(12,8))
# 显示绘图结果
plt.show()
'''另一类有用的图是散点矩阵图。使用 pandas 库能够轻易实现它。在代码中加入
scatter_matrix() 函数就可以绘制散点矩阵图。将 daily_pct_change 作为参数传递给该函
数，在对角线上使用核密度估计（KDE）做图。另外，使用 alpha 参数设置透明度，
figsize 参数设置图片大小。'''
# 对 `daily_pct_change` 数据绘制散点矩阵图
pd.plotting.scatter_matrix(daily_pct_change, diagonal='kde',
alpha=0.1,figsize=(12,12))
# 显示绘图结果
plt.show()
'''注意如果你在本地运行代码，可能需要使用 plotting 模块来绘制散点矩阵图（例如
pd.plotting.scatter_matrix() ）。而且，最好要知道核密度估计图估算了随机变量的概率
密度函数。
恭喜你成功地完成了第一项常见的金融分析：收益率探索。现在让我们进入下一个主题：移
动窗口。
移动窗口
移动窗口指的是在一特定的时间窗口内计算数据的统计量，并在数据中按特定的间隔滑动窗
口。这样，只要窗口在时间序列的日期内不断滑动，统计量就被连续的计算。
在Pandas中有许多函数都可以计算移动窗口，比如 rolling_mean()、rolling_std()…… 在
这里查看这些函数。
然而，注意这些函数大多数将被弃用，所以最好将 rolling() 函数和 mean()、std()…… 结合
使用，当然也依赖于实际要计算的移动窗口的种类。
但是，移动窗口到底意味着什么呢？
当然确切的含义取决于对数据使用的统计量。例如，移动平均值平滑了数据中的短期波动并
突出了长期趋势。'''
# 选取调整的收盘价
adj_close_px = aapl['Adj Close']
# 计算移动均值
moving_avg = adj_close_px.rolling(window=40).mean()
# 查看后十项结果
print(moving_avg[-10:])
'''提示：在IPython控制台中尝试Pandas包中其他标准的移动窗口函数，比如
rolling_max()、 rolling_var() 或者 rolling_median()。注意也可以结合使用rolling()和
max()、 var() 或 median() 来得到相同的结果。
当然，你可能没能真正理解这一切。也许使用 Matplotlib 做一幅简单的图，可以帮助你理
解移动平均值及其实际的含义：'''
# 短期的移动窗口
aapl['42'] = adj_close_px.rolling(window=40).mean()
# 长期的移动窗口
aapl['252'] = adj_close_px.rolling(window=252).mean()
# 绘制调整的收盘价，同时包含短期和长期的移动窗口均值
aapl[['Adj Close', '42', '252']].plot()
# 显示绘图结果
plt.show()
'''波动率计算
股票的波动率衡量了股票在特定时间内收益率的变化。常常将一只股票的波动率和另一只股
票比较，以寻找风险较小的股票；或是将之与市场指数比较，来检查股票在整个市场上的波
动。一般来说，波动率越高，该股票的投资风险更大，导致人们选择投资其他股票。
对数收益率的历史移动标准差，也就是历史移动波动率，可能更令人感兴趣。也可以使用
pd.rolling_std(data, window=x) * math.sqrt(window) 来计算。'''
# 定义最小周期
min_periods = 75
# 计算波动率
vol = daily_pct_change.rolling(min_periods).std() * np.sqrt(min_periods)
# 绘制波动率曲线
vol.plot(figsize=(10, 8))
# 显示绘图结果
plt.show()
'''通过计算股票百分比变化的移动窗口标准差得到波动率。从上述代码中可以清楚地看到这
一点。
注意窗口的大小能够改变整体的结果：如果扩大窗口（也就是让min_periods变大），结果
将变得不那么有代表性。如果缩小窗口，结果将更接近于标准差。
考虑到所有这些，你会发现基于数据采样频率得到合适的窗口大小绝对是一项技能。
普通最小二乘回归
完成了上述所有计算后，你还可以使用更传统的回归分析（比如普通最小二乘回归（OLS
）），对金融数据进行更多的统计分析。
要做到这一点，你必须使用 statsmodels 库，它不仅提供了用于估算多种统计模型的类和
函数，还能让你进行统计检验以及统计的数据探索。
注意你确实可以使用Pandas实现OLS回归，但是在将来的版本中其 ols 模块将被弃用。所
以最明智的做法是使用 statsmodels 包。'''
# 导入`statsmodels` 包的 `api` 模块，设置别名 `sm`
import statsmodels.api as sm
# 获取调整的收盘价数据
all_adj_close = all_data[['Adj Close']]
# 计算对数收益率
all_returns = np.log(all_adj_close / all_adj_close.shift(1))
# 提取苹果公司数据
aapl_returns = all_returns.iloc[all_returns.index.get_level_values('Ticker') == 'AAPL']
aapl_returns.index = aapl_returns.index.droplevel('Ticker')
# 提取微软公司数据
msft_returns = all_returns.iloc[all_returns.index.get_level_values('Ticker') == 'MSFT']
msft_returns.index = msft_returns.index.droplevel('Ticker')
# 使用 aapl_returns 和 msft_returns 创建新的数据框
return_data = pd.concat([aapl_returns, msft_returns], axis=1)[1:]
return_data.columns = ['AAPL', 'MSFT']
# 增加常数项
X = sm.add_constant(return_data['AAPL'])
# 创建模型
model = sm.OLS(return_data['MSFT'],X).fit()
# 输出模型的摘要信息
print(model.summary())
'''Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly
specified.
注意在合并 AAPL 和 MSFT 收益率数据时，使用了 [1:] 切片，这样就没有缺失值来扰乱你
的模型了。
当你研究模型摘要结果时，请注意以下内容：
Dep. Variable 指出哪个变量是模型的响应。
Model 是拟合中使用的模型，在本案例中是 OLS。
另外，Method 指出模型参数是如何被计算的。在本案例中被设置为 Least Squares。
目前为止，还没有出现任何新的信息，这些都已经在上述代码中设置了。然而，也有另一些
令人感兴趣的项，比如：
观测量的数目（No. Observations）。注意你也可以使用Pandas包中的 info() 函数得到
它，只要在 IPython 控制台中运行 return_data.info()。
残差的自由度（DF Residuals）
DF Model 指模型参数的数目，注意它并不包括代码中定义的 X 的常数项。
这基本上是左边栏的内容。右边栏给出了更多关于拟合的信息。例如你可以看到：
R-squared 是决定系数。这个分数表明回归线接近真实数据点的程度。在本例中，结果是
0.281，用百分比表示该分数是 28.1% 。当决定系数是 0% 时，表明模型完全不能解释响
应数据在其均值附近的变异性。当然，当分数为 100% 时情况恰恰相反。
Adj. R-squared 分数乍看上去和 R-squared 的数值差不多。然而，该度量背后的计算是基
于观测量的数目和残差的自由度调整了 R-squared 的值。在本例中这一调整没有起到多少
作用，致使两者的结果相近。
F-statistic 衡量该拟合的显著性。通过将模型的均方误差除以残差的均方误差来计算。
Prob (F-statistic) 指得到上述 F-statistic 结果的概率，假设零假设认为它们是无关的。
Log-Likelihood 指的是似然函数的对数。
AIC 是赤池信息量准则，这一指标根据观测量的数目和模型的复杂性，对对数似然度进行了
调整。
最后，BIC 或者是贝叶斯信息准则，类似于上述 AIC，但是它用更多的参数更严格地惩罚模
型。
在模型摘要的第一部分下方，汇报了模型的每一项系数：
coef 表示系数的估计值。
std err 是系数估计的标准误差。
t 代表t统计量。该度量用于测量系数的统计显著性。
P>|t| 表示系数等于0为真的零假设。如果该值小于置信水平（通常是0.05），则表明该系
数对应的项和响应之间存在统计学上的显著关系。
最后，在模型摘要的最后部分，你将看到用于评估残差分布的其他统计检验：
Omnibus 是 Omnibus D’Angostino 检验，它为偏斜和峰度的存在提供了组合的统计检
验。
Prob(Omnibus) 将 Omnibus 度量转变成了概率。
其次，Skew 或偏斜，测量数据关于均值的对称性。
Kurtosis 给出了分布形状的指示，因为它比较了接近均值的数据量和远离均值（在尾部）的
数据量。
Durbin-Watson 是对自相关的存在的检验，Jarque-Bera (JB) 是另一项对偏斜和峰度（
kurtosis）的检验。也可以将其结果转换成概率，即为Prob (JB)。
最后，Cond. No 是对多重共线性的检验。
使用 Matplotlib 绘制普通最小二乘回归拟合的直线。'''
# 绘制 AAPL 和 MSFT 收益率的散点图
plt.plot(return_data['AAPL'], return_data['MSFT'], 'r.')
# 增加坐标轴
ax = plt.axis()
# 初始化 `x`
x = np.linspace(ax[0], ax[1] + 0.01)
# 绘制回归线
plt.plot(x, model.params[0] + model.params[1] * x, 'b', lw=2)
# 定制此图
plt.grid(True)
plt.axis('tight')
plt.xlabel('Apple Returns')
plt.ylabel('Microsoft returns')
# 输出此图
plt.show()
'''也可以使用收益率的移动相关性对结果进行核查。只需对滚动相关性的结果调用 plot() 函
数：'''
# 绘制滚动相关性
return_data['MSFT'].rolling(window=252).corr(return_data['AAPL']).plot()
# 显示该图
plt.show()