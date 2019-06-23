import urllib.request
import requests
from requests.exceptions import RequestException
import re
from bs4 import BeautifulSoup
import json
import time
from lxml import etree #解析的#

# -----------------------------------------------------------------------------

def get_one_page(url):
     try:
         headers = {
             'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.181 Safari/537.36'}
         # 不加headers爬不了
         response = requests.get(url, headers=headers)
         if response.status_code == 200:
             return response.text
         else:
             return None
     except RequestException:
         return None


 # 1 用正则提取内容
def parse_one_page(html):
     pattern = re.compile(
         '<dd>.*?board-index.*?>(\d+)</i>.*?data-src="(.*?)".*?name"><a.*?>(.*?)</a>.*?star">(.*?)</p>.*?releasetime">(.*?)</p.*?integer">(.*?)</i>.*?fraction">(.*?)</i>.*?</dd>', re.S)
     # re.S表示匹配任意字符，如果不加.无法匹配换行符
     items = re.findall(pattern, html)
     # print(items)
     for item in items:
         yield {
             'index': item[0],
             'thumb': get_thumb(item[1]),
             'name': item[2],
             'star': item[3].strip()[3:],
             # 'time': item[4].strip()[5:],
             # 用函数分别提取time里的日期和地区
             'time': get_release_time(item[4].strip()[5:]),
             'area': get_release_area(item[4].strip()[5:]),
             'score': item[5].strip() + item[6].strip()
         }

 # 2 用lxml结合xpath提取内容
def parse_one_page2(html):
     parse = etree.HTML(html)
     items = parse.xpath('//*[@id="app"]//div//dd')
     # 完整的是//*[@id="app"]/div/div/div[1]/dl/dd
     # print(type(items))
     # *代表匹配所有节点，@表示属性
     # 第一个电影是dd[1],要提取页面所有电影则去掉[1]
     # xpath://*[@id="app"]/div/div/div[1]/dl/dd[1]
     # lst = []
     for item in items:
         yield{
             'index': item.xpath('./i/text()')[0],
             #./i/text()前面的点表示从items节点开始
             #/text()提取文本
             'thumb': get_thumb(str(item.xpath('./a/img[2]/@data-src')[0].strip())),
             # 'thumb': 要在network中定位，在elements里会写成@src而不是@data-src，从而会报list index out of range错误。
             'name': item.xpath('./a/@title')[0],
             'star': item.xpath('.//p[@class = "star"]/text()')[0].strip(),
             'time': get_release_time(item.xpath(
                 './/p[@class = "releasetime"]/text()')[0].strip()[5:]),
             'area': get_release_area(item.xpath(
                 './/p[@class = "releasetime"]/text()')[0].strip()[5:]),
             'score' : item.xpath('.//p[@class = "score"]/i[1]/text()')[0] + \
             item.xpath('.//p[@class = "score"]/i[2]/text()')[0]
         }


 # 3 用beautifulsoup + css选择器提取
def parse_one_page3(html):
     soup = BeautifulSoup(html, 'lxml')
     # print(content)
     # print(type(content))
     # print('------------')
     items = range(10)
     for item in items:
         yield{

             'index': soup.select('dd i.board-index')[item].string,
             # iclass节点完整地为'board-index board-index-1',写board-inde即可
             'thumb': get_thumb(soup.select('a > img.board-img')[item]["data-src"]),
             # 表示a节点下面的class = board-img的img节点,注意浏览器eelement里面是src节点，而network里面是data-src节点，要用这个才能正确返回值

             'name': soup.select('.name a')[item].string,
             'star': soup.select('.star')[item].string.strip()[3:],
             'time': get_release_time(soup.select('.releasetime')[item].string.strip()[5:]),
             'area': get_release_area(soup.select('.releasetime')[item].string.strip()[5:]),
             'score': soup.select('.integer')[item].string + soup.select('.fraction')[item].string
         }


 # 4 用beautifulsoup + find_all提取
def parse_one_page4(html):
     soup = BeautifulSoup(html,'lxml')
     items = range(10)
     for item in items:
         yield{

            'index': soup.find_all(class_='board-index')[item].string,
            'thumb': soup.find_all(class_ = 'board-img')[item].attrs['data-src'],
            # 用.get('data-src')获取图片src链接，或者用attrs['data-src']
            'name': soup.find_all(name = 'p',attrs = {'class' : 'name'})[item].string,
            'star': soup.find_all(name = 'p',attrs = {'class':'star'})[item].string.strip()[3:],
            'time': get_release_time(soup.find_all(class_ ='releasetime')[item].string.strip()[5:]),
            'area': get_release_time(soup.find_all(class_ ='releasetime')[item].string.strip()[5:]),
            'score':soup.find_all(name = 'i',attrs = {'class':'integer'})[item].string.strip() + soup.find_all(name = 'i',attrs = {'class':'fraction'})[item].string.strip()
        }

# -----------------------------------------------------------------------------

# 提取时间函数
def get_release_time(data):
    pattern = re.compile(r'(.*?)(\(|$)')
    items = re.search(pattern, data)
    if items is None:
        return '未知'
    return items.group(1)  # 返回匹配到的第一个括号(.*?)中结果即时间


# 提取国家/地区函数
def get_release_area(data):
    pattern = re.compile(r'.*\((.*)\)')
    # $表示匹配一行字符串的结尾，这里就是(.*?)；\(|$,表示匹配字符串含有(,或者只有(.*?)
    items = re.search(pattern, data)
    if items is None:
        return '未知'
    return items.group(1)


# 获取封面大图
# http://p0.meituan.net/movie/5420be40e3b755ffe04779b9b199e935256906.jpg@160w_220h_1e_1c
# 去掉@160w_220h_1e_1c就是大图
def get_thumb(url):
    pattern = re.compile(r'(.*?)@.*?')
    thumb = re.search(pattern, url)
    return thumb.group(1)


# 数据存储到csv
def write_to_file3(item):
    with open('猫眼top100.csv', 'a', encoding='utf_8_sig',newline='') as f:
        # 'a'为追加模式（添加），append的缩写
        # utf_8_sig格式导出csv不乱码
        fieldnames = ['index', 'thumb', 'name', 'star', 'time', 'area', 'score']
        w = csv.DictWriter(f,fieldnames = fieldnames)
        # w.writeheader()
        w.writerow(item)

# 封面下载
def download_thumb(name, url,num):
    try:
        response = requests.get(url)
        with open('封面图/' + name + '.jpg', 'wb') as f:
            f.write(response.content)
            print('第%s部电影封面下载完毕' %num)
            print('------')
    except RequestException as e:
        print(e)
        pass
     # 存储格式是wb,因为图片是二进制数格式，不能用w，否则会报错

# -----------------------------------------------------------------------------

def main(offset):
    url = 'http://maoyan.com/board/4?offset=' + str(offset)
    html = get_one_page(url)
    # print(html)
    # parse_one_page2(html)

    for item in parse_one_page(html):  # 切换内容提取方法
        print(item)
        write_to_file3(item)

        # 下载封面图
        download_thumb(item['name'], item['thumb'],item['index'])


# if __name__ == '__main__':
#     for i in range(10):
#         main(i * 10)
        # time.sleep(0.5)
        # 猫眼增加了反爬虫，设置0.5s的延迟时间

# 2 使用多进程提升抓取效率
from multiprocessing import Pool
if __name__ == '__main__':
    pool = Pool()
    pool.map(main, [i * 10 for i in range(10)])