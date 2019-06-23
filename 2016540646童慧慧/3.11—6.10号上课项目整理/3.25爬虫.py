import requests
from requests import RequestException


def get_one_page(url):
    try:
        headers={
                 'User-Agent':'Mozilla/5.0(windows NT 6.1;WOW64)AppleWebKit/537.36(KHTM, like Chrome/66.0.3359.181 Safari/537.36'} #模拟用户的语句
        response=requests.get(url)
        if response.status_code==200:  #避免出错进行的验证
            return response.text
        else:
            return None
    except RequestException:
        return None
    #以上是一个可调用的模块

def main():
    url='https://maoyan.com/board/4?offset=0'
    html=get_one_page(url)
    print(html)

if __name__ == '__main__':
    main()

#用于查找排名的正则表达式
'<dd>.*?board-index.*?>(d+)</i>'
'scr="(.*?)".*?'