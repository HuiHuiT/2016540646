#import requests
#response=requests.get("http://www.hufe.edu.cn")
#print(response)
#content=requests.get("http://www.hufe.edu.cn").content
#print(content)

#import urllib.request
#content=urllib.request.urlopen("http://www.163.com").read()
#print(content)


import requests
from bs4 import BeautifulSoup
def trade_spider(max_page):
    page=1
    while page<=max_page:
        url = 'http://www.163.com'
        source_code = requests.get(url)
        plain_text = source_code.text
        soup = BeautifulSoup(plain_text)
        for link in soup.findAll('a'):
            href = link.get('href')
            print(href)

