import requests
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
        return None    # try-except语句捕获异常

def main():
    url = 'http://maoyan.com/board/4?offset=0'
    html = get_one_page(url)
    print(html)


if __name__ == '__main__':
    main()
