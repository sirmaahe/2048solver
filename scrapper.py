import requests
import time
from bs4 import BeautifulSoup


host = 'https://zona.media'

links = set()
for i in range(10):
    mediazona = requests.get(
        '{}/_load?selector=categoryMaterials&page={}&total=906&cat_name=Тексты'.format(host, i),
    )
    soup = BeautifulSoup(mediazona.json()['data'][0]['html'], 'html.parser')
    elements = soup.find_all('a')

    for e in elements:
        links.add(e.get('href'))
# print(links)

def get_site(link):
    mz = requests.get(
        '{}{}'.format(host, link)
    )
    soup2 = BeautifulSoup(mz.text, 'html.parser')
    text = ''
    for p in soup2.find_all('p')[:-6]:
        text += p.get_text()
    return text


with open('text.txt', 'w+') as f:
    for i, link in enumerate(links):
        try:
            f.write(get_site(link))
        except Exception:
            pass

        if i % 2 == 0:
            time.sleep(2)
