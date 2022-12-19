# -*- coding: utf-8 -*-

"""Use to parse websites and get only the information of interest quickly

Mostly for copy-paste.
"""

from typing import Union
from collections import Counter

import requests
from bs4 import BeautifulSoup


def get_soup(url: str):
    rq = requests.get(url=url, headers={
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36'
    })
    return BeautifulSoup(rq.text, 'html.parser')


def _get_super_fast_python_posts():
    # used to make a list of links to read
    # links that lead to text appear thrice
    base_url = 'https://superfastpython.com/category/'

    categories = [
        'threading',
        'multiprocessing',
        'asyncio',
    ]

    def __get_links_per_page(category, page_num: int) -> list:
        soup = get_soup(f'{base_url}{category}/page/{page_num}')
        res = []
        counter = Counter()
        for a in soup.find_all('a', href=True):
            counter[a['href']] += 1
        for k, v in counter.items():
            if v == 3:
                res.append(k)
        return res

    for cat in categories:
        soup = get_soup(f'{base_url}{cat}/page/1')
        npages = 1
        for a in soup.find_all('a', href=True):
            a = a['href'].strip('/').split('/')
            if len(a) > 2 and a[-2] == 'page':
                npages = max(npages, int(a[-1]))
        links = []
        for i in range(1, npages + 1):
            links += __get_links_per_page(cat, i)
        for link in links[::-1]:
            print(link)
        print('\n')


__author__ = 'Petr Panov'
__copyright__ = 'Copyleft 2022, Milky Way'
__credits__ = ['Petr Panov']
__license__ = 'MIT'
__version__ = '1.0.0'
__maintainer__ = 'Petr Panov'
__email__ = 'pvpanov93@gmail.com'
__status__ = "Draft"
