import requests
import re
import pandas as pd
import numpy as np
import time
russian = 'а б в г д е ё ж з и й к л м н о п р с т у ф х ц ч ш щ ъ ы ь э ю я'
russian = russian.replace(' ','')

def find_families(page, families):
    start = 'Начните вводить интересующее вас слово или понятие.'
    end = 'next-page-control'
    fl = False
    for i in page:
        if end in i:
            break
        if fl == True:
            if '<li><a' in i:
                a = re.findall(r'>(.*?)</a></li>', i[i.find('<li><')+len('<li><'):len(i)])
                families.append(a[0].strip().lower().capitalize())
        if start in i:
            fl = True

    return families

#url = 'https://gufo.me/dict/surnames_ru?page=7&letter=%D0%B0'
families = []
with requests.Session() as ses:
    ses.headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:91.0) Gecko/20100101 Firefox/91.0'}
    for letter in russian:
        i = 1
        while 1:
            url = f"https://gufo.me/dict/surnames_ru?page={i}&letter={letter}"
            print(url)
            time.sleep(0.3)
            res = ses.get(url)
            text = res.text
            if 'Not Found (#404)' in text:
                break
            else:
                text = text.split('\n')
                families = find_families(text, families)

            i+=1
    print(families)
    df = pd.DataFrame({'Family': np.array(families)})
    df.to_csv('Family.csv', encoding='utf-8', index=False)
    ses.close()
