import requests
import re
import pandas as pd
import numpy as np
import time


def find_family(text, families):
    start = '<div class="columns3"><p class="news-item">'
    for i in text:
        if start in i:
            fam = re.findall(r'.html">(.*?)</a> <img class', i)
            # print(fam)
            for d in fam:
                if d[len(d) - 1] == 'а':
                    families.append(d[0:len(d) - 1])
                else:
                    families.append(d)
            break
    # print(families)
    return families


families = []
with requests.Session() as ses:
    ses.headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:91.0) Gecko/20100101 Firefox/91.0'}
    for i in range(1, 431):
        url = f"https://imena-znachenie.ru/familii/?PAGEN_1={i}"

        print(i, url)
        time.sleep(0.2)
        res = ses.get(url)
        text = res.text

        text = text.split('\n')
        families = find_family(text, families)

    families = np.array(families)

    dt = pd.read_csv('Family.csv', encoding='utf-8')
    fam1 = np.array(dt['Family'])

    all_families = np.append(fam1, families)
    all_families = np.unique(all_families)

    df = pd.DataFrame()
    df['family'] = all_families
    df.to_csv('families2.csv', encoding='utf-8', index=False)
    ses.close()
