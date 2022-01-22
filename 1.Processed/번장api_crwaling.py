## 패키지 호출

from this import d
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json
import time
import datetime

from pandas.io.json import json_normalize
from bs4 import BeautifulSoup
pd.options.display.max_info_columns =200
pd.options.display.max_columns = 200
pd.options.display.max_info_rows =100
pd.options.display.max_rows = 100

import os
from tqdm import tqdm_notebook

# pages, query_list 수기 지정
# 해당 내용 안내 출력
print('&& pages 변수 range 지정으로 변경 가능 default : range(1,3)&& /n')
print('&& query_list에 검색을 하고 싶은 키워드 list 생성하기 default:''[''아디다스','에어컨'']'' &&\n')
# pages = range(1,101)
pages = range(1,3)
# query_list = ['아디다스','에어컨','에어팟','오토바이','블라우스','팔찌','카메라','코트','컴퓨터','구두','귀걸이','갤럭시',
#               '갤럭시케이스','안경','모자','하이힐','후드','아이폰케이스','아이폰','자켓','청바지','점퍼','중고차','키보드','노트북',
#               '마우스','목도리','네일','니트','목걸이','나이키','닌텐도','원피스','패딩','바지','향수','냉장고','레고',
#               '반지','셔츠','치마','스피커','정장','선글라스','티셔츠','트레이닝','티비','지갑','세탁기','시계']
query_list = ['아디다스','에어컨']

def bun_api_crwaling():
    total_df =[]

    for query in query_list:
        cat_df = pd.DataFrame()
        
        for page in tqdm_notebook(pages):
            
            pid_list = [] # 아이템 리스트 담기
            bunjang_url = f'https://api.bunjang.co.kr/api/1/find_v2.json?q={query}&order=date&page={page}&stat_device=w&stat_category_required=1&req_ref=search&version=4'
            response = requests.get(bunjang_url.encode('utf-8'))
            
            try:
                item_list = response.json()["list"]
                ids = [item["pid"] for item in item_list]
                pid_list.extend(ids)
            except:
                continue
            
            # 아이템별 데이터 프레임 생성
            df = pd.DataFrame()
            product_id, image_link, title, keyword, cat1, cat2, cat3, view = [],[],[],[],[],[],[],[]
            for pid in pid_list:
                url = f"https://api.bunjang.co.kr/api/1/product/{pid}/detail_info.json?version=4"
                response = requests.get(url)
            
                try : 
                    product_id.append(response.json()['item_info']['pid'])
                except : 
                    product_id.append(0)
                try :
                    image_link.append(response.json()['item_info']['product_image'])
                except : 
                    image_link.append(0)
                try : 
                    title.append(response.json()['item_info']['name'])
                except : 
                    title.append(0)
                try : 
                    keyword.append(response.json()['item_info']['keyword'])
                except : 
                    keyword.append(0)
                try : 
                    cat1.append(list(response.json()['item_info']['category_name'][0].values())[0])
                except : 
                    cat1.append(0)
                try : 
                    cat2.append(list(response.json()['item_info']['category_name'][1].values())[0])
                except : 
                    cat2.append(0)                
                try : 
                    cat3.append(list(response.json()['item_info']['category_name'][2].values())[0])
                except : 
                    cat3.append(0)
                try : 
                    view.append(response.json()['item_info']['num_item_view'])
                except : 
                    view.append(0)

            df['product_id'] = product_id
            df['title'] = title
            df['keyword'] = keyword
            df['cat1'] = cat1
            df['cat2'] = cat2
            df['cat3'] = cat3
            df['view'] = view
            df['image'] = image_link
            temp = df
            cat_df = pd.concat([cat_df,temp])

        total_df.append(cat_df)
    return(total_df)

print('&& 현재 저장 위치 : Users/ppangppang/Desktop, 변경하려면 path 수정 &&')
def save_data(total_df):
    path = '/Users/ppangppang/Desktop/'
    for i in range(len(total_df)):
        total_df[i].to_csv(f'{path}{query_list[i]}.csv',index=False)
    print('파일 저장 완료')