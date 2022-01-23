
c1 = []
c2 = []
c3 = []
c4 = []
c5 = []

# 필요한 패키지 로드

import numpy as np
import streamlit as st
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 파이토치
import torch
import torch.nn.functional as F
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import torchvision
from torchvision import datasets, models, transforms

import pandas as pd
import numpy as np
import time
import os
from tqdm.notebook import tqdm
import re
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # device 객체

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


# 한글 폰트 설정하기
fontpath = '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf'
font = fm.FontProperties(fname=fontpath, size=10)
plt.rc('font', family='NanumBarunGothic')
matplotlib.font_manager._rebuild()

transforms_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# class_names = ['airc', 'airpods', 'bike', 'camera', 'car', 'coat', 'desktop', 'dressshose', 'electro', 'galaxy', 'glass', 'hat', 'iphone',
#                'jewelry', 'jumper', 'keyboard', 'laptop', 'mouse', 'onepiece', 'pants', 'shirt', 'skirt', 'sneaker', 'top', 'tv', 'wallet', 'watch']
class_names = ['에어컨', '에어팟', '오토바이', '카메라', '자동차', '코트', '데스크탑', '구두', '전자제품', '갤럭시', '안경', '모자', '아이폰',
               '쥬얼리', '점퍼', '키보드', '노트북', '마우스', '원피스', '바지', '셔츠', '치마', '스니커즈', '상의', 'tv', '지갑', '시계']

## 이미지 모델 호출
model = models.resnet50(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 27)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

checkpoint = torch.load('/content/drive/MyDrive/Colab Notebooks/1조/2.Image_model/모델백업/ResNet50_Pre_T/rn50b96l31_49.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
check = checkpoint['epoch']
loss = checkpoint['loss']
model.eval()


class_names = ['에어컨', '에어팟', '오토바이', '카메라', '자동차', '코트', '데스크탑', '구두', '전자제품', '갤럭시', '안경', '모자', '아이폰',
               '쥬얼리', '점퍼', '키보드', '노트북', '마우스', '원피스', '바지', '셔츠', '치마', '스니커즈', '상의', 'tv', '지갑', '시계']


## 태그추천 모델 호출
tag_list = ['노트북','에어팟','아이폰','키보드','마우스','갤럭시']

total_df = []
for i in range(len(tag_list)):
    total_df.append(pd.read_csv(f'/content/drive/MyDrive/Colab Notebooks/1조/3.Recommen_model/0.tag_model/{tag_list[i]}.csv'))
tag_dict={'노트북':total_df[0],'에어팟':total_df[1],'아이폰':total_df[2],'키보드':total_df[3],'마우스':total_df[4],'갤럭시':total_df[5]}



## 이미지 출력

def imshow(input, title):
    input = input.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    input = std * input + mean
    input = np.clip(input, 0, 1)
    plt.imshow(input)
    plt.title(title)
    plt.show()

## 입력한 제목 전처리 및 태그출력
def find_tag(title, cat_matrix):
    test = title[0].split(' ')
    for i, j in enumerate(test):
        i = re.sub('\W+',' ', j)
    test_input = []
    for i in test:
        if i in cat_matrix.columns:
            test_input.append(i)
    test_input.append('label')
    test_matrix = cat_matrix[test_input]

    # 상위 top5 태그 추출
    test_matrix['target'] = 0
    for i in range(len(test_input)-1):
        test_matrix['target'] += test_matrix[test_input[i]]
    test_matrix['target'] = test_matrix['target'] / (len(test_input)-1)
    test_matrix = test_matrix[['label','target']].sort_values(by='target', ascending=False)[:5]
    # list로 출력
    global tag
    tag = test_matrix.label.to_list()
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        global c1
        c1 = st.button(tag[0])
    with col2:
        global c2
        c2 = st.button(tag[1])
    with col3:
        global c3
        c3 = st.button(tag[2])
    with col4:
        global c4
        c4 = st.button(tag[3])
    with col5:
        global c5
        c5 = st.button(tag[4])

    return(tag)
    #return(test_matrix)

from PIL import Image
def load_image(image_file):
	img = Image.open(image_file)
	return img

output1 = []

def main():
    try:
        st.title("Team 1")
        st.header('상품 판매')

        st.subheader('상품이미지')
        image_file = st.file_uploader("Upload Images", type=['jpg'])
        if image_file is not None:
            # # st.write(type(image_file))
            # # st.write(dir(image_file))
            # '''
            # {"Filename": image_file.name
            #      , "FileType": image_file.type
            #      , "FileSize": image_file.size
            #  }
            # '''
            img = load_image(image_file)
            st.image(img)
            image = transforms_test(img).unsqueeze(0).to(device)



        with torch.no_grad():
            outputs = model(image)
            _, preds = torch.max(outputs, 1)
            imshow(image.cpu().data[0], title='예측 결과: ' + class_names[preds[0]])
            print(class_names[preds[0]])


            st.subheader('카테고리')
            category = class_names[preds[0]]
            click1 = st.button(category)
            if click1:
                category = category

            with st.expander('카테고리가 맞지 않으신가요?'):
                # coll1, coll2 = st.columns([1,10])
                # with coll1:
                #     click2 = st.button(' ')
                # with coll2:
                #     st.write('직접 카테고리를 선택')
            # if click2:

                option = st.selectbox('',
                                  ('카테고리를 선택해주세요.', '에어컨', '에어팟', '오토바이', '카메라', '자동차', '코트', '데스크탑', '구두', '전자제품', '갤럭시', '안경', '모자', '아이폰',
                   '쥬얼리', '점퍼', '키보드', '노트북', '마우스', '원피스', '바지', '셔츠', '치마', '스니커즈', '상의', 'tv', '지갑', '시계'))
                if option == '카테고리를 선택해주세요.':
                    category = category
                else:
                    category = option


        st.subheader('제목')

        title = st.text_input('제목을 입력하세요.')

        if st.session_state.get('output1', None) is None:
            st.session_state.output1 = []


        if title:
            st.subheader('연관태그')
            select_matrix = tag_dict[category]
            find_tag([title], select_matrix)
            c6 = st.text_input('연관태그 직접 입력')
            if c1:
                st.session_state.output1.append('#' + tag[0])
            if c2:
                st.session_state.output1.append('#' + tag[1])
            if c3:
                st.session_state.output1.append('#' + tag[2])
            if c4:
                st.session_state.output1.append('#' + tag[3])
            if c5:
                st.session_state.output1.append('#' + tag[4])
            if c6:
                st.session_state.output1.append('#' + c6)
            if st.session_state.output1 != []:
                st.multiselect('연관태그는 최대 5개까지 입력 가능합니다.',
                            set(st.session_state.output1), set(st.session_state.output1))








    except UnboundLocalError:
        pass






if __name__ == '__main__':
    main()
