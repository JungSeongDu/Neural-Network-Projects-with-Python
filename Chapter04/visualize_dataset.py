import os
from matplotlib import pyplot as plt
import random

#print(os.listdir())


#파일명 리스트를 가져온다
_,_,cat_images = next(os.walk('Chapter04/Dataset/kagglecatsanddogs_5340/PetImages/Cat')) #고양이
#os.walk() 함수는 디렉터리 구조를 순회하면서 각 디렉터리에 대한 정보를 튜플 형태로 생성
#이 튜플의 첫 번째 요소는 현재 디렉터리의 경로이고, 
#두 번째 요소는 서브디렉터리의 리스트, 
#세 번째 요소는 현재 디렉터리에 있는 파일들의 리스트

#print(cat_images)

#가로 세개 , 세로 세개 (총 9개)짜리 차트를 준비한다.
fig , ax = plt.subplots(3,3,figsize=(20,10))

#무작위로 선택한 이미지로 차트를 구성
for idx,img in enumerate(random.sample(cat_images,9)):
    # plt.imread() : Matplotlib 라이브러리에서 이미지 파일을 읽어들이는 함수
    img_read = plt.imread('Chapter04/Dataset/kagglecatsanddogs_5340/PetImages/Cat/'+img)
    ax[int(idx/3), idx%3].imshow(img_read)
    # int(idx/3) : 현재 인덱스 idx를 3으로 나눈 몫
    # idx%3 : idx를 3으로 나눈 나머지
    ax[int(idx/3), idx%3].axis('off')
    ax[int(idx/3), idx%3].set_title('Cat/'+img)

plt.show()


#파일명 리스트를 가져온다
_,_,dog_images = next(os.walk('Chapter04/Dataset/kagglecatsanddogs_5340/PetImages/Dog')) #개

# 가로 3개, 세로 3개(총 9개)짜리 차트를 준비한다
fig, ax = plt.subplots(3,3, figsize=(20,10))

# 무작위로 선택한 이미지를 그린다
for idx, img in enumerate(random.sample(dog_images, 9)):
    img_read = plt.imread('Chapter04/Dataset/kagglecatsanddogs_5340/PetImages/Dog/'+img)
    ax[int(idx/3), idx%3].imshow(img_read)
    ax[int(idx/3), idx%3].axis('off')
    ax[int(idx/3), idx%3].set_title('Dog/'+img)

plt.show()