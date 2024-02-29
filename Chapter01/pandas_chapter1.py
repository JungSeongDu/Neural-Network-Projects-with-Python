# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 15:50:45 2024

@author: sungd
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#df = pd.read_csv("raw_data.csv")

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
                 names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])

print(df.info())
print(df.describe())
print(df.head(10))
df.loc[df['sepal_length'] > 5.0]

#클래스별로 마커모양을 지정한다
marker_shapes = marker_shapes = ['.', '^', '*']

#점 차트를 그린다
ax = plt.axes()
for i,species in enumerate(df['class'].unique()):
    species_data = df[df['class'] == species]
    species_data.plot.scatter(x='sepal_length',
                              y='sepal_width',
                              marker=marker_shapes[i],
                              s=100,
                              title = "sepal width vs length by species",
                              label=species,figsize=(10,7), ax=ax
                              )
    
plt.show()
plt.clf()

df['petal_length'].plot.hist(title = "histogram of petal length")
df.plot.box(title='bpxplot of sepal length & width, and petal length & width')
plt.show()

#----------------------------------------------------

#범주형
#원 핫 인코딩
df2 = pd.DataFrame({'Days' : ['monday','tuesday','wednesday','thursday','friday','saturday','sunday']})
print(df2)
print(pd.get_dummies(df2))

#-----------------------------------------------------

#결측치 보간
#붓꽃데이터 셋을 다시 가져온다
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
df3 = pd.read_csv(URL,names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])

#로우 10개를 무작위로 고른다
random_index = np.random.choice(df3.index,replace=False,size=10)

#무작위로 고른 로우의 spal_length 값을 None으로 바꾼다
"""
loc사용 : df.loc[ : , 칼럼이름] [칼럼여러개] , [칼럼슬라이싱]도 가능
iloc사용 : df.iloc[ : , 칼럼번호 ] [ 칼럼번호 여러개 ] , [ 칼럼번호 슬라이싱 ]도 가능
"""
df3.loc[random_index,'sepal_length'] = None
print(df3.isnull().any()) #결측값 확인

print("삭제전 행 : %d" %(df3.shape[0]))
df4 = df3.dropna() #결측값이 있는 로우 제거
print("삭제 후 행 : %d" %(df4.shape[0]))

#결측값을 평균으로 대체
df3.sepal_length = df3.sepal_length.fillna(df.sepal_length.mean())
print(df3.isnull().any())

