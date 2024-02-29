# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 21:02:22 2024

@author: sungd
"""

from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
import numpy as np

#레이어를 나란히 연결하려면 먼저 커라스의 Sequential 모델을 선언해야 한다
model = Sequential()

#레이어 추가
#레이어1
model.add(Dense(units=4 , activation='sigmoid', input_dim=3))

#출력레이어
model.add(Dense(units=1,activation='sigmoid'))
print(model.summary())


#모델을 컴파일 & 훈련
sgd = optimizers.SGD(learning_rate=1)
model.compile(loss='mean_squared_error', optimizer=sgd)

#결과를 재현할 수 있게 랜덤시드를 고정
np.random.seed(9)
x = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
y = np.array([[0],[1],[1],[0]])

model.fit(x,y
          ,epochs=1500
          ,verbose=False)
print(model.predict(x))