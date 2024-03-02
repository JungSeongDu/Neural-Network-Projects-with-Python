import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import Binarizer
from sklearn.metrics import roc_curve
from keras.models import Sequential
from keras.layers import Dense
import seaborn as sns
import numpy as np


df = pd.read_csv("Chapter02/data/diabetes.csv")

# 현재 디렉토리의 파일 목록 확인
#print(os.listdir())

#print(df.head())

#데이터셋 분할
X = df.loc[:, df.columns != 'Outcome']
y = df.loc[:,'Outcome']

X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2)
X_train , X_val , y_train , y_val = train_test_split(X_train,y_train,test_size=0.2)

#케라스 모델 만들기
model = Sequential()
model.add(Dense(32,activation='relu',input_dim=8)) #첫번째 은닉 레이어
model.add(Dense(16,activation='relu')) #두번째 은닉 레이어
model.add(Dense(1,activation='sigmoid'))

#모델 컴파일
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])

#모델 훈련
#200회 반복훈련
model.fit(X_train,y_train,epochs=200)

#모델 성능
scores = model.evaluate(X_train,y_train)
print("Training Accuracy: %.2f%%\n" % (scores[1]*100))
scores = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy: %.2f%%\n" % (scores[1]*100))

#혼동 행렬
# 예측 수행
y_test_pred_probs = model.predict(X_test)

# Binarizer 생성 및 적용
threshold = 0.5  # 임계값 설정
binarizer = Binarizer(threshold=threshold)
y_test_pred_classes = binarizer.fit_transform(y_test_pred_probs)

# 혼동 행렬 생성
c_matrix = confusion_matrix(y_test, y_test_pred_classes)

# 시각화
ax = sns.heatmap(c_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Diabetes', 'Diabetes'], yticklabels=['No Diabetes', 'Diabetes'], cbar=False)
ax.set_xlabel("Prediction")
ax.set_ylabel("Actual")
plt.show()
plt.clf()

#ROC곡선
y_test_pred_probs = model.predict(X_test)
FPR, TPR, _ = roc_curve(y_test, y_test_pred_probs)
plt.plot(FPR, TPR)
plt.plot([0,1],[0,1],'--', color='black') #대각선
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
plt.clf()