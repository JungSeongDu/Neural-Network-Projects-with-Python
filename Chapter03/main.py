import matplotlib
matplotlib.use("TkAgg")
from utils import preprocess, feature_engineer
import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import BatchNormalization
from sklearn.metrics import mean_squared_error

try:
    print("Reading in the dataset. This will take some time..")
    df = pd.read_csv('Chapter03/data/new-york-city-taxi-fare-prediction/NYC_taxi.csv', parse_dates=['pickup_datetime'], nrows=500000)
except:
    print("""
      Dataset not found in your computer.
      Please follow the instructions in the link below to download the dataset:
      https://raw.githubusercontent.com/PacktPublishing/Neural-Network-Projects-with-Python/master/chapter3/how_to_download_the_dataset.txt
      """)
    quit()


#데이터 천처리, 특징 공학 과정
df = preprocess(df)
df = feature_engineer(df)

#변수 스케일링과정을 구축
#이 코드를 통해 'fare_amount'를 제외한 나머지 변수들이 스케일링
df_prescaled = df.copy()
df_scaled = df.drop(['fare_amount'],axis=1)
df_scaled = scale(df_scaled)
cols = df.columns.tolist()
cols.remove('fare_amount')
df_scaled = pd.DataFrame(df_scaled,columns = cols , index=df.index) #스케일링된 데이터를 다시 데이터프레임 형태로 변환
df_scaled = pd.concat([df_scaled,df['fare_amount']],axis=1) #concat ==> 데이터프레임을 행 또는 열의 방향으로 결합할 때 활용
df = df_scaled.copy()

#데이터 프레임을 훈련데이터셋과 테스트데이터섹으로 나눈다
X = df.loc[:,df.columns != 'fare_amount']
y = df.fare_amount
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

# 케러스 신경망을 구축
model = Sequential()
model.add(Dense(128,activation='relu',input_dim=X_train.shape[1])) #shape[0]은 행의 수를 나타내고 shape[1]은 열의 수
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1)) #회귀모델에는 활성화 함수를 적용하지 않음 ==> 출력값을 왜곡하고 모델 성능에 악영향을 줄수 있음

model.compile(loss='mse',optimizer='adam',metrics=['mse'])
# loss='mse' : 모델이 최소화하려는 손실 함수를 지정
# optimizer='adam' : 모델을 최적화하는 데 사용되는 최적화 알고리즘을 지정
# metrics=['mse'] : 모델의 평가 지표를 지정

model.fit(X_train,y_train,epochs=1)

#결과 분석
train_pred = model.predict(X_train)
train_rmse = np.sqrt(mean_squared_error(y_train, train_pred)) #평균 제곱 오차(Mean Squared Error, MSE)를 계산하는 함수

test_pred = model.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
print("Train RMSE: {:0.2f}".format(train_rmse))
print("Test RMSE: {:0.2f}".format(test_rmse))
print('------------------------')

#테스트 데이터 셋에서 무작위로 로우를 하나 뽑아서 모델에 전달하고 예측값을 산출
#다음의 예즉 결과를 RMSE를 계산해 출력
def predict_random(df_prescaled, X_test, model):
    sample = X_test.sample(n=1, random_state=np.random.randint(low=0, high=10000))
    idx = sample.index[0] #선택된 샘플의 인덱스를 저장

    actual_fare = df_prescaled.loc[idx,'fare_amount']
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_of_week = day_names[df_prescaled.loc[idx,'day_of_week']]
    hour = df_prescaled.loc[idx,'hour']
    predicted_fare = model.predict(sample)[0][0]
    rmse = np.sqrt(np.square(predicted_fare-actual_fare))

    print("Trip Details: {}, {}:00hrs".format(day_of_week, hour))
    print("Actual fare: ${:0.2f}".format(actual_fare))
    print("Predicted fare: ${:0.2f}".format(predicted_fare))
    print("RMSE: ${:0.2f}".format(rmse))

predict_random(df_prescaled, X_test, model)
