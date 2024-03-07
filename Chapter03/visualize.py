import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

#print(os.listdir())

df = pd.read_csv('Chapter03/data/new-york-city-taxi-fare-prediction/NYC_taxi.csv', parse_dates=['pickup_datetime'], nrows=500000)

#print(df.head())

# 뉴욕시의 경도 범위
nyc_min_longitude = -74.05
nyc_max_longitude = -73.75

# 뉴욕시의 위도 범위
nyc_min_latitude = 40.63
nyc_max_latitude = 40.85

df2 = df.copy(deep=True)

# 승하차 및 하차 위치를 뉴욕시 내로 한정
for long in ['pickup_longitude', 'dropoff_longitude']:
    df2 = df2[(df2[long] > nyc_min_longitude) & (df2[long] < nyc_max_longitude)]

for lat in ['pickup_latitude', 'dropoff_latitude']:
    df2 = df2[(df2[lat] > nyc_min_latitude) & (df2[lat] < nyc_max_latitude)]

    landmarks = {'JFK Airport': (-73.78,40.643),
             'Laguardia Airport': (-73.87, 40.77),
             'Midtown': (-73.98, 40.76),
             'Lower Manhattan': (-74.00, 40.72),
             'Upper Manhattan': (-73.94, 40.82),
             'Brooklyn': (-73.95, 40.66)}
    
def plot_lat_long(df, landmarks, points='Pickup'):
    plt.figure(figsize = (12,12)) # 차트 크기를 설정한다
    if points == 'pickup':
        plt.plot(list(df.pickup_longitude), list(df.pickup_latitude), '.', markersize=1) #list()를 사용한 이유는 plt.plot 함수가 기본적으로 NumPy 배열이나 리스트와 같은 순차적인 데이터를 받기 때문
        #plt.scatter(df['pickup_longitude'], df['pickup_latitude'], marker='.', s=1) #scatter 함수는 pandas Series를 바로 입력으로 받을 수 있음
    else:
        plt.plot(list(df.dropoff_longitude), list(df.dropoff_latitude), '.', markersize=1)

    for landmark in landmarks:
        plt.plot(landmarks[landmark][0], landmarks[landmark][1], '*', markersize=15, alpha=1, color='r') # 랜드마크를 지도 위에 표시한다
        plt.annotate(landmark, (landmarks[landmark][0]+0.005, landmarks[landmark][1]+0.005), color='r', backgroundcolor='w') # add 0.005 offset on landmark name for aesthetics purposes

    plt.title("{} Locations in NYC Illustrated".format(points))
    plt.grid(None)
    plt.xlabel("Latitude")
    plt.ylabel("Longitude")
    plt.show()

plot_lat_long(df2, landmarks, points='Pickup')

plot_lat_long(df2, landmarks, points='Drop Off')

#요일 및 시간별 승차 통계
df['year'] = df['pickup_datetime'].dt.year
df['month'] = df['pickup_datetime'].dt.month
df['day'] = df['pickup_datetime'].dt.day
df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
df['hour'] = df['pickup_datetime'].dt.hour

df['day_of_week'].plot.hist(bins=np.arange(8)-0.5,ec='black',ylim=(60000,75000))
plt.xlabel('Day of Week (0=Monday, 6=Sunday)')
plt.title('Day of Week Histogram')
plt.show()

df['hour'].plot.hist(bins=24, ec='black')  #막대(bin)의 갯수를 지정하는 매개변수
plt.title('Pickup Hour Histogram')
plt.xlabel('Hour')
plt.show()

#데이터 전처리 
print(df.isnull().sum())
#print(df.describe())

df['fare_amount'].hist(bins=500)
plt.xlabel("Fare")
plt.title("Histogram of Fares")
plt.show() 


#이상치 제거
#0달러 보다 작거나 100달러 보다 큰 요금을 가진 로우를 제거
df = df[(df['fare_amount'] >= 0 ) & (df['fare_amount']) <= 100 ]

#승객수가 0인 데이터 
df['passenger_count'].hist(bins=6 , ec='black')
plt.xlabel("Passenger Count")
plt.title("Histogram of Passenger Count")
plt.show()

#승객수가 0인 데이터 => 최빈 값으로 대체
df.loc[df['passenger_count']==0,'passenger_count'] = 1

#스차 및 하차뉘치 이상치
df.plot.scatter('pickup_longitude','pickup_latitude')
plt.show()