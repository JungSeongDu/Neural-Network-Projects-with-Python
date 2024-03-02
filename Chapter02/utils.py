import pandas as pd
import numpy as np
from sklearn import preprocessing
import json

df = pd.read_csv("Chapter02/data/diabetes.csv")

#print(df.head())
#print(df.describe())

#print(df.isnull().any())

"""
#각 컬럼이 가진 0의 개수
for col in df.columns:
    missing_rows  = df.loc[df[col]==0].shape[0]
    print(col + ":" + str(missing_rows))
"""

#0값을 NaN으로 바꿔 판다스가 결측값을 인식하게 만듬
df['Glucose'] = df['Glucose'].replace(0,np.nan)
df['BloodPressure'] = df['BloodPressure'].replace(0,np.nan)
df['SkinThickness'] = df['SkinThickness'].replace(0, np.nan)
df['Insulin'] = df['Insulin'].replace(0, np.nan)
df['BMI'] = df['BMI'].replace(0, np.nan)

"""
#잘 바꾸었는지 확인
for col in df.columns:
    missing_rows  = df.loc[df[col]==0].shape[0]
    print(col + ":" + str(missing_rows))
"""

#NaN 값을 평균으로 대체
df['Glucose'] = df['Glucose'].fillna(df['Glucose'].mean())
df['BloodPressure'] = df['BloodPressure'].fillna(df['BloodPressure'].mean())
df['SkinThickness'] = df['SkinThickness'].fillna(df['SkinThickness'].mean())
df['Insulin'] = df['Insulin'].fillna(df['Insulin'].mean())
df['BMI'] = df['BMI'].fillna(df['BMI'].mean())

#표준화
df_scaled = preprocessing.scale(df) #데이터를 표준 정규 분포로 변환
df_scaled = pd.DataFrame(df_scaled,columns=df.columns)

df_scaled['Outcome'] = df['Outcome']

df = df_scaled

#print(df.describe().loc[['mean','std','max']].round(2).abs())

"""
df = df.to_json(orient='index')

f = open('summary.json','w')
f.write(df)
f.close

with open('summary.json','r') as f:
    json_data = json.load(f)

pregnancies_data = json_data.get("0", {}).get("Pregnancies", None) #json_data.get("0", None)
print(pregnancies_data)
"""

"""
df = df.to_excel(index=False)

f = open('summary.xlsx','w')
f.write(df)
f.close
"""