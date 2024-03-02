import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

df = pd.read_csv("Chapter02/data/diabetes.csv")

# 3*3 크기로 서브 차트를 만든다
plt.subplots(3, 3, figsize=(6, 6))

# 각 특징의 변수의 밀도 차트를 그린다
for idx, col in enumerate(df.columns[:-1]):
    ax = plt.subplot(3, 3, idx + 1)
    ax.yaxis.set_ticklabels([]) #y축의 눈금 라벨(숫자)을 제거

    # Outcome에 따른 밀도 차트 그리기
    sns.histplot(df.loc[df.Outcome == 0, col], stat="density", common_norm=False, kde_kws={'linestyle': '-', 'color': 'black', 'label': "No Diabetes"})
    sns.histplot(df.loc[df.Outcome == 1, col], stat="density", common_norm=False, kde_kws={'linestyle': '--', 'color': 'black', 'label': "Diabetes"})
    
    ax.set_title(col)

#차트가 8개 이므로 9번째는 숨긴다
plt.subplot(3, 3, 9).set_visible(False)
plt.tight_layout()
plt.show()
