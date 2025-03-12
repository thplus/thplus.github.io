---
title: 4주차 AI 데이터 전처리 Mini Quest
date: 2025-02-19
categories: [Assignment, 4th Week]
tags: [mini_quest, python, numpy, pandas, seaborn, matplotlib, sklearn]      # TAG names should always be lowercase
math: true
---

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
```

# **Data Preprocessing(데이터 전처리)**

1. 간단한 데이터셋을 사용하여 데이터 전처리 과정을 단계별로 진행해보세요. 각 단계에서 필요한 코드를 작성하고 그 결과를 확인합니다.


```python
data = {
'학생': ['A', 'B', 'C', 'D', 'E'],
'수학': [90, np.nan, 85, 88, np.nan],
'영어': [80, 78, np.nan, 90, 85],
'과학': [np.nan, 89, 85, 92, 80]
}
```

- 문제설명</br>
  1. 데이터 수집 : 가상의 학생 성적 데이터를 생성합니다.</br>
  2. 결측값 처리 : 데이터 셋에 누락된 값이 있을 때, 이를 평균값으로 대체 합니다.</br>
  3. 이상치 제거 : 데이터셋에 이상치가 없는지 확인하고 필요하면 제거합니다.</br>
  4. 데이터 정규화 : 수학, 영어, 과학 점수를 0과 1 사이의 값으로 스케일링 합니다.</br>
  5. 데이터 분할 : 데이터셋을 학습용과 검증용으로 나눕니다.</br>


```python
# 1. 데이터 수집
df = pd.DataFrame(data)

# 2. 결측값 처리
df.fillna({'수학': df['수학'].mean()}, inplace=True) # pandas 3.0 식
df.fillna({'영어': df['영어'].mean()}, inplace=True)
df.fillna({'과학': df['과학'].mean()}, inplace=True)

# 3. 이상치 제거 (IQR 이용)
for col in ['수학', '영어', '과학']:
  Q1 = df[col].quantile(0.25)
  Q3 = df[col].quantile(0.75)
  IQR = Q3 - Q1
  lower_bound = Q1 - 1.5 * IQR
  upper_bound = Q3 + 1.5 * IQR
  outlier = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
  if outlier.empty:
    print(f"'{col}'에 이상치가 없습니다.")
  else:
    print("이상치 : ")
    print(outlier[[col]])
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

# 4. 데이터 정규화
scaler = MinMaxScaler()
df[['수학', '영어', '과학']] = scaler.fit_transform(df[['수학', '영어', '과학']])

# 5. 데이터 분할
train, test = train_test_split(df, test_size=0.2, random_state=42)
print("학습용 데이터셋 : ")
print(train)
print("검증용 데이터셋 : ")
print(test)
```

    이상치 : 
         수학
    0  90.0
    2  85.0
    '영어'에 이상치가 없습니다.
    '과학'에 이상치가 없습니다.
    학습용 데이터셋 : 
      학생   수학        영어   과학
    3  D  1.0  1.000000  1.0
    4  E  0.0  0.583333  0.0
    검증용 데이터셋 : 
      학생   수학   영어    과학
    1  B  0.0  0.0  0.75
    

2. 가상의 데이터셋을 사용하여 전처리 과정을 직접 구현해보세요. 각 단계에 맞춰 코드를 작성하고 최종적으로 학습용과 검증용 데이터로 나누는 작업을 합니다.


```python
# 가상의 제품 판매 데이터
data = {
'제품': ['A', 'B', 'C', 'D', 'E'],
'가격': [100, 150, 200, 0, 250],
'판매량': [30, 45, np.nan, 55, 60]
}
```

- 문제
  1. 결측값을 중앙값으로 대체합니다.
  2. 이상치를 제거합니다. (ex. 가격이 0 이하인 경우)
  3. 데이터를 표준화합니다. (평균 0, 표준편차 1)
  4. 데이터를 학습용과 검증용으로 나눕니다. (학습용 70%, 검증용 30%)

- 문제설명
  1. 데이터 수집 : 가상의 제품 판매 데이터를 생성합니다.
  2. 결측값 처리 : 데이터셋에 누락된 값이 있을 때, 이를 중앙값으로 대체합니다.
  3. 이상치 제거 : 가격이 0 이하인 데이터를 제거 합니다.
  4. 데이터 표준화 : 가격과 판매량 데이터를 평균 0, 표준편차 1로 변환합니다.
  5. 데이터 분할 : 데이터셋을 학습용과 검증용으로 나눕니다.


```python
# 1. 데이터 수집
df = pd.DataFrame(data)

# 2. 결측값 처리
df.fillna({'판매량': df['판매량'].median()}, inplace=True)

# 3. 이상치 제거
df = df[df['가격'] > 0]

# 4. 데이터 표준화
scaler = StandardScaler()
df[['가격', '판매량']] = scaler.fit_transform(df[['가격', '판매량']])

# 5. 데이터 분할
train_df, valid_df = train_test_split(df, test_size=0.3, random_state=42)

print("학습용 데이터셋:")
print(train_df)
print("\n검증용 데이터셋:")
print(valid_df)
```

    학습용 데이터셋:
      제품        가격       판매량
    0  A -1.341641 -1.501111
    2  C  0.447214  0.346410
    
    검증용 데이터셋:
      제품        가격       판매량
    1  B -0.447214 -0.115470
    4  E  1.341641  1.270171
    
