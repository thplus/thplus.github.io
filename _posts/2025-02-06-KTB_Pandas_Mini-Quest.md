---
title: 2주차 Pandas Mini Quest
date: 2025-02-06
categories: [Assignment, 2nd Week]
tags: [mini_quest, python, numpy, pandas]     # TAG names should always be lowercase
math: true
---

#**시리즈(Series)**

1. Pandas의 Series를 리스트 `[5, 10, 15, 20]`을 사용하여 생성하고, Series의 인덱스를 확인하는 코드를 작성하세요.


```python
import pandas as pd

data = [5, 10, 15, 20]
series = pd.Series(data)
print(series.index)
```

    RangeIndex(start=0, stop=4, step=1)
    

2. 다음 딕셔너리를 이용하여 Pandas Series를 생성하고, 인덱스를 활용하여 'b'의 값을 출력하는 코드를 작성하세요.


```python
data = {'a': 100, 'b': 200, 'c': 300}

series = pd.Series(data)
print(series['b'])
```

    200
    

3. 다음 Pandas Series에 대해 결측값(`NaN`)을 확인하고, 모든 결측값을 `0`으로 채운 후 Series의 값을 출력하는 코드를 작성하세요.


```python
series = pd.Series([1, 2, None, 4, None, 6])

print(series.isnull(), '\n')

series = series.fillna(0)
print(series)
```

    0    False
    1    False
    2     True
    3    False
    4     True
    5    False
    dtype: bool 
    
    0    1.0
    1    2.0
    2    0.0
    3    4.0
    4    0.0
    5    6.0
    dtype: float64
    

#**데이터프레임(DataFrame)**

1. 다음 데이터프레임에서 열(column)의 이름들을 출력하는 코드를 작성하세요.


```python
data = {'이름': ['홍길동', '김철수', '박영희'],
        '나이': [25, 30, 28],
        '성별': ['남', '남', '여']}
df = pd.DataFrame(data)

print(df.columns)
```

    Index(['이름', '나이', '성별'], dtype='object')
    

2. 다음 데이터프레임에서 나이 열(age)을 기준으로 오름차순으로 정렬된 새로운 데이터프레임을 생성하고 출력하는 코드를 작성하세요.


```python
data = {'이름': ['홍길동', '김철수', '박영희'],
        '나이': [25, 30, 28],
        '성별': ['남', '남', '여']}
df = pd.DataFrame(data)

print(df.sort_values(by='나이', ascending=True))
```

        이름  나이 성별
    0  홍길동  25  남
    2  박영희  28  여
    1  김철수  30  남
    

3. 아래와 같은 데이터 프레임이 있습니다. 각 학생의 총점을 계산하는 새로운 열을 추가한 후, 총점이 250점 이상인 학생들만 포함된 데이터프레임을 생성하세요.


```python
data = {'이름': ['홍길동', '김철수', '박영희', '이순신'],
        '국어': [85, 90, 88, 92],
        '영어': [78, 85, 89, 87],
        '수학': [92, 88, 84, 90]}
df = pd.DataFrame(data)

df['총점'] = df['국어'] + df['영어'] + df['수학']

new_df = df[df['총점'] >= 250]
print(new_df)
```

        이름  국어  영어  수학   총점
    0  홍길동  85  78  92  255
    1  김철수  90  85  88  263
    2  박영희  88  89  84  261
    3  이순신  92  87  90  269
    

#**필터링(Filtering)**

1. 주어진 데이터프레임에서 나이가 30 이상인 행만 필터링하는 코드를 작성하세요.


```python
data = {'이름': ['홍길동', '김철수', '박영희', '이순신', '강감찬'],
'나이': [25, 30, 35, 40, 45],
'도시': ['서울', '부산', '서울', '대구', '부산']}

df = pd.DataFrame(data)
print(df[df['나이'] >= 30])
```

        이름  나이  도시
    1  김철수  30  부산
    2  박영희  35  서울
    3  이순신  40  대구
    4  강감찬  45  부산
    

2. 주어진 데이터프레임에 도시가 '서울'이거나 점수가 80점 이상인 데이터를 필터링하는 코드를 작성하세요.


```python
data = {'이름': ['홍길동', '김철수', '박영희', '이순신', '강감찬'],
'나이': [25, 30, 35, 40, 45],
'도시': ['서울', '부산', '서울', '대구', '부산'],
'점수': [85, 90, 75, 95, 80]}

df = pd.DataFrame(data)

print(df[(df['도시'] == '서울') | (df['점수'] >= 80)])
```

        이름  나이  도시  점수
    0  홍길동  25  서울  85
    1  김철수  30  부산  90
    2  박영희  35  서울  75
    3  이순신  40  대구  95
    4  강감찬  45  부산  80
    

3. 주어진 데이터프레임에서 `query()`를 사용하여 나이가 35이상이고 점수가 80점 초과인 데이터를 필터링하는 코드를 작성하세요.


```python
data = {'이름': ['홍길동', '김철수', '박영희', '이순신', '강감찬'],
'나이': [25, 30, 35, 40, 45],
'도시': ['서울', '부산', '서울', '대구', '부산'],
'점수': [85, 90, 75, 95, 80]}

df = pd.DataFrame(data)

print(df.query('나이 >= 35 and 점수 > 80'))
```

        이름  나이  도시  점수
    3  이순신  40  대구  95
    

#**그룹화(Grouping)**

1. 주어진 데이터프레임에서 `groupby()`를 사용하여 '부서'별로 급여의 합계를 구하는 코드를 작성하세요.


```python
data = {'이름': ['홍길동', '김철수', '박영희', '이순신'],
        '부서': ['영업', '영업', '인사', '인사'],
        '급여': [5000, 5500, 4800, 5100]}

df = pd.DataFrame(data)

new_df = df.groupby('부서')['급여'].sum()
print(new_df)
```

    부서
    영업    10500
    인사     9900
    Name: 급여, dtype: int64
    

2. 아래 데이터프레임에서 `groupby()`와 `agg()`를 사용하여 '부서'별 급여의 합계(`sum`)와 평균(`mean`)을 계산하는 코드를 작성하세요.


```python
data = {'이름': ['홍길동', '김철수', '박영희', '이순신',  '강감찬', '신사임당'],
        '부서': ['영업', '영업', '인사', '인사', 'IT', 'IT'],
        '급여': [5000, 5500, 4800, 5100, 6000, 6200]}

df = pd.DataFrame(data)

new_df = df.groupby('부서')['급여'].agg(['sum', 'mean'])
print(new_df)
```

          sum    mean
    부서               
    IT  12200  6100.0
    영업  10500  5250.0
    인사   9900  4950.0
    

3. 부서별로 데이터를 그룹화한 뒤, 각 부서의 평균 급여가 5000 이상인 경우만 필터링하는 코드를 작성하시오.


```python
data = {'이름': ['홍길동', '김철수', '박영희', '이순신',  '강감찬', '신사임당'],
        '부서': ['영업', '영업', '인사', '인사', 'IT', 'IT'],
        '급여': [5000, 5500, 4800, 5100, 6000, 6200]}

df = pd.DataFrame(data)

new_df = df.groupby('부서').filter(lambda x: x['급여'].mean() >= 5000)
print(new_df)
```

         이름  부서    급여
    0   홍길동  영업  5000
    1   김철수  영업  5500
    4   강감찬  IT  6000
    5  신사임당  IT  6200
    

#**병합(Merging)**

1. 두 개의 데이터프레임을 `고객ID`를 기준으로 내부 조인(`inner join`)하여 공통된 데이터를 병합하는 코드를 작성하세요.


```python
df1 = pd.DataFrame({'고객ID': [1, 2, 3], '이름': ['홍길동', '김철수', '이영희']})
df2 = pd.DataFrame({'고객ID': [2, 3, 4], '구매액': [10000, 20000, 30000]})

new_df = pd.merge(df1, df2, on='고객ID', how='inner')
print(new_df)
```

       고객ID   이름    구매액
    0     2  김철수  10000
    1     3  이영희  20000
    

2. 왼쪽 데이터프레임을 기준으로 병합(`left join`)을 수행하고, 구매액이 없는 경우 `NaN`을 유지하는 코드를 작성하세요.


```python
df1 = pd.DataFrame({'고객ID': [1, 2, 3], '이름': ['홍길동', '김철수', '이영희']})
df2 = pd.DataFrame({'고객ID': [2, 3, 4], '구매액': [15000, 20000, 30000]})

new_df = pd.merge(df1, df2, on='고객ID', how='outer')
print(new_df)
```

       고객ID   이름      구매액
    0     1  홍길동      NaN
    1     2  김철수  15000.0
    2     3  이영희  20000.0
    3     4  NaN  30000.0
    

3. 두 개 이상의 열을 기준으로 병합하고, 중복 열의 이름을 구분하도록 접미사(`suffixes`)를 지정하는 코드를 작성하세요.


```python
df1 = pd.DataFrame({
     '고객ID' : [1, 2, 3],
     '도시' : ['서울', '부산', '대전'],
     '구매액' : [10000, 20000, 30000]
})

df2 = pd.DataFrame({
     '고객ID' : [1, 2, 3],
     '도시' : ['서울', '부산', '광주'],
     '구매액' : [15000, 25000, 35000]
})

new_df = pd.merge(df1, df2, how='outer', on=['고객ID', '도시'], suffixes=('(df1)', '(df2)'))
print(new_df)
```

       고객ID  도시  구매액(df1)  구매액(df2)
    0     1  서울   10000.0   15000.0
    1     2  부산   20000.0   25000.0
    2     3  광주       NaN   35000.0
    3     3  대전   30000.0       NaN
    

#**결측치 처리(Missing Data)**

1. 주어진 데이터프레임에서 결측치를 탐지하고, 열별 결측치 개수를 출력하는 코드를 작성하세요.


```python
import numpy as np

data = {'이름' : ['홍길동', '김철수', np.nan, '이영희'],
        '나이' : [25, np.nan, 30, 28],
        '성별' : ['남', '남', '여', np.nan]}

df = pd.DataFrame(data)

print(df.isnull(), '\n', df.isnull().sum())
```

          이름     나이     성별
    0  False  False  False
    1  False   True  False
    2   True  False  False
    3  False  False   True 
     이름    1
    나이    1
    성별    1
    dtype: int64
    

2. 주어진 데이터프레임에서 결측치가 포함된 행을 삭제하고, 새로운 데이터프레임을 출력하는 코드를 작성하세요.


```python
data = {'이름' : ['홍길동', '김철수', np.nan, '이영희'],
        '나이' : [25, np.nan, 30, 28],
        '성별' : ['남', '남', '여', np.nan]}

df = pd.DataFrame(data)

new_df = df[df.isnull().sum(1) == 0]
print(new_df)
```

        이름    나이 성별
    0  홍길동  25.0  남
    

3. 결측치가 포함된 '나이' 열을 평균값으로 대체하고, 새로운 데이터프레임을 출력하는 코드를 작성하세요.


```python
data = {'이름' : ['홍길동', '김철수', np.nan, '이영희'],
        '나이' : [25, np.nan, 30, 28],
        '성별' : ['남', '남', '여', np.nan]}

df = pd.DataFrame(data)

df['나이'] = df['나이'].fillna(df['나이'].mean())
print(df)
```

        이름         나이   성별
    0  홍길동  25.000000    남
    1  김철수  27.666667    남
    2  NaN  30.000000    여
    3  이영희  28.000000  NaN
    

#**피벗(Pivot)**

1. 주어진 데이터프레임을 `pivot()`함수를 사용하여 날짜(`날짜`)를 행 인덱스로, 제품(`제품`)을 열로, 판매량(`판매량`)을 값으로 설정하는 코드를 작성하세요.


```python
data = {
     '날짜' : ['2024-01-01', '2024-01-01', '2024-01-02', '2024-01-02'],
     '제품' : ['A', 'B', 'A', 'B'],
     '판매량' : [100, 200, 150, 250]
}

df = pd.DataFrame(data)

print(df.pivot(index='날짜', columns='제품', values='판매량'))
```

    제품            A    B
    날짜                  
    2024-01-01  100  200
    2024-01-02  150  250
    

2. `pivot_table()`을 사용하여 주어진 데이터프레임에서 카테고리(`카테고리`)를 행 인덱스로, 제품(`제품`)을 열로 설정하고, 판매량(`판매량`)의 합계를 출력하는 코드를 작성하세요.


```python
data = {
     '카테고리' : ['전자', '가전', '전자', '가전'],
     '제품' : ['A', 'B', 'A', 'B'],
     '판매량' : [100, 200, 150, 250]
}

df = pd.DataFrame(data)

print(df.pivot_table(index='카테고리', columns='제품', values='판매량', aggfunc='sum'))
```

    제품        A      B
    카테고리              
    가전      NaN  450.0
    전자    250.0    NaN
    

3. 주어진 데이터프레임에서 여러 값을 동시에 피벗하여, `pivot()`을 사용해 날짜(`날짜`)를 행으로, 제품(`제품`)을 열로 설정하고, 판매량(`판매량`)과 이익(`이익`)을 동시에 피벗하는 코드를 작성하세요.


```python
data = {
     '날짜': ['2024-01-01', '2024-01-01', '2024-01-02', '2024-01-02'],
     '제품' : ['A', 'B', 'A', 'B'],
     '판매량' : [100, 200, 150, 250],
     '이익' : [20, 50, 30, 60]
}

df = pd.DataFrame(data)

print(df.pivot(index='날짜', columns='제품', values=['판매량', '이익']))
```

                판매량       이익    
    제품            A    B   A   B
    날짜                          
    2024-01-01  100  200  20  50
    2024-01-02  150  250  30  60
    

#**중복 제거(Duplicates Removal)**

1. 주어진 데이터프레임에서 기본적으로 중복된 행을 제거하는 코드를 작성하세요.


```python
data = {
     '이름' : ['김철수', '이영희', '김철수', '박민수'],
     '나이' : [25, 30, 25, 40],
     '성별' : ['남', '여', '남', '남']
}

df = pd.DataFrame(data)

df = df.drop_duplicates()
print(df)
```

        이름  나이 성별
    0  김철수  25  남
    1  이영희  30  여
    3  박민수  40  남
    

2. 다음 데이터프레임에서 특정 열을 기준으로 중복을 제거한 후, 중복 여부를 확인하는 코드를 작성하세요.


```python
data = {
     '제품' : ['노트북', '태블릿', '노트북', '스마트폰'],
     '가격' : [1500000, 800000, 1500000, 1000000],
     '카테고리' : ['전자기기', '전자기기', '전자기기', '전자기기']
}

df = pd.DataFrame(data)

df = df.drop_duplicates(subset='제품')
print(df.duplicated())
print(df)
```

    0    False
    1    False
    3    False
    dtype: bool
         제품       가격  카테고리
    0   노트북  1500000  전자기기
    1   태블릿   800000  전자기기
    3  스마트폰  1000000  전자기기
    

3. 다음 데이터프레임에서 중복된 모든 행을 삭제한 후, 남은 데이터를 저장하고 다시 불러오는 코드를 작성하세요.


```python
data = {
     '학생' : ['김민수', '박지현', '김민수', '이정훈'],
     '성적' : [90, 85, 90, 88],
     '학교' : ['A고', 'B고', 'A고', 'C고']
}

df = pd.DataFrame(data)

new_df = df.drop_duplicates()
print(new_df, '\n')
print(df)
```

        학생  성적  학교
    0  김민수  90  A고
    1  박지현  85  B고
    3  이정훈  88  C고 
    
        학생  성적  학교
    0  김민수  90  A고
    1  박지현  85  B고
    2  김민수  90  A고
    3  이정훈  88  C고
    

#**문자열 처리(String Operation)**

1. Pandas의 `.str`접근자를 사용하여 아래 시리즈의 모든 문자열을 소문자로 변환하는 코드를 작성하세요.


```python
data = pd.Series(["HELLO", "WOLRD", "PYTHON", "PANDAS"])

data = data.str.lower()
print(data)
```

    0     hello
    1     wolrd
    2    python
    3    pandas
    dtype: object
    

2. 다음 데이터프레임에서 '이름'컬럼의 문자열 앞 뒤 공백을 제거하고, 특정 문자열 "Doe"가 포함된 행을 필터링하는 코드를 작성하세요.


```python
df = pd.DataFrame({"이름" : [" John Doe ", "Alice ", " Bob", "Charlie Doe "]})

df['이름'] = df['이름'].str.strip()
print(df[df['이름'].str.contains('Doe', na=False)])
```

                이름
    0     John Doe
    3  Charlie Doe
    

3. 아래 데이터프레임에서 '설명' 컬럼의 문자열을 공백을 기준으로 나누고, 각 단어의 첫 글자만 추출하여 새로운 컬럼 '약어'를 생성하는 코드를 작성하세요.


```python
df = pd.DataFrame({"설명": ["빅데이터 분석", "데이터 과학", "머신 러닝", "딥 러닝"]})

df['약어'] = df['설명'].str.split().apply(lambda x: ''.join([word[0] for word in x]))
print(df)
```

            설명  약어
    0  빅데이터 분석  빅분
    1   데이터 과학  데과
    2    머신 러닝  머러
    3     딥 러닝  딥러
    
