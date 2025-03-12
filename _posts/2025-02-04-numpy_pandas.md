---
title: NumPy & Pandas
date: 2025-02-04
categories: [Today I Learn, 2nd Week]
tags: [python, numpy, pandas]     # TAG names should always be lowercase
math: true
---

## 한 줄 정리

||한 줄 정리|비고|
|--|--|--|
|[Numpy](#numpy)|대규모 다차원 배열 및 행렬 연산을 위한 파이썬 라이브러리||
|[차원](#numpy에서의-차원)|배열에서 데이터를 구성하는 축의 개수|Scalar = 0, Vector = 1, Matrix = 2, Tensor = 3|
|[형태](#numpy에서의-shape)|배열의 모양, 각 차원별 요소의 개수|`reshape()`, `resize()`메서드로 shape 변환 가능|
|[인덱스](#numpy에서의-인덱싱)|배열 안에서 인자의 위치|`start:end:step`구조로 slicing 활용|
|[유니버설 함수](#numpy에서의-연산)|반복적으로 수행되는 벡터 연산을 제공하는 함수||
|[Pandas](#pandas)|구조화 된 데이터 프레임 객체를 조작하거나 분석하기 위한 파이썬 라이브러리||
|[Series](#pandas에서의-series)|1차원 배열 형태의 데이터 구조|Vector 형태라고 볼 수 있다.|
|[데이터 프레임](#pandas에서의-데이터프레임)|2차원 테이블 형태의 데이터 구조(행과 열로 구성)|Matrix 형태라고 볼 수 있다.|
|[그룹화](#pandas에서의-그룹화)|데이터를 일정 기준에 따라 변환하거나 연산|`groupby()`메서드 활용|
|[Merge](#pandas에서의-merge)|여러 데이터 프레임을 공통 열이나 행으로 결함|`merge()`메서드 활용|
|[Pivot](#pandas에서의-pivot)|데이터 프레임을 일정 기준에 따라 재구성하는 과정으로 행과 열을 재배치|`pivot()`, `pivot_table()`메서드 활용|

## NumPy
- NumPy는 대규모 다차원 배열 및 행렬 연산을 위한 <span style = "color:orange">**고성능 수학 함수와 도구를 제공**</span>하는 파이썬 라이브러리이다.<br/>
  데이터 구조를 기반으로 <span style = "color:orange">**수학 연산, 선형대수, 통계, 변환, 기본적인 수치 계산**</span> 등이 가능하도록 다양한 함수와 도구를 포함하고 있다.

- NumPy를 사용하는 이유는 대규모 수치 데이터를 빠르고 ```메모리 효율적```으로 처리하기 위해서이다.

- NumPy의 설치
    ```python
    pip install numpy
    ```
- NumPy는 자주 사용하는 라이브러리이다 보니 np로 호출하는 것이 관례이다.

    ```python
    import numpy as np
    ```
- NumPy에서의 간단한 배열 생성

    ```python
    import numpy as np

    a = np.array([1, 3, 5, 7])
    print("NumPy Array : ", a)
    ```
- 출력 결과
    ```
    NumPy Array : [1 3 5 7]
    ```

### NumPy에서의 차원
- 배열을 구성하는 축의 개수를 의미한다. 이와 같은 차원을 `.ndim`을 이용해 확인할 수 있다.
  
  ```python
  import numpy as np

  scalar = np.array(1)
  vector = np.array([1, 2, 3])
  matrix = np.array([[1, 2], [3, 4]])
  tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

  print(f"scalar = {scalar.ndim}")
  print(f"vector = {vector.ndim}")
  print(f"matrix = {matrix.ndim}")
  print(f"tensor = {tensor.ndim}")
  ```
  출력
  ```
  scalar = 0
  vector = 1
  matrix = 2
  tensor = 3
  ```

  |차원|명칭|예시|
  |--|--|--|
  |0차원|스칼라(Scalar)|```np.array(1)```&rarr;```1```|
  |1차원|벡터(Vector)|```np.array([1, 2, 3])```&rarr;```[1 2 3]```|
  |2차원|행렬(Matrix)|```np.array([[1, 2], [3, 4]])```&rarr;```[[1 2] [3 4]]```|
  |3차원|텐서(Tensor)|```np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])```|
  |n차원|n차원 배열|```np.array(...)```|

### NumPy에서의 Shape
  
- NumPy에서는 ```.shape```을 사용해 형태를 확인할 수 있다.
  
  ```python
  import numpy as np

  scalar = np.array(1)
  vector = np.array([1, 2, 3])
  matrix = np.array([[1, 2], [3, 4]])
  tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

  print(f"scalar = {scalar.ndim}")
  print(f"vector = {vector.ndim}")
  print(f"matrix = {matrix.ndim}")
  print(f"tensor = {tensor.ndim}")
  ```
  출력
  ```
  scalar = ()
  vector = (3,)
  matrix = (2, 2)
  tensor = (2, 2, 2)
  ```
- ```reshape()```를 사용해 기존 데이터를 유지한 채 새로운 형태로 배열을 변경할 수 있다.
  
  ```python
  import numpy as np

  array = np.array([1, 2, 3, 4])

  reshaped = array.reshape(2, 2)
  print(reshaped)
  ```
  출력
  ```
  [[1 2]
   [3 4]]
  ```
- ```resize()```를 이용해 원본 배열을 확장하거나 축소하는 등의 변경을 할 수 있다.
  
  ```python
  import numpy as np

  array = np.array([1, 2, 3, 4])

  array.resize(2, 3)
  print(array)
  ```
  출력
  ```
  [[1 2 3]
   [4 0 0]]
  ```
- ```flatten()```을 사용하여 다차원 배열을 1차원 배열로 변환할 수 있다.
  
  ```python
  import numpy as np

  array = np.array([[1, 2, 3], [4, 5, 6]])
  flattened_array = array.flatten()
  print(flattened_array)
  ```
  출력
  ```
  [1 2 3 4 5 6]
  ```
- `transpose()`를 사용해 데이터의 행과 열을 바꾸거나 다차원 배열의 축 순서를 변경할 수 있다.
  
  ```python
  import numpy as np

  array = np.array([[1, 2, 3], [4, 5, 6]])
  transposed = array.transpose()
  print(transposed)
  ```
  출력
  ```
  [[1 4]
   [2 5]
   [3 6]]
  ```

### NumPy에서의 데이터 타입
- NumPy에서 제공하는 데이터 타입은 C 언어의 기본 데이터 타입을 기반으로 하며 일반적인 파이썬의 데이터 타입보다 **메모리 사용량이 적고 연산 속도가 빠른 것**이 특징이다.
  
- `.dtype`속성을 통해 데이터 타입을 확인할 수 있고 배열 생성시 `dtype`인수로 지정하거나 `astype()`을 사용하여 변환할 수 있다.
  
  ```python
  import numpy as np

  int_array = np.array([1, 2, 3], dtype=np.int32)
  print(int_array.dtype)

  float_array = np.array([1.5, 2.3, 3.7], dtype=np.float64)
  print(float_array.dtype)

  string_array = np.array(["apple", "banana", "cherry"], dtype=np.str_)
  print(string_array.dtype)
  ```
  출력
  ```
  int32
  float64
  <U6
  ```
  ```python
  array = np.array([1, 2, 3], dtype=np.int32)

  # int32 -> float64로 변환
  converted_array = array.astype(np.float64)
  print(converted_array.dtype)

  # float64 -> int8로 변환 (데이터 손실 주의)
  converted_int_array = converted_array.astype(np.int8)
  print(converted_int_array.dtype)
  ```
  출력
  ```
  float64
  int8
  ```
  
### NumPy에서의 인덱싱
- Indexing은 배열의 특정 요소에 접근하거나 값을 참조하기 위한 방법이다.
- 반복문을 사용하지 않고 직접 요소에 접근할 수 있어 처리 속도가 향상된다. 또한 배열의 특정 부분만 선택하여 전체 배열을 복사할 필요가 없기 때문에 메모리 사용량을 절감할 수 있다.
- 정수 인덱싱
  
  ```python
  import numpy as np

  array = np.array([10, 20, 30, 40])
  print(array[0])
  print(array[-1])

  matrix = np.array([[1, 2, 3], [4, 5, 6]])
  print(matrix[1, 2])
  ```
  출력
  ```
  10
  40
  6
  ```
- 슬라이싱<br/>
  배열의 특정 범위를 선택하는 방법으로 `start:end:step` 형식으로 추출할 수 있다.

  ```python
  import numpy as np

  array = np.array([1, 2, 3, 4, 5])

  print(array[1:4])
  print(array[:3])
  print(array[::2])
  ```
  출력
  ```
  [2 3 4]
  [1 2 3]
  [1 3 5]
  ```
- 인덱싱을 활용한 데이터 수정<br/>
  위에서 배운 기본적인 인덱싱을 활용하여 아래와 같이 데이터를 수정할 수 있다.

  ```python
  import numpy as np

  array = np.array([1, 2, 3, 4])

  array[0] = 9
  print(array)

  array[1:3] = [5, 6]
  print(array)
  ```
  출력
  ```
  [9 2 3 4]
  [9 5 6 4]
  ```
### NumPy에서의 연산
- 요소별 연산<br/>
  NumPy는 배열의 각 요소에 대해 연산을 수행하며 동일한 크기의 배열에 대해 사칙 연산을 수행할 수 있다.
  ```python
  import numpy as np

  a = np.array([1, 2, 3])
  b = np.array([4, 5, 6])

  print(a + b)
  print(a - b)
  print(a * b)
  print(a / b)
  ```
  출력
  ```
  [5 7 9]
  [-3 -3 -3]
  [ 4 10 18]
  [0.25 0.4  0.5 ]
  ```
- 비교 연산<br/>
  배열의 요소의 값들을 비교하여 `bool`값을 `return`한다.
  ```python
  import numpy as np

  a = np.array([1, 2, 3])
  print(a > 2)
  print(a == b)
  print(a <= 2)
  ```
  출력
  ```
  [False False  True]
  [False False False]
  [ True  True False]
  ```
- 통계 연산<br/>
  NumPy는 배열의 집계 및 통계적 분석을 위한 다양한 함수를 제공한다.
  ```python
  array = np.array([1, 2, 3, 4, 5])

  print(np.mean(array)) #평균값
  print(np.median(array)) #중위값
  print(np.max(array)) #최대값
  print(np.min(array)) #최소값
  print(np.std(array)) #분산
  ```
  출력
  ```
  3.0
  3.0
  5
  1
  1.4142135623730951
  ```
- 선형대수 연산<br/>
  NumPy는 다양한 선형대수 함수를 제공한다.
  ```python
  matrix = np.array([[1, 2], [3, 4]])

  # 행렬 곱 (내적)
  vector = np.array([2, 3])
  print(np.dot(matrix, vector))

  # 역행렬 계산
  inverse = np.linalg.inv(matrix)
  print(inverse)
  ```
  출력
  ```
  [ 8 18]

  [[-2.   1. ]
   [ 1.5 -0.5]]
  ```
- 브로드캐스팅을 활용한 연산<br/>
  크기가 다른 배열 간의 연산을 수행할 때, NumPy는 작은 배열을 자동으로 확장하여 연산이 가능하도록 지원한다.
  ```python
  matrix = np.array([[1, 2, 3], [4, 5, 6]])
  vector = np.array([1, 2, 3])

  result = matrix + vector
  print(result)
  ```
  출력
  ```
  [[2 4 6]
   [5 7 9]]
  ```
- 유니버설 함수 <br/>
  NumPy에서 유니버설 함수는 배열의 각 요소에 대해 반복적으로 수행되는 벡터화된 연산을 제공하는 함수이다. 유니버설 함수는 여러 개의 입력 배열을 받아 연산을 수행가고 하나 이상의 출력을 반환할 수 있다.

  |유형|함수|
  |--|--|
  |산술 연산|`np.add()`,`np.substract()`,`np.multiply()`,`np.divide()`|
  |삼각 함수|`np.sin()`,`np.cos()`,`np.tan()`,`np.arcsin()`|
  |지수 로그 함수|`np.exp()`,`np.log()`,`np.log10()`|
  |비교 연산|`np.greater()`,`np.less()`,`np.equal()`|
  |논리 연산|`np.logical_and()`,`np.logical_or()`,`np.logical_not()`|
  |비트 연산|`np.bitwise_and()`,`np.bitwise_or()`,`np.bitwise_xor()`|

  NumPy 유니버설 함수는 `np.함수명(배열, 옵션)`형식으로 사용된다. 배열의 각 요소에 대해 연산을 수행하고 `out`매개변수를 사용해 결과를 기존 배열에 저장할 수 있다.
  ```python
  import numpy as np
  
  array1 = np.array([1, 2, 3])
  result1 = np.exp(array1)
  
  print(result1)

  array2 = np.array([4, 5, 6])
  output_array = np.array([0, 0, 0])
  result2 = np.add(array1, array2, out = output_array)

  print(output_array)
  ```
  출력
  ```
  [ 2.71828183  7.3890561  20.08553692]
  [5 7 9]
  ```
  `out`을 사용할 때, `np.empty_like(배열)`을 사용할 수 있다. 이는 배열의 크기는 같고 안의 요소는 없는 것을 만든다고 생각할 수 있다.
  ```python
  import numpy as np

  a = np.array([1, 2, 3])
  result = np.empty_like(a)

  np.muliply(a, 2, out = result)
  print(result)
  ```
  출력
  ```
  [2 4 6]
  ```

## Pandas
- Pandas는 <span style = "color:orange">**구조화된 데이터의 조작과 분석**</span>을 위한 데이터프레임 및 시리즈 객체를 제공하는 파이썬 라이브러리이다. <span style = "color:orange">**방대한 양의 데이터를 효율적으로 처리하고 직관적인 방식으로 데이터를 변환**</span>할 수 있고 <span style = "color:orange">**데이터 조작과 분석을 손쉽게 수행**</span>할 수 있도록 강력한 기능을 제공한다.
  
- Pandas를 사용하는 이유는 데이터를 구조화하여 효과적으로 처리하고 분석할 수 있도록 하기 위해서이다.
- Pandas의 설치
  ```python
  pip install pandas
  ```
- Pandas 역시 자주 사용하는 라이브러리로 pd로 호출하는 것이 관례이다.
  ```python
  import pandas as pd
  ```
- Pandas에서의 간단한 데이터 프레임 생성
  ```python
  import pandas as pd

  data = {
    '이름': ['홍길동', '김철수', '신창섭'],
    '나이': [20, 15, 23],
    '도시': ['서울', '대전', '인천']
  }

  df = pd.DataFrame(data)
  print(df)
  ```
- 출력 결과
  ```
      이름  나이  도시
  0  홍길동  20  서울
  1  김철수  15  대전
  2  신창섭  23  인천
  ```
### Pandas에서의 Series
- Pandas에서 Series는 인덱스를 가지는 1차원 배열 형태의 구조로 DataFrame(데이터프레임)의 구성 요소이다.
- Series 객체는 아래와 같은 속성을 통해 데이터를 쉽게 확인하고 조작할 수 있다.

  |속성|설명|예제|
  |--|--|--|
  |`value`|Series의 데이터 값을 ndarray 형식으로 return|`series.values`|
  |`index`|Series의 인덱스를 return|`series.index`|
  |`dtype`|Series의 데이터 유형을 return|`series.dtype`|
  |`shape`|Series의 크기를 return|`series.shape`|
  |`size`|Series의 총 요소 수를 return|`series.size`|
  |`name`|Series 객체의 이름 설정 및 확인|`series.name = '이름'`|

- Series 객체는 다양한 방법으로 생성할 수 있다.<br/>
  Series 객체를 생성할 때, 앞 글자가 대문자인 것을 항상 확인해야한다. `Series`
  ```python
  series1 = pd.Series([1, 2, 3, 4])
  print(series1)

  series2 = pd.Series({'a':10, 'b':20, 'c':30})
  print(series2)

  series3 = pd.Series([100, 200, 300], index=['x','y','z'])
  print(series3)

  series4 = pd.Series([1.5, 2.5, 3.5], dtype = 'float64')
  print(series4)
  ```
  출력
  ```
  0    1
  1    2
  2    3
  3    4
  dtype: int64
  a    10
  b    20
  c    30
  dtype: int64
  x    100
  y    200
  z    300
  dtype: int64
  0    1.5
  1    2.5
  2    3.5
  dtype: float64
  ```

- Series 기본 속성 활용
  ```python
  import pandas as pd

  s = pd.Series([10, 20, 30, 40], index=['a', 'b', 'c', 'd'])

  print(s.values)  # Series의 값 출력
  print(s.index)   # 인덱스 확인
  print(s.dtype)   # 데이터 타입 확인
  print(s.shape)   # 크기 확인 (튜플 형식)
  print(s.size)    # 요소 개수 확인
  s.name = 'Example Series'  # Series의 이름 설정
  print(s.name)    # Series의 이름 출력
  ```
  출력
  ```python
  [10 20 30 40]
  Index(['a', 'b', 'c', 'd'], dtype='object')
  int64
  (4,)
  4
  Example Series
  ```
- Series 데이터 연산<br/>
  NumPy 연산과 비슷하며 벡터 연산을 지원한다.
  ```python
  import pandas as pd

  s = pd.Series([1, 2, 3, 4])

  print(s + 10)
  print(s * 2)
  print(s[s > 2])
  ```
  출력
  ```
  0    11
  1    12
  2    13
  3    14
  dtype: int64
  0    2
  1    4
  2    6
  3    8
  dtype: int64
  2    3
  3    4
  dtype: int64
  ```
- 결측값 처리<br/>
  Series는 `.fillna()`메서드를 통해 결측값(`NaN`)을 자동으로 처리할 수 있다.
  ```python
  s = pd.Series([1, 2, None, 4])

  s_filled = s.fillna(0)
  print(s_filled)
  ```
  출력
  ```
  0    1.0
  1    2.0
  2    0.0
  3    4.0
  dtype: float64
  ```

### Pandas에서의 데이터프레임
- Pandas에서 데이터 프레임은 행과 열로 구성된 2차원 테이블 형태의 데이터 구조이다.<br/>
데이터 프레임을 사용하는 이유는 데이터를 체계적으로 구조화하여 효과적으로 조작하고 분석할 수 있도록 하기 위해서이다.

- 딕셔너리를 이용한 생성<br/>
앞서 소개한 Pandas를 이용한 데이터 프레임 생성을 통한 예시와 같다.<br/>
`pd.DataFrame()`메서드를 이용해 생성할 수 있으며 딕셔너리의 키가 열(column) 이름으로 설정되고 값들이 행(row)의 요소로 입력된다. 또한 인덱스는 자동으로 0부터 부여된다.
  ```python
  import pandas as pd

  data = {
    '이름': ['홍길동', '김철수', '신창섭'],
    '나이': [20, 15, 23],
    '도시': ['서울', '대전', '인천']
  }

  df = pd.DataFrame(data)
  print(df)
  ```
  출력
  ```
      이름  나이  도시
  0  홍길동  20  서울
  1  김철수  15  대전
  2  신창섭  23  인천
  ```
- 리스트를 이용한 생성<br/>
  리스트를 사용하여 데이터 프레임을 생성하면 열 이름이 없으므로 `pd.DataFrame()`메서드에 `columns`매개변수를 추가하여 열 이름을 지정할 수 있다.
  ```python
  import pandas as pd

  data = [[1, 2, 3],
          [4, 5, 6],
          [7, 8, 9]]
  df = pd.DataFrame(data, columns=['A', 'B', 'C'])
  print(df)
  ```
  출력
  ```
     A  B  C
  0  1  2  3
  1  4  5  6
  2  7  8  9
  ```
- NumPy를 이용한 생성<br/>
  NumPy 배열을 활용하면 대규모 수치 데이터를 효과적으로 처리할 수 있다.
  ```python
  import pandas as pd
  import numpy as np

  data = np.array([[1, 2, 3],
                   [4, 5, 6]])
  
  df = pd.DataFrame(data, columns = ['A', 'B', 'C'])
  print(df)
  ```
  출력
  ```
     A  B  C
  0  1  2  3
  1  4  5  6
  ```

- 데이터 프레임 기본 속성<br/>

  |속성|설명|
  |--|--|
  |`.head()`|첫 5개 행을 출력하여 데이터의 개요를 빠르게 확인|
  |`.tail()`|마지막 5개 행을 출력하여 최신 데이터를 검토|
  |`.shape`|행과 열의 개수를 튜플 형태로 반환|
  |`.columns`|열의 이름 목록 반환|
  |`.index`|행 인덱스 범위 반환|
  |`.info()`|데이터 타입, 결측치 여부 등을 요약|
  |`.decribe()`|수치형 데이터를 요약하여 평균, 표준편차, 최댓값 등을 제공|

- 데이터 접근<br/>
  데이터 프레임에서는 인덱스와 열 이름을 활용하여 특정 데이터에 쉽게 접근할 수 있다.
  ```python
  import pandas as pd

  data = {
    '이름': ['홍길동', '김철수', '신창섭'],
    '나이': [20, 15, 23],
    '도시': ['서울', '대전', '인천']
  }

  df = pd.DataFrame(data)
  print(df['이름'],'\n')
  print(df[['이름', '나이']], '\n')
  ```
  출력
  ```
  0    홍길동
  1    김철수
  2    신창섭
  Name: 이름, dtype: object 

      이름  나이
  0  홍길동  20
  1  김철수  15
  2  신창섭  23 
  ```
- Pandas에서는 `.iloc[]`과 `.loc[]`을 이용해서 데이터에 접근할 수 있다.
  `.iloc[]`은 정수기반 인덱싱으로 행과 열의 정수 위치를 사용하여 데이터에 접근한다. Python의 기본 슬라이싱과 유사하게 끝 인덱스는 포함하지 않는다.
  `.loc[]`은 레이블 기반 인덱싱으로 인덱스 레이블과 열 이름을 사용하여 데어터에 접근한다. 슬라이싱 할 때, 끝 레이블도 포함된다. 특정 인자 선택시 `.loc[row_index, column_name]`형식을 사용해 선택할 수 있다.
  ```python
  import pandas as pd

  data = {
    '이름': ['홍길동', '김철수', '신창섭'],
    '나이': [20, 15, 23],
    '도시': ['서울', '대전', '인천']
  }

  df = pd.DataFrame(data)

  df.index = ['a', 'b', 'c']
  print(df, '\n')
  print(df.iloc[1], '\n')
  print(df.loc['c'], '\n')
  ```
  출력
  ```
      이름  나이  도시
  a  홍길동  20  서울
  b  김철수  15  대전
  c  신창섭  23  인천 

  이름    김철수
  나이     15
  도시     대전
  Name: b, dtype: object 

  이름    신창섭
  나이     23
  도시     인천
  Name: c, dtype: object 
  ```
- 데이터 수정 및 연산<br/>
  아래 데이터 프레임으로 설명합니다.
  ```python
  import pandas as pd

  data = {
      "이름": ["김민수", "이영희", "박지훈", "최수빈", "정우성", "한예슬", "오지호", "황보승", "신세경", "강하늘"],
      "나이": [23, 21, 25, 24, 26, 22, 23, 24, 25, 22],
      "국어": [78, 85, 90, 88, 76, 95, 80, 82, 91, 87],
      "수학": [85, 90, 92, 88, 75, 93, 79, 84, 90, 89],
      "영어": [80, 82, 88, 90, 77, 94, 81, 86, 87, 90]
  }

  df = pd.DataFrame(data)
  ```
  특정 데이터 값 수정
  ```python
  df.loc[0, '영어'] = 55

  print(df.iloc[0])
  ```
  출력
  ```
  이름    김민수
  나이     23
  국어     78
  수학     85
  영어     55
  Name: 0, dtype: object
  ```
  새로운 열 추가
  ```python
  df['총점'] = df['국어'] + df['수학'] + df['영어']

  print(df)
  ```
  출력
  ```
      이름  나이  국어  수학  영어   총점
  0  김민수  23  78  85  55  218
  1  이영희  21  85  90  82  257
  2  박지훈  25  90  92  88  270
  3  최수빈  24  88  88  90  266
  4  정우성  26  76  75  77  228
  5  한예슬  22  95  93  94  282
  6  오지호  23  80  79  81  240
  7  황보승  24  82  84  86  252
  8  신세경  25  91  90  87  268
  9  강하늘  22  87  89  90  266
  ```
  그 외 기능은 아래 표 참조

  |속성|설명|
  |--|--|
  |`.drop('column_name', axis = 1, inplace = True)`|`columns_name`열 삭제|
  |`.sort_values(by = 'column_name, ascending = True)`|`column_name`열을 정렬, `ascending`이 `True`면 오름차순, `False`면 내림차순|
  |`.groupby('column_name)`|`column_name`으로 그룹화|
  |`.isnull()`|결측값 확인|
  |`.dropna(inplace = True)`|결측값 제거|
  |`.fillna()`|결측값을 다른 값으로 대체|
  |`.to_csv('file_name.csv')`|`file_name.csv`로 데이터프레임을 `csv`형식으로 저장|

### Pandas에서의 필터링
- 데이터 프레임이나 시리즈에서 특정 행이나 값을 선택하는 과정으로 대량의 데이터에서 원하는 정보를 효과적으로 추출하여 분석의 정확성을 높이기 위함이다.
- Boolean indexing<br/>
  `df[조건]`형태로 필터링을 수행하여 특정 컬럼의 값이 주어진 조건을 만족하는 경우 해당 행이 선택된다. 다중 조건 필터링을 위해 `&`,`|`,`~` 연산자를 사용할 수 있다.
  ```python
  import pandas as pd

  data = {
      "이름": ["김민수", "이영희", "박지훈", "최수빈", "정우성", "한예슬", "오지호", "황보승", "신세경", "강하늘"],
      "나이": [23, 21, 25, 24, 26, 22, 23, 24, 25, 22],
      "국어": [78, 85, 90, 88, 76, 95, 80, 82, 91, 87],
      "수학": [85, 90, 92, 88, 75, 93, 79, 84, 90, 89],
      "영어": [80, 82, 88, 90, 77, 94, 81, 86, 87, 90]
  }
  df = pd.DataFrame(data)
  
  print(df[df['나이'] >= 25], '\n')
  print(df[(df['국어'] >= 90) | (df['수학'] >= 90)], '\n')
  ```
  출력
  ```
      이름  나이  국어  수학  영어
  2  박지훈  25  90  92  88
  4  정우성  26  76  75  77
  8  신세경  25  91  90  87 

    이름  나이  국어  수학  영어
  1  이영희  21  85  90  82
  2  박지훈  25  90  92  88
  5  한예슬  22  95  93  94
  8  신세경  25  91  90  87 
  ```
- `query()`를 사용한 indexing<br/>
  SQL과 유사한 스타일로 데이터를 필터링 하는 방식으로 문자열 표현식을 통해 필터링이 가능하며 가독성이 높은 편이다.
  ```python
  #(...) Boolean indexing df 계속
  print(df.query('나이 >= 21 and 수학 <= 85'))
  ```
  출력
  ```
      이름  나이  국어  수학  영어
  0  김민수  23  78  85  80
  4  정우성  26  76  75  77
  6  오지호  23  80  79  81
  7  황보승  24  82  84  86
  ```
- 문자열 indexing `isin()`혹은`str.contains()`를 활용할 수 있다.<br/>
  `isin()` : 여러 값 중 하나라도 포함되어 있는지 여부를 확인<br/>
  `str.contains()` : 특정 문자열이 포함된 데이터를 필터링
  ```python
  print(df[df['이름'].isin(['김민수', '한예슬'])], '\n')
  print(df[df['수학'].isin([89, 90])], '\n')
  print(df[df['이름'].str.contains('신')], '\n')
  ```
  출력
  ```
      이름  나이  국어  수학  영어
  0  김민수  23  78  85  80
  5  한예슬  22  95  93  94 

      이름  나이  국어  수학  영어
  1  이영희  21  85  90  82
  8  신세경  25  91  90  87
  9  강하늘  22  87  89  90 

      이름  나이  국어  수학  영어
  8  신세경  25  91  90  87 
  ```

### Pandas에서의 그룹화
- Pandas 에서 grouping은 데이터를 특정 기준에 따라 그룹화하여 집계, 변환, 필터링 등의 연산을 수행하는 기능을 말한다. 그룹화는 기본적으로 `groupby()`메서드를 사용하여 수행된다. 그룹화는 데이터를 특정 기준으로 체계적으로 분류하여 유의미한 패턴과 통계를 도출하기 위해서 사용한다.
  ```python
  import pandas as pd

  data = {
      '이름': ['철수', '영희', '민수', '수진', '지훈', '하늘', '유리', '동현', '세진', '지우'],
      '부서': ['영업', '인사', '영업', 'IT', 'IT', '인사', '영업', 'IT', '인사', '영업'],
      '급여': [55000, 62000, 58000, 60000, 65000, 63000, 54000, 61000, 64000, 57000],
      '나이': [25, 30, 28, 35, 40, 32, 27, 38, 29, 31]
  }

  df = pd.DataFrame(data)
  
  mean_df = df.groupby('부서')['급여'].mean()
  print(mean_df, '\n')

  above_df = df.groupby('부서').filter(lambda x: x['급여'].mean() >= 60000)
  print(above_df)
  ```
  출력
  ```
  부서
  IT    62000.0
  영업    56000.0
  인사    63000.0
  Name: 급여, dtype: float64 

     이름  부서     급여  나이
  1  영희  인사  62000  30
  3  수진  IT  60000  35
  4  지훈  IT  65000  40
  5  하늘  인사  63000  32
  7  동현  IT  61000  38
  8  세진  인사  64000  29
  ```
  `[column_name].agg()`메서드를 사용하여 `sum`, `mean`, `max`, `min` 등의 집계를 할 수 있다.

### Pandas에서의 Merge
- Pandas에서 Merging은 여러 데이터프레임을 공통 열 또는 인덱스를 기준으로 결합하는 과정을 말한다. `merge()`메서드를 사용하여 데이터 프레임을 병합할 수 있다.<br/>
기본 문법은 `pd.merge(left, right, how='병합방식', on='기준 열')`이다.
  - `left`, `right` : 병합할 두 개의 데이터 프레임
  - `how` : 병합 방식
    1. `'inner'` : 공통된 키 값만 유지 $\bigcap$
    2. `'outer'` : 모든 데이터를 유지하며, 일치하지 않는 값은 `NaN`처리 $\bigcup$
    3. `'left'` : 왼쪽 데이터 프레임 기준으로 병합
    4. `'right'` : 오른쪽 데이터 프레임 기준으로 병합
  - `on` : 병합할 기준이 되는 공통열을 지정

- 데이터 프레임 연결 `concat()` 함수는 데이터를 단순히 연결하는 반면 `merge()`는 공통된 열을 기준으로 데이터를 결합한다.

### Pandas에서의 결측치 처리
- Pandas에서 결측치 처리는 데이터 프레임이나 시리즈에서 누락된 값을 탐지하고 제거하거나 대체하는 작업으로 결측치를 `NaN` 또는 `None`으로 표시하며 이러한 결측치를 탐지하고 적절히 처리하는 기능을 제공한다.<br/>
결측치 처리를 사용하는 이유는 데이터 분석의 정확성과 신뢰성을 확보하기 위해서이며 데이터 분석 및 머신러닝에서 데이터의 품질은 분석 결과의 정확성과 직결된다.

  |유형|메서드|설명|
  |--|--|--|
  |결측치 탐지|`isnull()`, `notnull()`, `info()`|어느 열과 행에 결측치가 있는지 탐색|
  |결측치 제거|`dropna()`|결측치를 포함한 행 또는 열을 제거|
  |결측치 대체|`fillna()`, `interpolate()`|특정 값으로 결측치 대체|

### Pandas에서의 Pivot
- Pandas에서 피벗은 데이터를 특정 기준에 따라 재구성하여 요약 통계를 계산하고 행과 열을 재배치하여 보다 쉽게 분석할 수 있도록 하는 과정을 말한다. 피벗을 사용하는 이유는 데이터를 특정 기준에 따라 재구성하여 의미있는 패턴을 발견하기 위해서로 데이터 프레임을 목적에 맞게 재구성하고 시각적으로 직관적인 형태로 만드는 것이다.<br/>
`pivot()`과 `pivot_table()`메서드를 사용하여 데이터를 특정 기준에 따라 재구성할 수 있다.<br/>
기본 문법은 `df.pivot(index = '행으로 설정할 column_name', columns = '열로 설정할 column_name', value = '값으로 설정할 column_name')`이다.
  ```python
  import pandas as pd

  data = {
      '날짜': ['2024-01-01', '2024-01-01', '2024-01-02', '2024-01-02'],
      '제품': ['A', 'B', 'A', 'B'],
      '판매량': [100, 200, 150, 250]
  }

  df = pd.DataFrame(data)
  print(df, '\n')

  df_pivot = df.pivot(index='날짜', columns='제품', values='판매량')
  print(df_pivot)
  ```
  출력
  ```
             날짜 제품  판매량
  0  2024-01-01  A  100
  1  2024-01-01  B  200
  2  2024-01-02  A  150
  3  2024-01-02  B  250 

  제품            A    B
  날짜                  
  2024-01-01  100  200
  2024-01-02  150  250
  ```

### Pandas에서의 중복제거
- 중복제거는 데이터의 정확성과 일관성을 유지하여 신뢰할 수 있는 분석 결과를 도출하기 위해 실행된다. `drop_duplicates()`를 사용하여 특정 열을 기준으로 중복된 행을 제거하거나 `duplicated()`로 중복 여부를 확인할 수 있다.

### Pandas에서의 문자열 처리
- 문자열 처리를 사용하는 이유는 비정형 문자열 데이털르 구조화하여 정확하고 일관된 분석이 가능하도록 하기 위해서이다. 문자열 처리의 주요 메서드는 아래와 같다.

  |메서드|설명|
  |--|--|
  |`str.lower()`|문자열을 소문자로 변환|
  |`str.upper()`|문자열을 대문자로 변환|
  |`str.strip()`|앞뒤 공백 제거|
  |`str.replace()`|특정 문자열을 다른 값으로 대체|
  |`str.contains()`|특정 문자열 포함 여부 확인|
  |`str.startswith()`|특정 문자열로 시작하는지 확인|
  |`str.endswith()`|특정 문자열로 끝나는지 확인|
  |`str.split()`|특정 구분자로 문자열 나누기|
  |`str.len()`|문자열 길이 반환|
  |`str.findall()`|정규 표현식 패턴과 일치하는 부분 검색|

- Python에서는 정규 표현식을 지원하기위해 re(regular expression) 모듈을 제공한다. 정규 표현식은 문자들로 이루어진 패턴을 정의하고 이 패턴을 사용해 특정 문자열을 찾거나 바꾸는 도구이다.<br/>
정규 표현식의 메타문자는 `. ^ $ * + ? { } [ ] \ | ( )`이며 자세한 내용은 [Python 정규식 HOWTO](https://docs.python.org/ko/3.13/howto/regex.html)참조

## 오늘의 회고
- 화요일은 배우는 양이 방대하므로 정리할 때, 모든 내용을 꼼꼼히 정리하기 보다는 필요한 내용을 잘 걸러서 정리해야 한다.
- NAS로 처음 .git에 add, commit하려면 보안 설정을 해야한다. 이 점도 항상 명시 해두어야 한다.
