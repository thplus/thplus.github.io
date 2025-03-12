---
title: 2주차 과제
date: 2025-02-05
categories: [Assignment, 2nd Week]
tags: [mini_quest, python, numpy, pandas]     # TAG names should always be lowercase
math: true
---

1. [차원(Dimension)](#차원dimension)
2. [형태(Shape)](#형태shape)
3. [데이터 타입(Datatype)](#데이터-타입data-type)
4. [인덱싱(Indexing)](#인덱싱indexing)
5. [연산(Operation)](#연산operation)
6. [유니버설 함수(Universal Function)](#유니버설-함수universal-function)

# **차원(Dimension)**

1. 다음 NumPy 배열의 차원 수를 출력하는 코드를 작성하세요.


```python
import numpy as np

array = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

print(array.ndim)
```

    3
    

2. 1차원 배열 [10, 20, 30, 40, 50, 60]을 2차원 배열로 변환하여 (2, 3) 형태로 출력하는 코드를 작성하세요.


```python
array = np.array([10, 20, 30, 40, 50, 60])
array = array.reshape(2, 3)
print(array)
```

    [[10 20 30]
     [40 50 60]]
    

3. 다음 1차원 배열에 새로운 차원을 추가하여 2차원 배열로 변환하고, 최종 배열의 차원 수를 출력하세요.
   - `newaxis`를 사용하여 배열의 차원을 추가하고, 새로운 차원에서의 배열 모양을 확인한 후 최종 차원 수를 출력하세요.


```python
array = np.array([7, 14, 21])
ex_array = array[:, np.newaxis]
print(ex_array, '\n')
print(ex_array.ndim)
```

    [[ 7]
     [14]
     [21]] 
    
    2
    

# **형태**

1. 주어진 2차원 NumPy 배열의 형태(Shape)를 출력하는 코드를 작성하세요.


```python
array = np.array([[1, 2, 3], [4, 5, 6]])
print(array.shape)
```

    (2, 3)
    

2. 1차원 배열 `[10, 20, 30, 40, 50, 60]`을 (2, 3)형태로 변경한 후, 새로운 배열의 형태를 출력하는 코드를 작성하세요.


```python
array = np.array([10, 20, 30, 40, 50, 60])
array = array.reshape(2, 3)

print(array, '\n')
print(array.shape)
```

    [[10 20 30]
     [40 50 60]] 
    
    (2, 3)
    

3. 다음 3차원 배열의 형태를 변경하여 (3, 2, 2) 형태로 조정하고, 최종 배열의 형태를 출력하는 코드를 작성하세요.


```python
array = np.array([
       [[1, 2], [3, 4]],
       [[5, 6], [7, 8]],
       [[9, 10], [11, 12]]
   ])
array = array.reshape(3, 2, 2)

print(array, '\n')
print(array.shape)
```

    [[[ 1  2]
      [ 3  4]]
    
     [[ 5  6]
      [ 7  8]]
    
     [[ 9 10]
      [11 12]]] 
    
    (3, 2, 2)
    

# **데이터 타입(Data Type)**

1. 아래의 NumPy 배열의 데이터 타입을 확인하는 코드를 작성하세요.


```python
array = np.array([10, 20, 30])
print(array.dtype)
```

    int64
    

2. 정수형 배열 `[1, 2, 3]`을 부동소수점형 배열로 변환하고 변환된 배열의 데이터 타입을 출력하는 코드를 작성하세요.


```python
array = np.array([1, 2, 3])
array = array.astype(np.float64)
print(array.dtype)
```

    float64
    

3. NumPy 배열 `[100, 200, 300]`을 `unit8`로 변환한 후, 메모리 사용량(바이트)을 출력하는 코드를 작성하세요.


```python
array = np.array([100, 200, 300], dtype=np.uint8)
print(array.nbytes)
```

    3
    

    <ipython-input-10-136ea0012e79>:1: DeprecationWarning: NumPy will stop allowing conversion of out-of-bound Python integers to integer arrays.  The conversion of 300 to uint8 will fail in the future.
    For the old behavior, usually:
        np.array(value).astype(dtype)
    will give the desired result (the cast overflows).
      array = np.array([100, 200, 300], dtype=np.uint8)
    

# **인덱싱(Indexing)**

1. 주어진 1차원 배열에서 첫 번째 요소와 마지막 요소를 출력하는 코드를 작성하세요.


```python
array = np.array([10, 20, 30, 40, 50])
print(array[0], array[-1])
```

    10 50
    

2. 주어진 2차원 배열에서 첫 번째 열과 두 번째 행을 출력하는 코드를 작성하세요.


```python
matrix = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])
print(matrix[0], matrix[1])
```

    [1 2 3] [4 5 6]
    

3. 주어진 배열에서 10보다 큰 요소들만 선택하고, 해당 요소들의 인덱스를 출력하는 코드를 작성하세요.


```python
array = np.array([5, 15, 8, 20, 3, 12])
print(np.where(array > 10))
```

    (array([1, 3, 5]),)
    

# **연산(Operation)**

1. 주어진 NumPy 배열에서 요소별(Element-wise) 덧셈을 수행하는 코드를 작성하세요.


```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(a + b)
```

    [5 7 9]
    

2. 다음 NumPy 배열에서 브로드캐스팅을 활용하여 각 행에 `[1, 2, 3]`을 더하는 코드를 작성하세요.


```python
matrix = np.array([[10, 20, 30], [40, 50, 60]])
vector = np.array([1, 2, 3])

print(matrix + vector)
```

    [[11 22 33]
     [41 52 63]]
    

3. 주어진 2차원 NumPy 배열에 대해, 축(axis)을 기준으로 최댓값을 구하고 최종 배열의 차원을 출력하는 코드를 작성하세요.


```python
array = np.array([[3, 7, 2], [8, 4, 6]])

print(array[0].max(), array[1].max(), array.ndim)
```

    7 8 2
    

# **유니버설 함수(Universal Function)**

1. 배열 `[1, 2, 3, 4]`에 대해 NumPy의 유니버설 함수를 사용하여 모든 요소를 제곱한 결과를 출력하는 코드를 작성하세요.


```python
array = np.array([1, 2, 3, 4])

sq_array = np.square(array)
print(sq_array)
```

    [ 1  4  9 16]
    

2. 다음 두 배열의 요소별 합을 계산하고, 결과를 새로운 배열이 아닌 기존 배열에 저장하는 코드를 작성하세요.


```python
array1 = np.array([10, 20, 30])
array2 = np.array([1, 2, 3])

out_array = np.empty_like(array1)
np.add(array1, array2, out=out_array)
print(out_array)
```

    [11 22 33]
    

3. 다음 배열에 대해 NumPy 유니버설 함수를 사용하여 요소별로 자연로그(log)를 계산하고, 특정 조건(`log 값 > 1`)을 만족하는 요소만 출력하는 코드를 작성하세요.


```python
array = np.array([1, np.e, 10, 100])

new_array = np.log(array)
print(array[new_array > 1])
```

    [ 10. 100.]
    
