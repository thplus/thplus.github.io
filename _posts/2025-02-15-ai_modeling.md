---
title: AI Modeling
date: 2025-02-15
categories: [Today I Learn, 3rd Week]
math: true
---

## AI 모델링
### 모델링이란
- 함수를 만드는 것
- 가장 대표적인 자연 함수 모델링 : $F = ma$
- AI 모델링의 기본은 분류이다.<br/>
    ```
               0
              /
    0.8 ____ /
             \
              \
               1
    ```
- 위의 예시를 보면 `0.8`을 `0`과 `1` 중 어디로 분류 해야하는지에 대한 문제이다.<br/> 많은 사람들이 `1`을 선택할 것이다. 이는 나도 모르게 $|0.8 - 1|$과 $|0.8 - 0|$을 해서 그 차이가 더 적은 `1`을 선택했을 것이다.<br/>
해당 내용이 우스워 보일 수 있지만 이는 AI에서 가장 중요한 개념인 `Loss Function`에 대한 부분을 인지하고 있었다는 점이다.

### Loss Function(손실함수)
- `Loss Function`이란 모델이 예측한 값과 실제 값의 차이를 측정하는 함수이다. 이를 통해 모델의 성능을 평가하고 모델이 어떤 방향으로 개선되어야 하는지 알려주는 역할을 한다. <br/>
즉, 바꿔 말하면 모델이 예측한 값과 실제 값의 차이인 `Loss Function`이 0에 가까울 수록 최소가 될 수록 해당 모델은 좋은 모델이 된다.

- 최소값을 찾기 위한 가장 좋은 도구는 미분이다.<br/>
우리가 실제 값을 $Y$, 예측값을 $\hat Y$라고 한다면 $f(x_1, x_2, \cdots, x_n) = |Y - \hat Y|$로 `Loss function`을 수학적 함수로 정의할 수 있다.<br/> 따라서 미분이 가능하다.

- `Loss Function`의 종류 : MSE(Mean Squared Error), Binary Cross-Entropy, Categorical Cross-Entropy 등이 있으며 해당 내용은 생략

- 미분값이 0인 점은 극값이지 최소값이 아니다. 3차원 이상 즉, 입력 변수 $x_n$에서 $n$이 3차 이상이면 최소값을 찾는 것이 매우 힘들다.<br/>
![alt text](/assets/images/3dexample.png)<br/>
해당 그래프를 보면 미분했을 때, 0이 되는 점이 많다는 것을 알 수 있고 이 중에 최소값을 알아내기는 더 힘들다.<br/>
![alt text](/assets/images/wave.png)<br/>
    >출처: [Pngtree](https://kor.pngtree.com/)<br/>

    위의 그림처럼 무한한 평면에 wave가 있는 함수라면 최소값 Global Minimum을 찾기란 매우 힘들다.

### Gradient Descent(경사 하강법)
- Gradient Descent(경사 하강법)란 함수의 최소값을 찾는 최적화 알고리즘이다. 해당 방법은 주로 머신러닝과 딥러닝에서 Loss Function을 최소화하는데 사용된다. Global Minimum을 찾기 위해 가장 빠르게 감소하는 지점을 찾아 움직이는 방법이다.

- 주요 개념은 함수의 Gradient $\nabla f$ 를 찾는다. 가장 빠르게 증가하는 방향으로 움직이는 벡터가 $\nabla f$ 이므로 가장 빠르게 감소하는 벡터인 $-\nabla f$ 를 이용해 함수의 최소값을 찾아나가면 된다.

- 1차 선형 근사식인 Newton's method를 기반으로 Gradient Descent를 업데이트 한다.<br/> Newton's Method: $x_{n+1} = x_n - \frac{f(x_0)}{f'(x_0)}$ <br/>
    Gradient Descent : $X_{n+1} = X_n -\alpha \nabla f(X_n)$ <br/>
    ![alt-text](/assets/images/Gradient_descent.gif)<br/>
    > 출처: [wikimedia](https://commons.wikimedia.org/wiki/File:Gradient_descent.gif)
    
- $\alpha$값을 어떻게 업데이트 하느냐에 따라서 SGD, Adam, RMSprop 등의 방법이 있다.

## 오늘의 회고
- 해당 내용을 통해서 AI 모델링의 기본 개념을 팀원들에게 설명하였다. Feed Forward와 Back Propagation의 과정도 설명하였으나 가장 중요한 개념은 `Loss Fuction`을 이해하고 분류를 잘하기 위해 `Loss Function`의 최소값을 찾아가는 과정이므로 이에 대한 내용만 작성하였다.