---
title: 다변수 함수의 미분
date: 2025-02-17
categories: [Today I Learn, 4th Week]
math: true
---
## 다변수 함수의 미분
- 정의역을 기준으로 하는 함수의 종류와 미분
    $$
    f : \mathbb{R} \rightarrow \mathbb{R}, \, x_0 \in \mathbb{R}, \, L \in \mathbb{R} \\
    \lim_{x \rightarrow x_0} \frac{f(x)-f(x_0)}{x-x_0} = L = f'(x_0) \\
    \Leftrightarrow \lim_{x \rightarrow x_0} \frac{f(x)-f(x_0)}{x-x_0} - L = 0
    \\
    \Leftrightarrow \lim_{x \rightarrow x_0} \frac{f(x)-f(x_0)-L(x - x_0)}{x-x_0} = 0
    \\
    \Leftrightarrow \lim_{x \rightarrow x_0} \frac{f(x)-(f(x_0)+L(x - x_0))}{x-x_0} = 0
    $$

- 다변수 함수에서 선형 근사식<br/>
    1변수 함수에서의 선형근사
    $$
    y - f(x_0) = f'(x_0)(x-x_0) \Leftrightarrow y = f'(x_0)(x-x_0) + f(x_0)
    $$
    다변수 함수에서의 선형근사
    $$
    f : \mathbb{R^n} \rightarrow \mathbb{R^m}, \, x_0 \in \mathbb{R^n}, \, L : \mathbb{R^n} \rightarrow \mathbb{R^m} \\
    \\
    \Leftrightarrow \lim_{x \rightarrow x_0} \left\vert\left\vert\frac{f(x)-(f(x_0)+L(x - x_0))}{x-x_0}\right\vert\right\vert = 0
    $$
    미분 행렬, 야코비 행렬<br/>
    $L(x) = [L]x = D\cdot f(x_0)x$<br/>
    <br/>
    $x_0$에서의 선형근사<br/>
    $M(x) = f(x_0) + L(x-x_0) = f(x_0)+Df(x_0)(x-x_0)$

- 방향 도함수와 기울기 벡터(Gradient)<br/>
    방향 도함수<br/>
    $\vec u = \langle a,\,b \rangle$일 때,
    $$D_uf(x_0, y_0) = \lim_{h \rightarrow 0} \frac{f(x_0+ha, \, y_0+hb)-f(x_0, \, y_0)}{h}$$
    이변수함수 $f$가 $x$와 $y$의 미분가능한 함수이면 $f$는 모든 단위벡터 $\vec u = <a,\,b>$ 방향으로 방향 도함수가 존재하고 방향도함수 $D_uf = f_x(x,y)a + f_y(x,y)b$이다.<br/>
    <br/>
    기울기 벡터(Gradient)
    $\nabla f = \langle f_x(x,y),\, f_y(x,y)\rangle = \frac{\partial f}{\partial x}i + \frac{\partial f}{\partial y}j$
    
    $\therefore D_uf = \nabla f \cdot \vec u$<br/>
    $\left\vert\left\vert D_uf \right\vert\right\vert = \left\vert\left\vert \nabla f \cdot \vec u \right\vert\right\vert = \left\vert \nabla f \right\vert \left\vert \vec u \right\vert \left\vert \cos\theta \right\vert$

- 미분의 행렬 표현<br/>
    $ f \in P^2, \, f(x) = ax^2 + bx +c$ <br/>
    $ D(f(x)) = 2ax + b $<br/>
    $ \beta = \{1, x, x^2\}$<br/>
    $[D(f(x))] = \begin{bmatrix} 0&b&0 \\ 0&0& 2a \\ 0&0&0\end{bmatrix}$

- 단순 선형 회귀
    $ f(x) = wx + b $ <br/>
    $$ l(w, b) = \sum_{i=1}^n(f(x_i)-y_i)^2 = \sum_{k=1}^n (wx_i + b - y_i)^2 $$
    $$ \nabla l(w,b) = \left( \sum_{i=1}^n 2(wx_i + b - y_i)x_i, \, \sum_{i=1}^n 2(wx_i+b-y_i) \right) = (0, \,0)$$

- 다중 선형 회귀(다변수, 행렬식)<br/>
    $ Y = X\beta + \epsilon $ <br/>
    $ \epsilon = Y - \hat Y$ <br/>
    $ S = \epsilon' \epsilon$
    $ = (Y - X\beta)^T(Y-X\beta)$<br/>
    $\hat\beta = (X^TX)^{-1}X^TY $

- 경사하강법(Gradient Descent)<br/>
    1차 선형 근사식 : $x_{n+1} = x_n - \frac{f(x_0)}{f'(x_0)}$<br/>
    Gradient Descent : $X_{n+1} = X_n -\alpha \nabla f(X_n)$

## 오늘의 회고
- 선형 대수학과 더불어 다변수 함수에서의 미분을 하였고 미분을 통해 최소값을 찾는 과정을 살펴보면서 경사하강법까지 훑어볼 수 있는 시간이었다.