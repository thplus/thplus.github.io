---
title: 확률분포
date: 2025-02-12
categories: [Today I Learn, 3rd Week]
math: true
---
## 주제 1: 확률분포
- 확률(Probability)<br/>
    확률은 사건에 대한 수학과 통계의 한 분야로 한 사건의 확률은 0과 1사이이며 확률이 클 수록 사건이 일어날 확률이 클 것이다.
- 사건(Events)<br/>
    시행 결과의 부분집합이다. 사건은 서로 배타적이어야하며 어떤 시행을 하더라도 모든 사건 내에서 결과가 나와야한다.<br/>
    예를 들어, 동전을 던지는 것을 가정해보면 사건 E = {H, T}라고 할 수 있다. H와 T는 동시에 일어날 수 없는 서로 배타적이며 어떤 시행을 하더라도 H 혹은 T가 나온다.<br/>
    만약 H 혹은 T가 아니라 다른 것이 발생한다면 사건 E가 발생했다고 볼 수 없다.
- 표본공간<br/>
    시행의 모든 결과를 모은 집합
- 이산 표본공간<br/>
    표본공간 안에 있는 원소들이 모두 떨어져 있다.

    $$
    f : \Omega(이산 표본공간) \rightarrow [0,\;1]\\
    \displaystyle\sum_{x\,\in\, \Omega}f(x)\,=\,1
    $$

- 확률 질량 함수 : 표본공간의 한 점을 확률값에 대응시키는 함수<br/>

    $$
    E:사건,\, E\,\subset \, \Omega
    $$

    사건의 확률 $P(E)\,=\,\displaystyle\sum_{x\in E}f(x)$<br/>
- 연속 표본공간<br/>
    시행의 결과가 실수값(연속된 값)

    $$
    f : \Omega(연속 표본공간) \rightarrow [0,\;\infin)\\
    \int_{x\,\in\, \Omega}f(x)\,=\,1
    $$

- 확률 밀도 함수 : 표본공간의 한 구간을 확률값에 대응시키는 함수

    $$
    E:사건,\, E\,\subset \, \Omega
    $$

    사건의 확률 $P(E)\,=\, \int_{x \in E}f(x)\,dx$

- 정규분포(Normal Distribution)
    연속 확률 분포 중 하나로 무작위로 수집된 자료가 실수 값 분포하는 것이다.
    정규분포의 확률밀도 함수
    
    $$
    f(x) = \frac{1}{\sigma \sqrt{2\pi}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
    = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}
    $$

## 오늘의 회고
- 확률분포를 배우면서 AI를 수학적으로 접근하는 부분을 좀 더 심도있게 학습할 수 있었다.