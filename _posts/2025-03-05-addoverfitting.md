---
title: Pre-Trained Model
date: 2025-03-05
categories: [Today I Learn, 6th Week]
tags: [python, tensorflow, pytorch]
math: true
---

## Early Stopping
- Early Stopping은 기계 학습에서 경사 하강법과 같은 반복 방법으로 학습자를 훈련할 때, 과적합을 방지하기 위해 사영되는 정규화의 한 형태이다. 학습 과정에서 모델의 성능이 더 이상 개선되지 않을 때, 학습을 조기에 멈추어 Overfitting을 방지할 수 있다.

- Early Stopping은 모델의 Overfitting을 방지하고 학습시간을 줄일 수 있다. 또한 최적의 성능을 보이는 지점을 찾아 학습을 조기에 멈추기 위해 사용된다. Validation Loss나 Validation Accuracy와 같은 지표를 모니터링하여 더 이상 성능이 햐상되지 않는 시점에서 훈련을 중지해야한다.

## HyperParameter Tuning
- HyperParameter는 Learning Rate나 Optimizer 등과 같은 매개변수를 말하며 HyperParameter Tuning은 기계 학습 모델의 성능을 최적화하기 위해 모델의 Parameter값을 조정하는 과정을 뜻한다. 해당 과정은 모델의 성능을 향상시키고 주어진 데이터와 문제에 대해 최상의 예측 성능을 달하는 것을 목표로 한다.

- HyperParameter 목록

    |이름|내용|
    |--|--|
    |Learning Rate|가중치 업데이트의 크기를 결정하는 값|
    |Batch Size|한 번 학습할 데이터 샘플의 개수|
    |Epochs|전체 데이터셋을 몇 번 반복하여 학습할 지 결정|
    |Optimizer|모델 학습 과정을 최적화하는 알고리즘 선택|
    |Momentum|SGD와 같은 Optimizer에서 Gradient 업데이트에 관성을 추가하여 학습을 가속화하고 안정화|
    |DropOut Rate|학습 과정에서 무작위로 뉴런을 끄는 비율|
    |Regularization Parameter|과적합을 방지하기 위해 가중치에 패널티를 부여하는 값|
    |Learning Rate Decay|학습이 진행됨에 따라 학습률을 감소시키는 방법|
    |Activation Function|각 뉴런의 출력값을 결정하는 함수|
    |Initialization Method|가중치 초기화 방법|

## 오늘의 회고
- 어제 학습하지 못한 Early Stopping과 HyperParameter Tuning에 대해 학습하였다.