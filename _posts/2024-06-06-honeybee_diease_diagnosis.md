---
title: 꿀벌 질병 진단 AI 서비스 review
date: 2024-06-06
categories: [Project, Ajou University]
tags: [python, tensorflow, matplotlib, yolo]
math: true
---

## 주제 선정
- 꿀벌은 농작물 생산을 위해 꽃가루를 암술에 묻혀 수정을 돕는 화분매개 역할을 한다. 국내에서는 한 해 평균 61만 개의 화분 매개용 벌통이 농작물 수분에 사용되고 있다. 특히 딸기, 토마토 등 과채류에서의 사용률은 67%에 달한다. 이처럼 화분매개는 농작물 생산에 꼭 필요한 과정이며 화분매개벌의 생존 기간과 활동은 농작물 생산에 크게 영향을 미친다.

- 꿀벌은 약 32조 4,000억 ~ 81조 2,000억 원의 가치를 생산하는 것으로 추정되며 이는 화분매개동물 중 가장 높은 경제적 가치에 해당한다. 우리나라에서는 화분매개 동물의 경제적 기여는 약 6조원으로 추정되며 이중 4분의 3이 꿀벌에 의한 기여로 화분매개동물의 활동성은 생태계 보전과 세계 경제 유지 그리고 인간 생존에 있어 핵심적인 요소라고 분석된다.

- 이처럼 중요한 꿀벌의 집단 폐사 현상은 2006년 미국에서 최초로 발견된 이후 우리나라를 비롯한 세계 곳곳에서 수해 동안 이어지며 특히 올해 상황은 더 심각할 수 있다는 현장 목소리가 잇따르고 있다. 그린피스와 한국양봉협회에 따르면 2023년 4월 기준 협회 소속 농가 벌통 153만 7,000여 개 가운데 61%인 94만 4,000여 개에서 꿀벌이 폐사한 것으로 추산했다. 올해는 월동률이 농가 평균적으로 60%는 됐지만 0%인 농가가 4 곳으로 꿀벌 집단 폐사 규모는 점차 커지고 있는 추세이다.
    ![alt text](/assets/images/honeybee_1.png)

- 이러한 꿀벌의 집단 폐사 원인은 정확히 알 수 없으나 국내외 연구자들과 정부의 발표에 의하면 3가지 원인을 뽑고 있다.

- 첫 번째는 **기후변화**이다. 한 연구에 의하면 극한의 한파, 온화한 겨울 날씨, 3월의 서리 등 최근 이상기후 현상들이 온도에 민감한 꿀벌의 사망과 높은 상관관계를 가지고 있다고 판단했다.

- 두 번째는 **진드기 및 바이러스에 의한 질병 문제**이다. 우리나라 정부는 피해의 주요 원인을 전염병을 일으키는 진드기의 확산을 꼽았다. 전염되기 쉬운 벌집 환경과 전염 속도가 빠른 질병의 특징이 꿀벌의 집단 폐사를 일으킨다.

- 세 번째는 **식물 살충제**이다. 살충제에 만성적으로 노출된 꿀벌은 면역 체계가 약해지고 채밀 능력 이상을 겪을 수 있다.

## 프로젝트 목표
- 3가지 원인을 바탕으로 꿀벌을 지키기 위한 우리나라의 노력으로는 농촌진흥청에서 개발한 '화분매개용 스마트벌통'이다. 화분매개용 스마트벌통은 벌통에 각종 감지기(센서)를 적용해 온도 및 이산화탄소 농도 등 벌통 내부 환경을 최적으로 유지한다. 이를 통해 기후변화 문제로 인한 꿀벌의 집단 폐사 문제를 해결하고자 현장에 적용하고 있다. 또한 이미지 심화 학습 기술을 이용해 일정 시간동안 벌통에서 출입하는 벌의 전체 수를 계산하여 벌의 활동량을 파악한다. 하지만 두 번째 원인인 질병 문제를 해결하기 위해 적용된 기술은 아직 없다.

- 따라서 본 프로젝트에서는 두 번째 질병 문제를 해결하기 위해 이미지 처리 기술을 이용한 꿀벌 질병 진단 모델을 구출하여 질병을 조기 진단하는데 일조하고자 한다.

## 데이터셋
- AI Hub의 [지능형 양봉 데이터](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=71488)를 활용한다.

    | 구분            | 생애단계별 Bounding Box | 생애단계별 polygon Segmentation | 파괴데이터 Bounding Box | 합계   | 비율   |
    |---------------|----------------------|----------------------------|----------------------|------|------|
    | 알            | 20,150                | 1,900                      | 1,900                | 23,950 | 8.73% |
    | 애벌레        | 20,150                | 1,900                      | 1,900                | 23,950 | 8.73% |
    | 번데기        | 20,150                | 1,900                      | 1,900                | 23,950 | 8.73% |
    | 수일벌-이탈리안 | 20,150                | 1,920                      | 1,920                | 23,990 | 8.75% |
    | 수일벌-카니올란 | 20,150                | 1,920                      | 1,920                | 23,990 | 8.75% |
    | 수일벌-한봉    | 20,150                | 1,916                      | 1,920                | 23,968 | 8.74% |
    | 수일벌-호박벌  | 20,090                | 1,920                      | 1,920                | 23,930 | 8.73% |
    | 여왕벌-이탈리안 | 20,150                | 1,920                      | 1,920                | 23,990 | 8.75% |
    | 여왕벌-카니올란 | 20,150                | 1,920                      | 1,920                | 23,990 | 8.75% |
    | 여왕벌-한봉    | 20,150                | 1,920                      | 1,920                | 23,990 | 8.75% |
    | 여왕벌-호박벌  | 20,150                | 1,920                      | 1,920                | 23,990 | 8.75% |
    | 질병(백묵병)  | 10,500                | -                          | -                    | 10,500 | 3.83% |
    | **합계**      | **232,086**            | **21,060**                  | **21,060**            | **274,206** | **100%** |

- 백묵병이 발생하지 않은 경우를 class 0, 백묵병이 발생한 경우를 class 1, 두 개의 class로 나누어져 있다. 각 클래스별 이미지 예시는 아래와 같다.

- Class 0 예시
    ![alt text](/assets/images/honeybee_claas0.jpg)

- Class 1 예시
    ![alt text](/assets/images/honeybee_class1.jpg)

## CNN을 이용한 분류
- AI Hub에서 제공하는 데이터는 이미지 데이터와 라벨링 데이터로 제공이 되며 라벨링 데이터를 통해 각 이미지의 클래스를 구분하여 CNN 모델 학습을 진행하였다.

- 학습용 데이터셋은 백묵병이 없는 데이터(Class 0) 1040개, 백묵병이 발생한 데이터(Class 1) 1000개를 통해 학습을 진행하였다.

- 테스트 데이터셋은 백묵병이 없는 데이터(Class 0) 400개, 백묵병이 발생한 데이터(Class 1) 400개를 사용하였다.

- 모델 구성은 아래와 같이 Convolution Layer 3개, Max Pooling Layer 3개로 구성하였다.

    ```python
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (224, 224, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    ```

- 이후에는 모델을 4개층부터 6개층까지 다양하게 바꿔가며 학습을 진행하였고 모델 평가를 진행했을 때, Accuracy 값이 0.99 이상으로 매우 높은 값이 도출되었다.
    ![alt text](/assets/images/honeybee_model_eval.png)

- 결과 예시
    ![alt text](/assets/images/honeybee_pred0.png)
    ![alt text](/assets/images/honeybee_pred1.png)

## 문제점
- Accuracy가 굉장히 높아 모델 성능이 높다는 점은 좋으나 이러한 현상이 발생한 가장 큰 이유는 데이터의 획일성 때문으로 추측된다. CNN학습을 위해 확보한 데이터셋은 몇몇 일관된 환경에서 촬영된 이미지이다. 즉, **백묵병을 진단했다기 보다는 해당 환경을 분류했을 가능성이 높다.**

- 기간이 짧은 프로젝트로 다양한 환경에 대한 데이터를 추가적으로 확보하기 쉽지 않으며 단순 CNN 구조로 꿀벌의 질병을 판단할 순 있지만 실사용을 하는데 어려움을 갖는다.

- 앞서 설명한 바와 같이 꿀벌의 질병은 전염성이 빠르며 조기에 발견하는 것이 가장 중요하다. 또한 질병을 확인하기 위해서는 벌통을 열어 확인해야하는데, 이 시점에서 **해당 모델의 필요성은 거의 없다고 봐도 무방하다.**

## 개선 방향
1. 환경을 분류하지 않아야 한다.

2. 실시간으로 꿀벌의 질병을 감지해야 한다.

- 위 두 가지를 해결할 수 있는 모델이 YOLO이기 때문에 YOLO를 이용해 프로젝트를 다시 구성하였다.

### YOLO를 선택한 이유
- YOLO는 이미지 전체를 한 번에 처리하여 객체를 감지하므로 단일 대상의 특징뿐만 아니라 이미지 전체의 맥락을 학습하게 된다. 객체 탐지에 초점을 맞추기 때문에 환경적 요소에 덜 민감하며 객체 자체의 특징을 학습하는데 더 유리하다.

- YOLO는 이미지를 한 번만 보고 Object Detection을 수행하기 때문에 Real-time Object Detection이 필요한 상황에서 가장 먼저 고려해볼 수 있는 딥러닝 모델이다. 카메라를 통해 벌통 내부의 상황을 실시간으로 판단해야하는 실제 운영 조건을 고려했을 때, 매우 적합한 모델이라고 판단하였다.

- 객체 탐지 모델이기 때문에 다른 환경에서의 이미지 데이터를 추가적으로 활용할 수 있다. 따라서 다른 class를 추가하여 추가적인 질병과 위협을 탐지할 수 있다.

- 추가적으로 백묵병 외에도 가장 위협이 된다는 바로아 기생충과 말벌을 추가적으로 학습하였다.

## Class정의
1. 백묵병(Chalkbrood)<br/>
    곰팡이 균사(병원체; Ascosphaera apis)가 자라면서 유충의 체액이 말라 백묵과 같이 굳어지는 질병으로 감염 시 급속히 증식하여 서서히 봉세를 약화시켜 정상 봉군에 이르지 못하게 만든다.<br/>
    ![alt text](/assets/images/honeybee_chalkbrood.png)

    - 데이터 셋은 [#데이터셋](#데이터셋) 참조

2. 애벌레(Larva)<br/>
    꿀벌의 생애 단계는 알, 애벌레, 번데기, 성충이다. 백묵병은 꿀벌의 생애 단계 중 애벌레에서 발생하는 질병 중 하나이기 때문에 애벌레를 '백묵병 음성'의 대상으로 선정하였다.<br/>
    ![alt text](/assets/images/honeybee_larva.png)

    - 데이터 셋은 [#데이터셋](#데이터셋) 참조

3. 바로아 응애(Varroa destructor)<br/>
    기생충 진드기의 일종으로 꿀벌이 꽃에 꿀을 따러 올 때, 몸에 착 달라붙어 벌집으로 이동한다. 벌집으로 이동한 꿀벌을 통해 애벌레, 번데기 및 성충에 기생하여 체액을 빨아먹으며 번식을 시작한다.<br/>
    ![alt text](/assets/images/honeybee_varroa.png)

    - 데이터 셋은 [해당 링크](https://zenodo.org/records/4085044)를 활용하였다.

4. 꿀벌(Honey Bee)<br/>
    모델 구축의 기본적인 식별 대상이자 말벌의 구분 대상이며 바로아 응애는 꿀범의 몸에 붙어 기생하기 때문에 선정하였다.<br/>
    ![alt text](/assets/images/honeybee_honeybee.png)

    - 데이터 셋은 모든 데이터를 활용하였다.

5. 말벌(Hornet)<br/>
    말벌은 꿀벌을 전문적으로 잡아먹는 포식자로 먹이의 70% 이상이 꿀벌이다. 10개의 봉군이 공격을 받아 폐사하는데 1주일 정도 걸리며 2~3주면 50개의 봉군이 폐사한다. 말벌의 여왕벌 한 마리를 잡으면 500마리 이상의 말벌을 방제하는 효과가 있다. 따라서 유인트랩, 개체 포살, 말벌집 제거를 통한 신속한 방제가 필요하다.<br/>
    ![alt text](/assets/images/honeybee_hornet.png)

## YOLO를 이용한 분류
- YOLO는 `.json`을 이용하여 라벨링을 하지 않는다. 따라서 제공되는 데이터에 대한 라벨링을 다시 해야했다. 라벨링은 `labelImg` 프로그램을 사용했으며 예시는 아래와 같다.

    ![alt text](/assets/images/honeybee_labeling.png)

- yaml 파일 만들기

    ```python
    import yaml

    data = { 'train' : 'train data set 주소',
            'val' : 'validation data set 주소',
            'test' : 'test data set 주소',
            'names' : ['Larvae', 'Chalkbrood', 'Bee', 'Varroa', 'Hornet'],
            'nc' : 5}

    with open('yaml 파일 주소', 'w') as f:
    yaml.dump(data, f)

    with open('yaml 파일 주소', 'r') as f:
    dataset_yaml = yaml.safe_load(f)
    display(dataset_yaml)
    ```

- 우리는 해당 `YOLOv8 nano`를 이용해 전이학습을 하였다.

    ```python
    from ultralytics import YOLO

    model = YOLO('yolov8n.pt')
    ```

- YOLO v8 custom data training
    
    ```python
    model.train(data='yaml 파일 주소', epochs=100, patience=30, batch=32)
    ```

- 전이 학습한 모델 확인

    ```python
    print(type(model.names), len(model.names))

    print(model.names)
    ```

    출력

    ```
    <class 'dict'> 5
    {0: 'Larvae', 1: 'Chalkbrood', 2: 'Bee', 3: 'Varroa', 4: 'Hornet'}
    ```

## YOLO 학습 결과
- 백묵병 검출결과<br/>
    ![alt text](/assets/images/honeybee_yolo_chalkbrood.png)

- 바로아 응애 검출결과<br/>
    ![alt text](/assets/images/honeybee_yolo_varroa1.png)<br/>
    ![alt text](/assets/images/honeybee_yolo_varroa2.png)

- 말벌 검출결과<br/>
    ![alt text](/assets/images/honeybee_yolo_hornet.png)

- 실시간 객체 탐지<br/>
    <video controls style="max-width: 100%; height: auto;">
    <source src="/assetes/videos/honeybee_video1" type="video/mp4">
    </video>

    <video controls style="max-width: 100%; height: auto;">
    <source src="/assetes/videos/honeybee_video2" type="video/mp4">
    </video>

## YOLO 성능평가
### Confusion Matrix
![alt text](/assets/images/honeybee_confusionmatrix.png)

- 결과 중 가장 주목해야 할 점은 background를 Larva로 인식한 값이 0.61이라는 것이다. 이는 Larva가 대체적으로 흰색을 띠고 있고, 사용한 이미지 데이터셋의 선명도 및 조명 등 데이터의 품질을 고려했을 때, 명확한 특징을 이용해 확인하기 어려운 한계가 있다고 판단하였다.

- 하지만 Larva는 질병이 없는 일반적인 상태로 질병이 Larva로 인식된 것이 아니라 background가 Larva로 인지한 것으로 이상치로 여기지 않았다.

- 또한 FP보다는 FN을 최소화하는 방향으로 모델 최적화를 시행하였다. 정상 상태를 질병으로 예측(FP)하는 것은 사람이 한 번 더 크로스 체크할 수 있지만 질병을 정상 상태로 예측(FN)하는 것은 위협을 빠르게 제거하기 힘들기 때문이다.

### PR Curve
![alt text](/assets/images/honeybee_prcurve.png)

- PR Curve는 ROC Curve와는 달리 True를 True로 예측하는 것에 초점을 많이 맞춘 성능 지표로 정확히 True로 선별하는 것이 중요한 질병 예측과 Object Detection 등에서 많이 활용되는 지표이다.

- 해당 그래프를 바탕으로 객체마다 결과를 해석하면 Larva(0.931)와 Bee(0.902)는 검출 성능이 높다는 것을 알 수 있다. Larva는 Labeling 결과를 보면 알 수 있듯이 압도적으로 많은 데이터를 사용하여 학습되었기 때문이라고 판단된다. Bee는 애벌레, 백묵병, 바로아 응애, 말벌 Dataset을 통해 가장 다양한 환경에서 수집되어 학습되었기 때문에 높은 성능을 보인다고 해석할 수 있다.

- Chalkbrood의 경우 0.867의 성능을 보이며 Recall의 값이 증가하더라도 Precision의 값이 0.9 정도로 일정한 값을 보여준다. 일반적으로 Precision과 Recall은 서로 trade-off 관계가 있는 경우가 많다. 하지만 Chalkbrood은 다양한 Recall 값에서도 높은 Precision을 유지하고 있다는 것을 알 수 있다. 이는 높은 비율의 True Positive (TP)와 낮은 비율의 False Positive (FP)의 성능을 보여주며 모델이 높은 성능을 유지하고 있음을 의미한다.

- 이와 반대로 Hornet의 경우 0.831의 성능을 보이지만 Recall의 값이 증가함에 따라 Precision의 값이 줄어들어 일반적인 trade-off 관계가 나타난다.

- 마지막으로 Varroa은 0.721로 가장 낮은 성능을 보여준다. Dataset에서 Varroa의 크기가 작고 해상도가 선명하지 않아 filter에서 정확히 특징이 걸러지지 않았을 것으로 추측된다. 이러한 이유로 confusion matrix에서도 Varroa를 background로 검출한 값이 0.42로 정확히 검출되지 않은 것을 확인할 수 있다. mAP(mean Average Precision)의 경우 Object detection의 정확도를 평가하는 지표로 사용되는데, mAP가 0.85로 모델의 전반적인 성능이 우수하다고 할 수 있다.

## 회고
- 본 프로젝트는 기후 변화, 질병, 살충제 세 가지 원인으로 인해 발생하는 꿀벌 집단 폐사 문제에 대해 '질병' 측면에 초점을 맞추었다. 이를 위해 꿀벌의 질병 진단을 수행하고, 더불어 꿀벌 생태계에 악영향을 미치는 '말벌'을 검출하기 위한 모델을 구축하였다.

- 해당 프로젝트에서 CNN은 Accuracy가 0.99로 아주 높은 성능을 보여주었다. 하지만 우리가 진행하는 프로젝트에는 성격이 맞지 않아 YOLO 모델로 중간에 바꾸어 진행하였다.프로젝트 기간이 짧아 모델을 바꾸는데 불안감이 있었지만 문제에 알맞은 솔루션을 찾아보면서 많은 것을 배웠다.

- 본 프로젝트는 앞으로 본 데이터로 학습된 라이브러리를 포함한 애플리케이션을 만들어 벌통 내부의 영상과 센서 데이터를 넣으면 질병의 초기 징후를 식별할 수 있을 것이다. 이 결과를 실시간으로 양봉주인에게 알람을 보내 적절한 방제를 통한 질병 피해를 최소화할 수 있을 것이다.

- 궁극적으로 양봉 농가의 생산성 향상에 기여할 수 있다. 질병 모니터링 예방을 통해 벌통 내 꿀벌의 건강을 유지하는데 도움을 줄 것이고 건강한 꿀벌은 양질의 꿀을 생산하는데 기여할 것이다.

- 정부는 올해부터 '양봉농가 질병관리 지원 사업'을 실시하여 응애 모니터링 방법, 꿀벌 질병 진단법 등에 대한 실습을 제공하고 있다. 그러나 수많은 벌집과 벌집 속 벌방을 양봉가가 하나씩 확인하는데는 많은 시간과 노동력이 필요하다. 따라서 조기 질병 진단 서비스를 도입할 경우 국가 예산 및 양봉가의 노동력과 시간을 절약할 수 있다.
