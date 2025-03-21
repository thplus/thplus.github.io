---
title: CNN 연구 개인프로젝트
date: 2025-03-31
categories: [Project, KTB]
tags: [python, langchain, fastapi]
math: true
---

## 서론
&nbsp;&nbsp;최근 다양한 CNN(Convolutional Neural Network) 기반의 이미지 분류 모델들이 제안되며 이미지 인식 분야에서 뛰어난 성능을 보이고 있다. 하지만 이러한 모델들은 구조의 복잡성이나 파라미터 수에 따라 연산 비용과 메모리 사용량에 큰 차이를 보이며, 특히 데이터셋의 특성에 따라 성능 및 효율성이 달라질 수 있다.<br/>
&nbsp;&nbsp;실제 응용 환경에서는 모델의 경량화 또한 중요한 과제로 떠오르고 있다. 모바일 기기, 임베디드 시스템, 자동화 기계 등에서는 모델의 정확도뿐만 아니라 처리 속도와 자원 효율성이 중요한 요소로 작용하기 때문이다.<br/>
&nbsp;&nbsp;따라서 본 프로젝트에서는 'Rice Image Dataset'을 활용하여 여러 CNN 기반 모델들이 해당 데이터셋에서 어떤 성능을 보이는지 비교하고 모델별 특징 및 효율성을 분석하고자 한다. 특히, 이미지 분류 과정에서 생성되는 Feature map을 시각화함으로써 각 모델이 어떤 방식으로 이미지를 인식하고 구분하는지 직관적으로 이해하고자 하였다.<br/>
&nbsp;&nbsp;이를 통해 모델 경량화 여부를 판단하여 보고 모델 경량화 기법을 일부 적용해 봄으로써 성능저하 없이 예측 효율성을 개선할 수 있는 가능성도 살펴보고자한다. 이러한 분석을 통해 단순 정확도 비교를 넘어서 실제 응용에 적합한 효율적인 모델을 선정하기 위한 방법을 살펴보고자 한다.<br/>

## 데이터셋 설명
&nbsp;&nbsp;본 프로젝트에서 사용한 데이터셋은 Kaggle의 'Rice Image Dataset'으로 Murat Koklu에 의해 제공되었다.<br/>
&nbsp;&nbsp;해당 데이터셋은 Arborio, Basmati, Ipsala, Jasmine, Karacadag로 총 5가지 class로 구분되어 있으며 각 데이터는 15,000개로 총 75,000개의 `.jpg` 이미지로 구성되어있다.<br/>
&nbsp;&nbsp; 각 이미지는 250x250 픽셀 크기를 가지며 검은 배경 위에 단일 쌀알이 위치한 형태로 구성되어 있다. 이미지들은 쌀알 이외의 잡음은 없고 배경과 객체가 명확히 구분되도록 전처리 되어있어 이미지 분류 모델 학습에 적합한 데이터이다.

### Data Load
(1)  Kaggle을 이용한 Data Load
  
```python
!kaggle datasets download -d nuratkokludataset/rice-image-dataset
!unzip rice-image-dataset.zip -d rice_dataset
```

(2)  Dataset 확인<br/>
  해당 데이터는 folder로 class화 해놓았기 때문에 folder별로 class index를 붙여주어야 한다.
  
```python
dataset_path = '.../rice_dataset/Rice_Image_Dataset'

class_folders = [f.name for f in os.scandir(dataset_path) if f.is_dir()]
num_classes = len(class_folders)

print("Class Folders:", class_folders)
print("Number of Classes:", num_classes)
```

```
Class Folders: ['Ipsala', 'Arborio', 'Karacadag', 'Jasmine', 'Basmati']
Number of Classes: 5
```

```python
path = pathlib.Path(dataset_path)

arborio = list(path.glob('Arborio/*.jpg'))
basmati = list(path.glob('Basmati/*.jpg'))
ipsala = list(path.glob('Ipsala/*.jpg'))
jasmine = list(path.glob('Jasmine/*.jpg'))
karacadag = list(path.glob('Karacadag/*.jpg'))

print(f'Arborio: {len(arborio)}')
print(f'Basmati: {len(basmati)}')
print(f'Ipsala: {len(ipsala)}')
print(f'Jasmine: {len(jasmine)}')
print(f'Karacadag: {len(karacadag)}')
```

```
rborio: 15000
Basmati: 15000
Ipsala: 15000
Jasmine: 15000
Karacadag: 15000
```

```python
basmati_img = img.imread(basmati[0])
arborio_img = img.imread(arborio[0])
ipsala_img = img.imread(ipsala[0])
jasmine_img = img.imread(jasmine[0])
karacadag_img = img.imread(karacadag[0])

fig,ax = plt.subplots(ncols=5, figsize=(20,5))
fig.suptitle ('Rice Category', fontsize=40)

ax[0].set_title("arborio")
ax[1].set_title("basmati")
ax[2].set_title("ipsala")
ax[3].set_title("jasmine")
ax[4].set_title("karacadag")
ax[0].imshow(arborio_img)
ax[1].imshow(basmati_img)
ax[2].imshow(ipsala_img)
ax[3].imshow(jasmine_img)
ax[4].imshow(karacadag_img)

plt.show()
```

![alt text](/assets/images/rice_category.png)

## CNN

## GoogLeNet

## VGG16

## MobileNet

## 결론

## Reference