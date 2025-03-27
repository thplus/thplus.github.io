---
title: CNN 경량화 연구
date: 2025-03-22
categories: [Project, KTB]
tags: [python, langchain, fastapi]
math: true
---

# 1. 서론
&nbsp;&nbsp;최근 다양한 CNN(Convolutional Neural Network) 기반의 이미지 분류 모델들이 제안되며 이미지 인식 분야에서 뛰어난 성능을 보이고 있다. 하지만 이러한 모델들은 구조의 복잡성이나 파라미터 수에 따라 연산 비용과 메모리 사용량에 큰 차이를 보이며, 특히 데이터셋의 특성에 따라 성능 및 효율성이 달라질 수 있다.<br/>
&nbsp;&nbsp;실제 응용 환경에서는 모델의 경량화 또한 중요한 과제로 떠오르고 있다. 모바일 기기, 임베디드 시스템, 자동화 기계 등에서는 모델의 정확도뿐만 아니라 처리 속도와 자원 효율성이 중요한 요소로 작용하기 때문이다.<br/>
&nbsp;&nbsp;따라서 본 프로젝트에서는 'Rice Image Dataset'을 활용하여 여러 CNN 기반 모델들이 해당 데이터셋에서 어떤 성능을 보이는지 비교하고 모델별 특징 및 효율성을 분석하고자 한다. 특히, 이미지 분류 과정에서 생성되는 Feature map을 시각화함으로써 각 모델이 어떤 방식으로 이미지를 인식하고 구분하는지 직관적으로 이해하고자 하였다.<br/>
&nbsp;&nbsp;이를 통해 더 이상 특징을 제대로 추출하지 못 하는 Layer를 일부 제거하거나 Filter의 수를 줄임 성능저하 없이 예측 효율성을 개선할 수 있는 가능성도 살펴보고자한다. 이러한 분석을 통해 단순 정확도 비교를 넘어서 실제 응용에 적합한 효율적인 모델을 선정하기 위한 방법을 살펴보고자 한다.<br/>

## (1) 모델 경량화
&nbsp;&nbsp;모델 경량화가 필요한 이유는 다양하지만 비용적 문제가 크다. 모델이 크고 계산량이 많아질수록 전력 소비량이 늘어나고 열 발생량도 높아진다. 심지어 최근에는 스마트폰, IoT 기기 등에도 탑재하고자 노력 중이다. 따라서 모델이 가볍고 성능은 뒤떨어지지 않는 것이 중요하다.<br/>
모델 경량화의 Solution은 Pruning, Quantization, Knowledg Distillation 등 다양하게 존재한다. 하지만 본 프로젝트에서는 Filter를 통과한 Featrue Map을 시각화하여 불필요한 Filter를 삭제해봄으로써 모델 경량화를 하고자 한다.<br/>

## (2) Feature Map
![alt text](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FOmgjJ%2FbtqXrD2js9s%2FGFjoiBuv70Hx53YKb9XOzK%2Fimg.png)<br/>

&nbsp;&nbsp;CNN은 Convolution Layer를 통과해 특징을 추출하게 되며 Filter를 통과하고 나온 형태를 Feature Map이라 한다. 해당 Feature Map은 특징을 강조하는 Pooling을 거치게 되는데, 이 과정이 끝나면 Fully Connected Layer로 들어가게 되어 우리가 흔히 아는 ANN구조와 같은 방식으로 작동한다.<br/>

&nbsp;&nbsp;예를 들어 고양이 사진을 CNN구조인 VGG16에 통과시키면 아래와 같은 Feature Map을 얻을 수 있다.<br/>
![alt text](/assets/images/cnnproject_catfeaturemap.png)<br/>
> [이미지 출처](https://m.blog.naver.com/luvwithcat/222148953574)| LifeofPy, CNN의 정의, 이미지의 feature map 기능과 kernel(filter)의 개념

&nbsp;&nbsp;하지만 본 프로젝트에서 사용하는 'Rice Image Dataset'의 경우 CNN의 층이 깊어질수록 불필요한 특징이 잡힐 수 있다. 해당 Dataset을 같은 VGG16에 통과시킨 후의 Feature Map을 보면 아래와 같이 block_1과 block_5에서의 차이를 볼 수 있다.<br/>
![alt text](/assets/images/cnnproject_vgg16_layer1.png)<br/>
&nbsp;&nbsp;위의 Feature Map은 VGG16의 첫 번째 Convolution Layer를 통과한 상태이다.<br/>
<br/>
![alt text](/assets/images/cnnproject_vgg16_layerfin.png)<br/>
&nbsp;&nbsp;위의 Feature Map은 VGG16의 마지막 Convolution Layer를 통과한 상태이다.
모든 Filter에 대한 Feature Map을 출력한 것은 아니지만 대부분의 Filter가 그냥 검정색임을 알 수 있다. 해당 현상이 문제가 되는 점은 Fully Connected Layer에서 발생할 수 있다.<br/>

## (3) Feed Forward 관점
&nbsp;&nbsp;Convolution → ReLU → MaxPooling 형태의 CNN 층을 수식으로 표현하면 아래와 같다.<br/>
$X \in \mathbb{R}^{C_{in}\times H \times W}$<br/>
$W \in \mathbb{R}^{C_{out}\times C_{in} \times K' \times K}$<br/>
$b_i \in \mathbb{R}$<br/>
&nbsp;&nbsp;출력 채널 $i$, 위치 $(m, n)$에서의 convolution 출력은 아래의 식과 같다.<br/>

$$
Z_i(m, n) = \sum_{c=1}^{C_{\text{in}}} \sum_{u=1}^{K} \sum_{v=1}^{K} W_{i,c,u,v} \cdot X_c(m + u, n + v) + b_i
$$

<br/>
&nbsp;&nbsp;ReLU는 음수를 0으로 만들고 양수는 그대로 유지하므로 아래의 식과 같다.<br/>

$$
\text{ReLU}(Z_i(m, n)) = \max(0, Z_i(m, n))
$$

<br/>
&nbsp;&nbsp; $2 \times 2$ 커널에서의 Max Pooling은 아래의 식과 같다.<br/>

$$
P_i(p, q) = \max_{\substack{0 \leq m < k \\ 0 \leq n < k}} A_i(s \cdot p + m,\, s \cdot q + n)
$$

<br/>
&nbsp;&nbsp; 따라서 모든 Feature Map의 값이 0인 경우를 보면 아래와 같이 출력된다.<br/>

$$
Z_i(m,n) = 0 \,\,\, \forall i,m,n \\
ReLU(Z_{i}(m,n))\max(0, Z_i(m,n)) = 0 \\
P_i(p, q) = \max(0, 0, 0, 0) = 0
$$
<br/>
&nbsp;&nbsp;이와 같이 Feature Map의 값이 모두 0이면 ReLU함수와 Pooling 층을 통과해도 그 결과는 0이 나온다.<br/>
&nbsp;&nbsp;Fully Connected Layer는 $y = Wx + b$와 같이 표현되고 모든 $x = 0$이라면 학습결과는 bias($b$)에 의존할 수 밖에 없으며 이는 추출된 특징으로 학습할 수 없다. 따라서 학습할 정보가 사라지므로 해당 모델은 찍기의 형태를 띌 수 밖에 없다.<br/>

## (4) Back propagation 관점
&nbsp;&nbsp;실제 학습을 위한 Back Propagation의 관점으로 생각해보면 FCL에서의 모든 $x = 0$이라면 Convolution Layer의 Filter는 바뀌지 않는다.<br/>

&nbsp;&nbsp;FCL이 $y = \phi(Wx+b)$라면 $x = 0$이므로 $y = \phi(b)$이다. 이에 따른 Gradient를 구해보면
$$
\frac{\partial L}{\partial W} = \frac{\partial L}{\partial y} \cdot \phi '(b) \cdot x^{T} = 0
$$
이므로 FCL의 weight는 학습되지 않는다.<br/>
&nbsp;&nbsp;입력 $X$에 대하여 $Z \rightarrow ReLU \rightarrow P$ 순으로 진행이 된다면 Convolution Layer의 weight $W_{conv}$의 Gradient는
$$
\frac{\partial L}{\partial W_{conv}} = \frac{\partial L}{\partial A} \cdot \frac{\partial A}{\partial Z} \cdot \frac{\partial Z}{\partial W_{conv}}
$$
이다. <br/>
&nbsp;&nbsp; 여기서 앞서 구한 $\frac{\partial L}{\partial W} = 0$이므로 $\frac{\partial Z}{\partial W_{conv}} = 0$ 이다. 따라서 Convolution Layer의 gradient가 0이 되어 학습이 되지 않은 채, Filter는 그대로 고정된 상태로 남는다.<br/>
&nbsp;&nbsp; 따라서 $\frac{\partial L}{\partial b}$인 bias만 업데이트 되어 의미 없는 학습만 계속하여 제대로 추론을 할 수 없게된다.

&nbsp;&nbsp;위와 같이 이론적으로는 Feature Map을 모두 보았을 때, 마지막 Feature Map의 값이 모두 0이라면 학습은 진행되지 않는다. 본 프로젝트에서 첨부한 Feature Map들은 gray scale로 출력하여 각 채널(R, G, B)별 실제 Feature Map의 값들이 전부 0인지는 시각적으로 확인할 순 없다. 하지만 GoogLeNet을 이용해 이를 확인할 수 있었다.<br/>
&nbsp;&nbsp;GoogLeNet 고유한 특징은 1x1 convolution 연산을 한다는 것이다. 1x1 convolution은 Feature Map의 크기를 바꾸지 않는다. 하지만 1x1 Filter를 통해 Feature Map의 수를 줄이는 것이 가장 큰 특징이다. 이를 바꾸어 이야기하면 R, G, B 각 채널의 특성들이 하나로 압축된다는 뜻이다. 또한 GoogLeNet은 총 27개의 Layer로 모델이 깊이 특성을 학습하여 실제 Feature Map들의 모든 값이 0이 될 확률이 다른 모델에 비해 클 수 있다.<br/>

&nbsp;&nbsp;아래의 내용은 실제 weight를 초기화 해보면서 학습을 시도한 결과 얻은 GoogLeNet 학습 결과이다.
![alt text](/assets/images/cnnproject_learningerror.png)<br/>
![alt text](/assets/images/cnnproject_learningerroraccuracy.png)<br/>
![alt text](/assets/images/cnnproject_learningerrorloss.png)<br/>

[학습 불가 Colab Link](https://colab.research.google.com/drive/1Wt_dMel9Vv8HwzGIlkuvochkIFNWLsv0?usp=sharing), [프로젝트 Colab Link](https://colab.research.google.com/drive/1WgjlYTLEkDyacbimKDHpdKw0LBZn4aTi?usp=sharing) 두 링크를 통해 비교해보면 코드는 같은 걸 알 수 있다.<br/>

&nbsp;&nbsp;따라서 본 프로젝트는 위의 결과를 토대로 시각적으로 Feature Map을 살펴보고 불필요한 Filter가 생성되는 Layer는 제거하거나 Filter의 개수를 줄여보면서 모델을 경량화하고 성능을 비교해보고자 한다.

# 2. 데이터셋 설명
&nbsp;&nbsp;본 프로젝트에서 사용한 데이터셋은 Kaggle의 'Rice Image Dataset'으로 Murat Koklu에 의해 제공되었다.<br/>
&nbsp;&nbsp;해당 데이터셋은 Arborio, Basmati, Ipsala, Jasmine, Karacadag로 총 5가지 class로 구분되어 있으며 각 데이터는 15,000개로 총 75,000개의 `.jpg` 이미지로 구성되어있다.<br/>
&nbsp;&nbsp; 각 이미지는 250x250 픽셀 크기를 가지며 검은 배경 위에 단일 쌀알이 위치한 형태로 구성되어 있다. 이미지들은 쌀알 이외의 잡음은 없고 배경과 객체가 명확히 구분되도록 전처리 되어있어 이미지 분류 모델 학습에 적합한 데이터이다.

## (1) Data Load
- Kaggle을 이용한 Data Load
  
```python
!kaggle datasets download -d nuratkokludataset/rice-image-dataset
!unzip rice-image-dataset.zip -d rice_dataset
```

- Dataset 확인<br/>
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

# 3. 모델 구성
## (1) CNN
- 일반적인 CNN으로 Convolution → ReLu → MaxPooling 으로 3개 층으로 쌓아보았으며 필터의 개수는 32 → 64 → 128 개를 사용하여 Feature map을 시각화 하였다.
  
  ```python
  def cnn(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)
    x = tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)
    x = tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    return model
  ```

- CNN 모델 Summary
  
  ```
  ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
  ┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
  ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
  │ input_layer (InputLayer)             │ (None, 224, 224, 3)         │               0 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ conv2d (Conv2D)                      │ (None, 224, 224, 32)        │             896 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ max_pooling2d (MaxPooling2D)         │ (None, 112, 112, 32)        │               0 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ conv2d_1 (Conv2D)                    │ (None, 112, 112, 64)        │          18,496 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ max_pooling2d_1 (MaxPooling2D)       │ (None, 56, 56, 64)          │               0 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ conv2d_2 (Conv2D)                    │ (None, 56, 56, 128)         │          73,856 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ max_pooling2d_2 (MaxPooling2D)       │ (None, 28, 28, 128)         │               0 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ flatten (Flatten)                    │ (None, 100352)              │               0 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ dense (Dense)                        │ (None, 128)                 │      12,845,184 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ dense_1 (Dense)                      │ (None, 5)                   │             645 │
  └──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
  Total params: 12,939,077 (49.36 MB)
  Trainable params: 12,939,077 (49.36 MB)
  Non-trainable params: 0 (0.00 B)
  ```
  
- Model Evaluation 시각화

  ![alt text](/assets/images/cnnproject_cnn_accuracy.png)<br/>

  ![alt text](/assets/images/cnnproject_cnn_loss.png)<br/>

  ```
  Restoring model weights from the end of the best epoch: 9.
  ```
  Best score인 9번 째 epoch의 evaluation은 아래와 같다.<br/>
  Train Accuracy: 0.9986<br/>
  Train Loss: 0.0040<br/>
  Validation Accruacy: 0.9963<br/>
  Validation Loss:  0.0132<br/>

- Feature Map 시각화

  ![alt text](/assets/images/cnnproject_cnn_layer1.png)<br/>

  ![alt text](/assets/images/cnnproject_cnn_layer2.png)<br/>
  
  ![alt text](/assets/images/cnnproject_cnn_layer3.png)<br/>

  &nbsp;&nbsp;부분적으로 filter을 거친 후 feature map이 검은색인 경우가 많다. 이는 계산에 큰 영향을 미치지 않을 것이라 판단되어 filter의 개수를 줄여 개선을 경량화를 해보고자 한다.

## (2) CNN 경량화
- 일반적인 CNN으로 Convolution → ReLu → MaxPooling 으로 2개 층으로 쌓아보았으며 필터의 갯수는 8 → 16 개를 사용하여 이전과 다르게 `strides` 추가하여 feature map의 크기를 줄여보았다.
  
  ```python
  def cnn_light(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(8, (3,3), activation='relu', strides = (2,2), padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D((2,2), strides = (2,2))(x)
    x = tf.keras.layers.Conv2D(16, (3,3), activation='relu', strides = (2,2), padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2,2), strides = (2,2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(16, activation='relu')(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    return model
  ```

- CNN 경량화 모델 Summary
  
  ```
  ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
  ┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
  ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
  │ input_layer_1 (InputLayer)           │ (None, 224, 224, 3)         │               0 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ conv2d_3 (Conv2D)                    │ (None, 112, 112, 8)         │             224 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ max_pooling2d_3 (MaxPooling2D)       │ (None, 56, 56, 8)           │               0 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ conv2d_4 (Conv2D)                    │ (None, 28, 28, 16)          │           1,168 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ max_pooling2d_4 (MaxPooling2D)       │ (None, 14, 14, 16)          │               0 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ flatten_1 (Flatten)                  │ (None, 3136)                │               0 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ dense_2 (Dense)                      │ (None, 16)                  │          50,192 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ dense_3 (Dense)                      │ (None, 5)                   │              85 │
  └──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
  Total params: 51,669 (201.83 KB)
  Trainable params: 51,669 (201.83 KB)
  Non-trainable params: 0 (0.00 B)
  ```

- Model Evaluation 시각화
  
  ![alt text](/assets/images/cnnproject_cnnlight_accuracy.png)<br/>

  ![alt text](/assets/images/cnnproject_cnn_loss.png)<br/>

  ```
  Restoring model weights from the end of the best epoch: 19.
  ```
  Best score인 19번 째 epoch의 evaluation은 아래와 같다.<br/>
  Train Accuracy: 0.9986<br/>
  Train Loss: 0.0044<br/>
  Validation Accruacy: 0.9970<br/>
  Validation Loss:  0.0108<br/>

- Feature Map 시각화
  
  ![alt text](/assets/images/cnnproject_cnnlight_layer1.png)<br/>

  ![alt text](/assets/images/cnnproject_cnnlight_layer2.png)<br/>

  다양한 filter 중 유효한 filter만 사용된 모습을 볼 수 있다.

### 모델 비교

|Model|Parameter|Validation Accuracy|Validation Loss|
|---|--------|--|--|
|CNN|12,939,077|0.9963|0.0132|
|CNN 경량화|**51,669**|**0.9970**|**0.0108**|

Parameter 수는 각 12,939,077와 51,669로 99.60(%) 경량화하였으며 Accuracy와 Loss를 보았을 때, 성능차이는 거의 없으며 오히려 경량화 모델이 소폭 높은 것을 확인할 수 있다.

## (3) GoogLeNet
- GoogLeNet의 구조는 아래 이미지와 같다.<br/>
  ![alt text](/assets/images/cnnproject_googlenet_architect.png)<br/>
  GoogLeNet은 총 27개의 Layer로 구성되어 있고 Stem network는 신경망의 초기 부분으로 일반적인 CNN의 은닉 구조를 가진다. Inception Module은 GoogLeNet의 핵심적인 구조로 Layer를 하나의 Sub-Network구조로 구성하여 연산량을 줄이는 구조이다.<br/>
  사전 학습된 Weight 없이 직접 해당 구조를 쌓아보았다.<br/>

  ```python
  class InceptionModule(layers.Layer):
    def __init__(self, f1, f3_reduce, f3, f5_reduce, f5, pool_proj, **kwargs):
        super(InceptionModule, self).__init__(**kwargs)
        # 1x1 conv branch
        self.branch1 = layers.Conv2D(f1, (1,1), padding='same', activation='relu')
        
        # 1x1 -> 3x3 branch
        self.branch2 = models.Sequential([
            layers.Conv2D(f3_reduce, (1,1), padding='same', activation='relu'),
            layers.Conv2D(f3, (3,3), padding='same', activation='relu')
        ])
        
        # 1x1 -> 5x5 branch
        self.branch3 = models.Sequential([
            layers.Conv2D(f5_reduce, (1,1), padding='same', activation='relu'),
            layers.Conv2D(f5, (5,5), padding='same', activation='relu')
        ])
        
        # 3x3 max pooling -> 1x1 conv branch
        self.branch4 = models.Sequential([
            layers.MaxPooling2D((3,3), strides=(1,1), padding='same'),
            layers.Conv2D(pool_proj, (1,1), padding='same', activation='relu')
        ])
    
    def call(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        # 채널 축을 기준으로 병합
        return tf.concat([branch1, branch2, branch3, branch4], axis=-1)
  ```

  ```python
  def create_googlenet(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    
    # Stem network
    x = layers.Conv2D(64, (7,7), strides=(2,2), padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D((3,3), strides=(2,2), padding='same')(x)
    x = layers.Conv2D(64, (1,1), padding='same', activation='relu')(x)
    x = layers.Conv2D(192, (3,3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((3,3), strides=(2,2), padding='same')(x)
    
    # Inception modules
    x = InceptionModule(64, 96, 128, 16, 32, 32)(x)
    x = InceptionModule(128, 128, 192, 32, 96, 64)(x)
    x = layers.MaxPooling2D((3,3), strides=(2,2), padding='same')(x)
    
    x = InceptionModule(192, 96, 208, 16, 48, 64)(x)
    x = InceptionModule(160, 112, 224, 24, 64, 64)(x)
    x = InceptionModule(128, 128, 256, 24, 64, 64)(x)
    x = InceptionModule(112, 144, 288, 32, 64, 64)(x)
    x = InceptionModule(256, 160, 320, 32, 128, 128)(x)
    x = layers.MaxPooling2D((3,3), strides=(2,2), padding='same')(x)
    
    x = InceptionModule(256, 160, 320, 32, 128, 128)(x)
    x = InceptionModule(384, 192, 384, 48, 128, 128)(x)
    
    # 최종 분류기
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    return model

  # GoogLeNet 모델 생성 및 컴파일
  googlenet = create_googlenet(input_shape=img_size + (3,), num_classes=5)
  googlenet.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
  googlenet.summary()
  ```

- GoogLeNet Summary
  
  ```
  ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
  ┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
  ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
  │ input_layer_86 (InputLayer)          │ (None, 224, 224, 3)         │               0 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ conv2d_176 (Conv2D)                  │ (None, 112, 112, 64)        │           9,472 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ max_pooling2d_44 (MaxPooling2D)      │ (None, 56, 56, 64)          │               0 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ conv2d_177 (Conv2D)                  │ (None, 56, 56, 64)          │           4,160 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ conv2d_178 (Conv2D)                  │ (None, 56, 56, 192)         │         110,784 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ max_pooling2d_45 (MaxPooling2D)      │ (None, 28, 28, 192)         │               0 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ inception_module_27                  │ (None, 28, 28, 256)         │         163,696 │
  │ (InceptionModule)                    │                             │                 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ inception_module_28                  │ (None, 28, 28, 480)         │         388,736 │
  │ (InceptionModule)                    │                             │                 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ max_pooling2d_48 (MaxPooling2D)      │ (None, 14, 14, 480)         │               0 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ inception_module_29                  │ (None, 14, 14, 512)         │         376,176 │
  │ (InceptionModule)                    │                             │                 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ inception_module_30                  │ (None, 14, 14, 512)         │         449,160 │
  │ (InceptionModule)                    │                             │                 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ inception_module_31                  │ (None, 14, 14, 512)         │         510,104 │
  │ (InceptionModule)                    │                             │                 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ inception_module_32                  │ (None, 14, 14, 528)         │         605,376 │
  │ (InceptionModule)                    │                             │                 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ inception_module_33                  │ (None, 14, 14, 832)         │         868,352 │
  │ (InceptionModule)                    │                             │                 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ max_pooling2d_54 (MaxPooling2D)      │ (None, 7, 7, 832)           │               0 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ inception_module_34                  │ (None, 7, 7, 832)           │       1,043,456 │
  │ (InceptionModule)                    │                             │                 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ inception_module_35                  │ (None, 7, 7, 1024)          │       1,444,080 │
  │ (InceptionModule)                    │                             │                 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ global_average_pooling2d_3           │ (None, 1024)                │               0 │
  │ (GlobalAveragePooling2D)             │                             │                 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ dropout_3 (Dropout)                  │ (None, 1024)                │               0 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ dense_7 (Dense)                      │ (None, 5)                   │           5,125 │
  └──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
  Total params: 5,978,677 (22.81 MB)
  Trainable params: 5,978,677 (22.81 MB)
  Non-trainable params: 0 (0.00 B)
  ```

- Model Evaluation 시각화
  
  ![alt text](/assets/images/cnnproject_googlenet_accuracy.png)<br/>

  ![alt text](/assets/images/cnnproject_googlenet_loss.png)<br/>

  ```
  Restoring model weights from the end of the best epoch: 24.
  ```
  Best score인 24번 째 epoch의 evaluation은 아래와 같다.<br/>
  Train Accuracy: 0.9988<br/>
  Train Loss: 0.0039<br/>
  Validation Accruacy: 0.9975<br/>
  Validation Loss: 0.0111<br/>

- feature map 시각화<br/>
  feature map은 앞선 stemp network만 시각화해보았다.
  > 앞서서 GoogLeNet 구조에 대해 설명해보는게 좋을 듯 싶음

  ![alt text](/assets/images/cnnproject_googlenet_layer1.png)<br/>
  ![alt text](/assets/images/cnnproject_googlenet_layer2.png)<br/>
  ![alt text](/assets/images/cnnproject_googlenet_layer3.png)<br/>

## (4) GoogLeNet 경량화
- GoogLeNet은 Stem Network의 filter의 개수를 줄이고 각 Inception Module의 filter 개수를 줄여주었다. 또한 2단계의 Inception Module하나를 삭제하므로써 연산량을 줄였다. Inception Module의 내부 구조는 GoogLeNet의 특징이므로 건드리지 않았다.
  
  ```python
  # Inception Module은 GoogLeNet과 같으므로 생략

  def create_googlenet_light(input_shape, num_classes):
      inputs = layers.Input(shape=input_shape)

      # Stem network
      x = layers.Conv2D(16, (7,7), strides=(2,2), padding='same', activation='relu')(inputs)
      x = layers.MaxPooling2D((3,3), strides=(2,2), padding='same')(x)
      x = layers.Conv2D(16, (1,1), padding='same', activation='relu')(x)
      x = layers.Conv2D(64, (3,3), padding='same', activation='relu')(x)
      x = layers.MaxPooling2D((3,3), strides=(2,2), padding='same')(x)

      # Inception modules
      x = InceptionModule(16, 16, 24, 4, 8, 8)(x)
      x = InceptionModule(24, 24, 32, 4, 8, 8)(x)
      x = layers.MaxPooling2D((3,3), strides=(2,2), padding='same')(x)

      x = InceptionModule(32, 32, 48, 8, 16, 16)(x)
      x = InceptionModule(48, 48, 64, 8, 16, 16)(x)
      x = InceptionModule(64, 64, 96, 12, 24, 24)(x)
      x = InceptionModule(96, 64, 96, 12, 24, 24)(x)
      x = layers.MaxPooling2D((3,3), strides=(2,2), padding='same')(x)

      x = InceptionModule(96, 64, 96, 12, 24, 24)(x)
      x = InceptionModule(96, 96, 128, 16, 32, 32)(x)

      # 최종 분류기
      x = layers.GlobalAveragePooling2D()(x)
      x = layers.Dropout(0.4)(x)
      outputs = layers.Dense(num_classes, activation='softmax')(x)

      model = models.Model(inputs, outputs)
      return model

  # GoogLeNet 모델 생성 및 컴파일
  googlenet_light = create_googlenet_light(input_shape=img_size + (3,), num_classes=5)
  googlenet_light.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
  googlenet_light.summary()
  ```

- GoogLeNet 경량화 Summary
  
  ```
  ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
  ┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
  ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
  │ input_layer (InputLayer)             │ (None, 224, 224, 3)         │               0 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ conv2d (Conv2D)                      │ (None, 112, 112, 16)        │           2,368 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ max_pooling2d (MaxPooling2D)         │ (None, 56, 56, 16)          │               0 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ conv2d_1 (Conv2D)                    │ (None, 56, 56, 16)          │             272 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ conv2d_2 (Conv2D)                    │ (None, 56, 56, 64)          │           9,280 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ max_pooling2d_1 (MaxPooling2D)       │ (None, 28, 28, 64)          │               0 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ inception_module (InceptionModule)   │ (None, 28, 28, 56)          │           7,148 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ inception_module_1 (InceptionModule) │ (None, 28, 28, 72)          │          11,172 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ max_pooling2d_4 (MaxPooling2D)       │ (None, 14, 14, 72)          │               0 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ inception_module_2 (InceptionModule) │ (None, 14, 14, 112)         │          23,512 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ inception_module_3 (InceptionModule) │ (None, 14, 14, 144)         │          44,488 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ inception_module_4 (InceptionModule) │ (None, 14, 14, 208)         │          86,396 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ inception_module_5 (InceptionModule) │ (None, 14, 14, 240)         │         103,580 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ max_pooling2d_9 (MaxPooling2D)       │ (None, 7, 7, 240)           │               0 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ inception_module_6 (InceptionModule) │ (None, 7, 7, 240)           │         109,852 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ inception_module_7 (InceptionModule) │ (None, 7, 7, 288)           │         181,392 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ global_average_pooling2d             │ (None, 288)                 │               0 │
  │ (GlobalAveragePooling2D)             │                             │                 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ dropout (Dropout)                    │ (None, 288)                 │               0 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ dense (Dense)                        │ (None, 5)                   │           1,445 │
  └──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
  Total params: 580,905 (2.22 MB)
  Trainable params: 580,905 (2.22 MB)
  Non-trainable params: 0 (0.00 B)
  ```

- Model Evaluation 시각화
  
  ![alt text](/assets/images/cnnproject_googlenetlight_accuracy.png)<br/>
  ![alt text](/assets/images/cnnproject_googlenetlight_loss.png)<br/>
  ```
  Restoring model weights from the end of the best epoch: 16.
  ```
  Best score인 16번 째 epoch의 evaluation은 아래와 같다.<br/>
  Train Accuracy: 0.9983<br/>
  Train Loss: 0.0071<br/>
  Validation Accruacy: 0.9981<br/>
  Validation Loss: 0.0070<br/>
  
- feature map 시각화<br/>
  feature map은 앞선 stemp network만 시각화해보았다.

  ![alt text](/assets/images/cnnproject_googlenetlight_layer1.png)<br/>
  ![alt text](/assets/images/cnnproject_googlenetlight_layer2.png)<br/>
  ![alt text](/assets/images/cnnproject_googlenetlight_layer3.png)<br/>

### 모델 비교

|Model|Parameter|Validation Accuracy|Validation Loss|
|---|--------|--|--|
|GoogLeNet|5,978,677|0.9975|0.0111|
|GoogLeNet 경량화|**580,905**|**0.9981**|**0.0070**|

Parameter 수는 각 5,978,677와 580,905로 90.28(%) 경량화하였으며 Accuracy와 Loss를 보았을 때, 성능차이는 거의 없으며 오히려 경량화 모델이 소폭 높은 것을 확인할 수 있다.

## (5) VGG16
- VGG16은 구조가 매우 간단하다. 구조는 아래와 같다.<br/>
  ![alt text](/assets/images/cnnproject_vgg16_architect.png)<br/>
  VGG16은 모든 Convolution Layer에 3x3 필터를 적용하는 것이 큰 특징이다. 구조가 간단하고 이해가 쉽고 변형을 시켜가면서 테스트 하기 용이해 자주 사용되는 모델이다. 해당 모델은 Feature Map 크기는 동일하지만 학습해야할 파라미터 수를 줄였다는 특징이 있다.<br/>
  해당 모델은 [Rice Image Project 예시](https://www.kaggle.com/code/sharduljoshi29/rice-classification-using-vgg16-99-accuracy)를 참고하여 Imagenet으로 사전학습 된 모델에서 미세조정 하였다. 또한 이미지 증강은 사용하지 않았고 'Rice Image Dataset'을 그대로 사용하였다.<br/>

  
  ```python
  input_tensor = Input(shape=img_size + (3,))  # (224, 224, 3)

  # VGG16 base model
  base_model = VGG16(include_top=False,
                    weights='imagenet',
                    input_tensor=input_tensor)

  # 필요한 레이어만 학습되도록 설정
  for layer in base_model.layers[:-4]:
      layer.trainable = False

  # 커스텀 분류기 추가
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Dense(512, activation='relu')(x)
  x = Dropout(0.3)(x)
  output_tensor = Dense(5, activation='softmax')(x)

  # 전체 모델 정의
  vgg16 = Model(inputs=input_tensor, outputs=output_tensor)
  vgg16.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

  # 모델 구조 출력
  vgg16.summary()
  ```

- VGG16 summary
  
  ```
  ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
  ┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
  ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
  │ input_layer_171 (InputLayer)         │ (None, 224, 224, 3)         │               0 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ block1_conv1 (Conv2D)                │ (None, 224, 224, 64)        │           1,792 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ block1_conv2 (Conv2D)                │ (None, 224, 224, 64)        │          36,928 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ block1_pool (MaxPooling2D)           │ (None, 112, 112, 64)        │               0 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ block2_conv1 (Conv2D)                │ (None, 112, 112, 128)       │          73,856 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ block2_conv2 (Conv2D)                │ (None, 112, 112, 128)       │         147,584 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ block2_pool (MaxPooling2D)           │ (None, 56, 56, 128)         │               0 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ block3_conv1 (Conv2D)                │ (None, 56, 56, 256)         │         295,168 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ block3_conv2 (Conv2D)                │ (None, 56, 56, 256)         │         590,080 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ block3_conv3 (Conv2D)                │ (None, 56, 56, 256)         │         590,080 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ block3_pool (MaxPooling2D)           │ (None, 28, 28, 256)         │               0 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ block4_conv1 (Conv2D)                │ (None, 28, 28, 512)         │       1,180,160 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ block4_conv2 (Conv2D)                │ (None, 28, 28, 512)         │       2,359,808 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ block4_conv3 (Conv2D)                │ (None, 28, 28, 512)         │       2,359,808 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ block4_pool (MaxPooling2D)           │ (None, 14, 14, 512)         │               0 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ block5_conv1 (Conv2D)                │ (None, 14, 14, 512)         │       2,359,808 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ block5_conv2 (Conv2D)                │ (None, 14, 14, 512)         │       2,359,808 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ block5_conv3 (Conv2D)                │ (None, 14, 14, 512)         │       2,359,808 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ block5_pool (MaxPooling2D)           │ (None, 7, 7, 512)           │               0 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ global_average_pooling2d_7           │ (None, 512)                 │               0 │
  │ (GlobalAveragePooling2D)             │                             │                 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ dense_14 (Dense)                     │ (None, 512)                 │         262,656 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ dropout_7 (Dropout)                  │ (None, 512)                 │               0 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ dense_15 (Dense)                     │ (None, 5)                   │           2,565 │
  └──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
  Total params: 14,979,909 (57.14 MB)
  Trainable params: 7,344,645 (28.02 MB)
  Non-trainable params: 7,635,264 (29.13 MB)
  ```

- Model Evaluation 시각화
  
  ![alt text](/assets/images/cnnproject_vgg16_accuracy.png)<br/>
  ![alt text](/assets/images/cnnproject_vgg16_loss.png)<br/>

  ```
  Restoring model weights from the end of the best epoch: 11.
  ```
  Best score인 11번 째 epoch의 evaluation은 아래와 같다.<br/>
  Train Accuracy: 0.9994<br/>
  Train Loss: 0.0024<br/>
  Validation Accruacy: 0.9979<br/>
  Validation Loss: 0.0099<br/>
  
- feature map 시각화<br/>

  ![alt text](/assets/images/cnnproject_vgg16_layer1.png)<br/>
  ![alt text](/assets/images/cnnproject_vgg16_layer2.png)<br/>
  ![alt text](/assets/images/cnnproject_vgg16_layer3.png)<br/>
  ![alt text](/assets/images/cnnproject_vgg16_layer4.png)<br/>
  ![alt text](/assets/images/cnnproject_vgg16_layer5.png)<br/>
  ![alt text](/assets/images/cnnproject_vgg16_layerfin.png)<br/>


## (6) VGG16 경량화
- 사전학습 되어있는 Filter를 그대로 사용하고 block_5는 특징 추출이 되지 않은 형태로 보이므로 삭제하여 경량화였다.

  ```python
  x = vgg16.get_layer('block4_pool').output

  # 분류기 추가
  x = GlobalAveragePooling2D()(x)
  x = Dense(512, activation='relu')(x)
  x = Dropout(0.3)(x)
  output_tensor = Dense(5, activation='softmax')(x)

  # 전체 모델 정의
  vgg_light = Model(inputs=input_tensor, outputs=output_tensor)

  # 필요한 레이어만 학습되도록 설정 (보통 block1,2는 고정)
  for layer in vgg16.layers:
      layer.trainable = False  # 전부 freeze 하거나 선택적으로 조절 가능

  # 컴파일
  vgg_light.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

  # 모델 구조 출력
  vgg_light.summary()
  ```

- VGG 경량화 Summary
  
  ```
  ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
  ┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
  ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
  │ input_layer (InputLayer)             │ (None, 224, 224, 3)         │               0 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ block1_conv1 (Conv2D)                │ (None, 224, 224, 64)        │           1,792 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ block1_conv2 (Conv2D)                │ (None, 224, 224, 64)        │          36,928 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ block1_pool (MaxPooling2D)           │ (None, 112, 112, 64)        │               0 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ block2_conv1 (Conv2D)                │ (None, 112, 112, 128)       │          73,856 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ block2_conv2 (Conv2D)                │ (None, 112, 112, 128)       │         147,584 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ block2_pool (MaxPooling2D)           │ (None, 56, 56, 128)         │               0 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ block3_conv1 (Conv2D)                │ (None, 56, 56, 256)         │         295,168 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ block3_conv2 (Conv2D)                │ (None, 56, 56, 256)         │         590,080 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ block3_conv3 (Conv2D)                │ (None, 56, 56, 256)         │         590,080 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ block3_pool (MaxPooling2D)           │ (None, 28, 28, 256)         │               0 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ block4_conv1 (Conv2D)                │ (None, 28, 28, 512)         │       1,180,160 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ block4_conv2 (Conv2D)                │ (None, 28, 28, 512)         │       2,359,808 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ block4_conv3 (Conv2D)                │ (None, 28, 28, 512)         │       2,359,808 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ block4_pool (MaxPooling2D)           │ (None, 14, 14, 512)         │               0 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ global_average_pooling2d_3           │ (None, 512)                 │               0 │
  │ (GlobalAveragePooling2D)             │                             │                 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ dense_6 (Dense)                      │ (None, 512)                 │         262,656 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ dropout_3 (Dropout)                  │ (None, 512)                 │               0 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ dense_7 (Dense)                      │ (None, 5)                   │           2,565 │
  └──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
  Total params: 7,900,485 (30.14 MB)
  Trainable params: 265,221 (1.01 MB)
  Non-trainable params: 7,635,264 (29.13 MB)
  ```

- Model Evaluation 시각화
  
  ![alt text](/assets/images/cnnproject_vgglight_accuracy.png)<br/>
  ![alt text](/assets/images/cnnproject_vgglight_loss.png)<br/>
  ```
  Restoring model weights from the end of the best epoch: 28.
  ```
  Best score인 28번 째 epoch의 evaluation은 아래와 같다.<br/>
  Train Accuracy: 0.9982<br/>
  Train Loss: 0.0061<br/>
  Validation Accruacy: 0.9977<br/>
  Validation Loss: 0.0077<br/>
  
- feature map 시각화<br/>

  ![alt text](/assets/images/cnnproject_vgg16_layer1.png)<br/>
  ![alt text](/assets/images/cnnproject_vgg16_layer2.png)<br/>
  ![alt text](/assets/images/cnnproject_vgg16_layer3.png)<br/>
  ![alt text](/assets/images/cnnproject_vgg16_layer4.png)<br/>

  사전 학습된 모델 그대로 가져왔으므로 Filter를 통과한 Feature map은 동일하다.

## (7) VGG Custom
- VGG16 모델과 경량화 모델은 모두 Imagenet으로 사전학습된 모델이다. 해당 모델은 VGG16의 구조를 따라가되 직접 학습한 모델이며 총 3개의 block으로 이루어져있다.
  
  ```python
  def vgg_custom(input_shape=(224, 224, 3), num_classes=5):
      inputs = layers.Input(shape=input_shape)

      # Block 1 - 16 filters
      x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
      x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
      x = layers.MaxPooling2D((2, 2))(x)

      # Block 2 - 32 filters
      x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
      x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
      x = layers.MaxPooling2D((2, 2))(x)

      # Block 3 - 64 filters
      x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
      x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
      x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
      x = layers.MaxPooling2D((2, 2))(x)

      x = layers.GlobalAveragePooling2D()(x)
      x = layers.Dense(512, activation='relu')(x)
      x = layers.Dropout(0.5)(x)
      outputs = layers.Dense(num_classes, activation='softmax')(x)

      model = models.Model(inputs, outputs)
      return model
    
  vgg_custom = vgg_custom(input_shape=img_size + (3,), num_classes=5)
  vgg_custom.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
  vgg_custom.summary()
  ```

- VGG Custom Summary
  
  ```
  ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
  ┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
  ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
  │ input_layer_1 (InputLayer)           │ (None, 224, 224, 3)         │               0 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ conv2d (Conv2D)                      │ (None, 224, 224, 16)        │             448 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ conv2d_1 (Conv2D)                    │ (None, 224, 224, 16)        │           2,320 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ max_pooling2d (MaxPooling2D)         │ (None, 112, 112, 16)        │               0 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ conv2d_2 (Conv2D)                    │ (None, 112, 112, 32)        │           4,640 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ conv2d_3 (Conv2D)                    │ (None, 112, 112, 32)        │           9,248 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ max_pooling2d_1 (MaxPooling2D)       │ (None, 56, 56, 32)          │               0 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ conv2d_4 (Conv2D)                    │ (None, 56, 56, 64)          │          18,496 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ conv2d_5 (Conv2D)                    │ (None, 56, 56, 64)          │          36,928 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ conv2d_6 (Conv2D)                    │ (None, 56, 56, 64)          │          36,928 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ max_pooling2d_2 (MaxPooling2D)       │ (None, 28, 28, 64)          │               0 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ global_average_pooling2d_4           │ (None, 64)                  │               0 │
  │ (GlobalAveragePooling2D)             │                             │                 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ dense_8 (Dense)                      │ (None, 512)                 │          33,280 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ dropout_4 (Dropout)                  │ (None, 512)                 │               0 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ dense_9 (Dense)                      │ (None, 5)                   │           2,565 │
  └──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
  Total params: 144,853 (565.83 KB)
  Trainable params: 144,853 (565.83 KB)
  Non-trainable params: 0 (0.00 B)
  ```

- Model Evaluation 시각화
  
  ![alt text](/assets/images/cnnproject_vggcustom_accuracy.png)<br/>
  ![alt text](/assets/images/cnnproject_vggcustom_loss.png)<br/>
  ```
  Restoring model weights from the end of the best epoch: 41.
  ```
  Best score인 41번 째 epoch의 evaluation은 아래와 같다.<br/>
  Train Accuracy: 0.9986<br/>
  Train Loss: 0.0051<br/>
  Validation Accruacy: 0.9988<br/>
  Validation Loss: 0.0042<br/>
  
- feature map 시각화<br/>

  ![alt text](/assets/images/cnnproject_vggcustom_layer1.png)<br/>
  ![alt text](/assets/images/cnnproject_vggcustom_layer2.png)<br/>
  ![alt text](/assets/images/cnnproject_vggcustom_layer3.png)<br/>

  각 블럭의 1층의 Feature Map만 시각화하였다.

### 모델 비교

|Model|Parameter|Validation Accuracy|Validation Loss|
|---|--------|--|--|
|VGG16|14,979,909|0.9979|0.0099|
|VGG 경량화|7,900,485|0.9977|0.0077|
|VGG Custom|**144,853**|**0.9988**|**0.0042**|

Parameter 수는 각 VGG16 모델 대비 99.03(%) 경량화하였으며 VGG 경량화 모델 대비 98.16(%)경량화 되었다. Accuracy와 Loss를 보았을 때, 성능차이는 VGG16 Custom 모델이 제일 높은 성능을 보여준다.

# 4. 모델별 일반화 성능비교

&nbsp;&nbsp;각 학습한 모델들을 `.h5`형태로 저장하였으며 일반화 성능과 추론 속도를 보기 위해 기존의 데이터를 가공하여 사용해보았다.

```python
raw_test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_fpath,
    image_size=img_size,
    batch_size=batch_size,
    shuffle=False,
    label_mode='categorical'
)

class_names = raw_test_ds.class_names

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),              # 수평 뒤집기
    layers.RandomRotation(0.1),                   # ±10% 회전
    layers.RandomZoom(0.1),                       # ±10% 확대/축소
    layers.RandomTranslation(0.1, 0.1),           # ±10% 가로/세로 이동
    layers.RandomContrast(0.1)                    # 명암 변화
])

# 정규화 레이어 적용
normalization_layer = tf.keras.layers.Rescaling(1./255)
test_ds = raw_test_ds.map(lambda x, y: (normalization_layer(data_augmentation(x)), y))

# 성능 최적화를 위한 prefetch
AUTOTUNE = tf.data.AUTOTUNE
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)
```

&nbsp;&nbsp;추론 속도는 정확한 값으로 측정하기 위해서 FLOPs로 대체하였다. FLOPs란 FLoating point Operations의 약자로 부동소수점 연산을 의미하며 주로 모델의 계산 복잡성을 측정하는데 사용된다. Device Performance는 대개 FLOPS(FLoating point Operations Per Second)로 측정하고 있으며 추론 시간은 Device의 성능이 높을수록 계산해야하는 FLOPs가 낮을수록 추론시간은 짧아진다. CNN 계열 모델의 이론적 추론시간은 아래와 같다.<br/>
$$
Inference\,Time = \frac{\sum_{l=1}^{L}FLOPs_l}{Device\,Performance\,(FLOPs/sec)}
$$
<br/>
&nbsp;&nbsp;일반적으로 Convolution Lyaer의 FLOPs의 계산은 아래와 같이 계산된다.<br/>

$$
FLOPs_{conv} = 2 \times C_{in} \times K^2 \times H_{out} \times W_{out} \times C_{out}
$$
<br/>
$C_{in}$: 입력 채널 수<br/>
$K$: 커널크기<br/>
$H_{out}, W_{out}$: Feature map의 height, width<br/>
$C_{out}$: 출력 채널 수<br/>

&nbsp;&nbsp;여기서 Layer를 제거하거나 Filter의 개수를 줄여 다음 Layer에서의 $C_{in}$이 줄어든다. 따라서 계산량이 줄어들어 상당한 FLOPs의 이득을 볼 수 있다. FLOPs 계산은 아래와 같이 하였다.
```python
def get_flops(model, batch_size=1):
    try:
        concrete = tf.function(lambda x: model(x))
        concrete_func = concrete.get_concrete_function(
            tf.TensorSpec([batch_size, 224, 224, 3], tf.float32))

        frozen_func = convert_to_constants.convert_variables_to_constants_v2(concrete_func)
        graph_def = frozen_func.graph.as_graph_def()

        with tf.Graph().as_default() as graph:
            tf.graph_util.import_graph_def(graph_def, name='')
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd='op', options=opts)
            return flops.total_float_ops
    except Exception as e:
        print(f"FLOPs 계산 실패: {e}")
        return None
```

## (1) CNN
- 총 필터 수: 224개

- 이미지 1장 당 FLOPs : 1,042,859,422 (약 1.04 GFLOPs)<br/>
  이미지 7,500장 : 약 7.82 TFLOPs

- Test Accuracy : 0.7092

- Confusion Matrix<br/>
  ![alt text](/assets/images/cnnproject_cnn_confusionmatrix.png)<br/>

  | Class        | Precision | Recall | F1-Score | Support |
  |--------------|-----------|--------|----------|---------|
  | **Arborio**   | 0.91      | 0.55   | 0.68     | 1500    |
  | **Basmati**   | 0.68      | 0.90   | 0.77     | 1500    |
  | **Ipsala**    | 0.88      | 0.96   | 0.92     | 1500    |
  | **Jasmine**   | 0.48      | 0.60   | 0.53     | 1500    |
  | **Karacadag** | 0.76      | 0.53   | 0.62     | 1500    |
  |              |           |        |          |         |
  | **Accuracy**  |           |        | **0.71** | 7500    |
  | **Macro Avg** | 0.74      | 0.71   | 0.71     | 7500    |
  | **Weighted Avg** | 0.74   | 0.71   | 0.71     | 7500    |
  
## (2) CNN 경량화
- 총 필터 수: 24개
  
- 이미지 1장 당 FLOPs : 7,551,694 (약 7.55 MFLOPs)<br/>
  이미지 7,500장 : 약 56.64 GFLOPs

- Test Accuracy : 0.7456

- Confusion Matrix<br/>
  ![alt text](/assets/images/cnnproject_cnnlight_confusionmatrix.png)<br/>

  | Class        | Precision | Recall | F1-Score | Support |
  |--------------|-----------|--------|----------|---------|
  | **Arborio**   | 0.85      | 0.60   | 0.70     | 1500    |
  | **Basmati**   | 0.67      | 0.96   | 0.79     | 1500    |
  | **Ipsala**    | 0.97      | 0.94   | 0.95     | 1500    |
  | **Jasmine**   | 0.54      | 0.77   | 0.63     | 1500    |
  | **Karacadag** | 1.00      | 0.47   | 0.64     | 1500    |
  |              |           |        |          |         |
  | **Accuracy**  |           |        | **0.75** | 7500    |
  | **Macro Avg** | 0.80      | 0.75   | 0.74     | 7500    |
  | **Weighted Avg** | 0.80   | 0.75   | 0.74     | 7500    |

## (3) GoogLeNet
- 총 필터 수: 5,808개

- 이미지 1장 당 FLOPs : 3,179,339,598 (약 3.18 GFLOPs)<br/>
  이미지 7,500장 : 약 23.85 TFLOPs

- Test Accuracy : 0.8675

- Confusion Matrix<br/>
  ![alt text](/assets/images/cnnproject_googlenet_confusionmatrix.png)<br/>

    | Class        | Precision | Recall | F1-Score | Support |
    |--------------|-----------|--------|----------|---------|
    | **Arborio**   | 0.93      | 0.69   | 0.79     | 1500    |
    | **Basmati**   | 0.92      | 0.98   | 0.95     | 1500    |
    | **Ipsala**    | 0.88      | 0.99   | 0.93     | 1500    |
    | **Jasmine**   | 0.71      | 0.91   | 0.80     | 1500    |
    | **Karacadag** | 0.99      | 0.76   | 0.86     | 1500    |
    |              |           |        |          |         |
    | **Accuracy**  |           |        | **0.87** | 7500    |
    | **Macro Avg** | 0.89      | 0.87   | 0.87     | 7500    |
    | **Weighted Avg** | 0.89   | 0.87   | 0.87     | 7500    |

## (4) GoogLeNet 경량화
- 총 필터 수: 1,456개

- 이미지 1장 당 FLOPs : 280,058,490 (약 0.28 GFLOPs)<br/>
  이미지 7,500장 : 약 2.10 TFLOPs

- Test Accuracy : 0.9319

- Confusion Matrix<br/>
  ![alt text](/assets/images/cnnproject_googlenetlight_confusionmatrix.png)<br/>

  | Class        | Precision | Recall | F1-Score | Support |
  |--------------|-----------|--------|----------|---------|
  | **Arborio**   | 0.91      | 0.87   | 0.89     | 1500    |
  | **Basmati**   | 0.97      | 0.99   | 0.98     | 1500    |
  | **Ipsala**    | 0.98      | 0.99   | 0.98     | 1500    |
  | **Jasmine**   | 0.83      | 0.98   | 0.90     | 1500    |
  | **Karacadag** | 1.00      | 0.83   | 0.91     | 1500    |
  |              |           |        |          |         |
  | **Accuracy**  |           |        | **0.93** | 7500    |
  | **Macro Avg** | 0.94      | 0.93   | 0.93     | 7500    |
  | **Weighted Avg** | 0.94   | 0.93   | 0.93     | 7500    |

## (5) VGG16
- 총 필터 수: 4,320개

- 이미지 1장 당 FLOPs : 30,713,485,342 (약 30.71 GFLOPs)<br/>
  이미지 7,500장 : 약 226.30 TFLOPs

- Test Accuracy : 0.9649

- Confusion Matrix<br/>
  ![alt text](/assets/images/cnnproject_vgg16_confusionmatrix.png)<br/>

  | Class        | Precision | Recall | F1-Score | Support |
  |--------------|-----------|--------|----------|---------|
  | **Arborio**   | 1.00      | 0.97   | 0.98     | 1500    |
  | **Basmati**   | 0.88      | 1.00   | 0.94     | 1500    |
  | **Ipsala**    | 0.97      | 1.00   | 0.99     | 1500    |
  | **Jasmine**   | 0.99      | 0.86   | 0.92     | 1500    |
  | **Karacadag** | 0.99      | 1.00   | 1.00     | 1500    |
  |              |           |        |          |         |
  | **Accuracy**  |           |        | **0.96** | 7500    |
  | **Macro Avg** | 0.97      | 0.96   | 0.96     | 7500    |
  | **Weighted Avg** | 0.97   | 0.96   | 0.96     | 7500    |

## (6) VGG 경량화
- 총 필터 수: 2,688개

- 이미지 1장 당 FLOPs : 27,938,627,102 (약 27.94 GFLOPs)<br/>
  이미지 7,500장 : 약 290.54 TFLOPs

- Test Accuracy : 0.9227

- Confusion Matrix<br/>
  ![alt text](/assets/images/cnnproject_vgglight_confusionmatrix.png)<br/>

  | Class        | Precision | Recall | F1-Score | Support |
  |--------------|-----------|--------|----------|---------|
  | **Arborio**   | 1.00      | 0.89   | 0.94     | 1500    |
  | **Basmati**   | 0.78      | 1.00   | 0.88     | 1500    |
  | **Ipsala**    | 0.99      | 1.00   | 0.99     | 1500    |
  | **Jasmine**   | 0.99      | 0.72   | 0.83     | 1500    |
  | **Karacadag** | 0.93      | 1.00   | 0.96     | 1500    |
  |              |           |        |          |         |
  | **Accuracy**  |           |        | **0.92** | 7500    |
  | **Macro Avg** | 0.94      | 0.92   | 0.92     | 7500    |
  | **Weighted Avg** | 0.94   | 0.92   | 0.92     | 7500    |

## (7) VGG Custom
- 총 필터 수: 288개

- 이미지 1장 당 FLOPs : 1,203,943,966 (약 1.20 GFLOPs)<br/>
  이미지 7,500장 : 약 9.02 TFLOPs

- Test Accuracy : 0.9909

- Confusion Matrix<br/>
  ![alt text](/assets/images/cnnproject_vggcustom_confusionmatrix.png)<br/>

  | Class        | Precision | Recall | F1-Score | Support |
  |--------------|-----------|--------|----------|---------|
  | **Arborio**   | 0.97      | 0.99   | 0.98     | 1500    |
  | **Basmati**   | 1.00      | 1.00   | 1.00     | 1500    |
  | **Ipsala**    | 1.00      | 1.00   | 1.00     | 1500    |
  | **Jasmine**   | 1.00      | 0.97   | 0.98     | 1500    |
  | **Karacadag** | 0.99      | 1.00   | 0.99     | 1500    |
  |              |           |        |          |         |
  | **Accuracy**  |           |        | **0.99** | 7500    |
  | **Macro Avg** | 0.99      | 0.99   | 0.99     | 7500    |
  | **Weighted Avg** | 0.99   | 0.99   | 0.99     | 7500    |

## (8) 성능비교

| 모델 이름         | Accuracy | Macro Precision | Macro Recall | Macro F1-score |
|-------------------|----------|------------------|---------------|----------------|
| CNN               | 0.71     | 0.74             | 0.71          | 0.71           |
| CNN Light         | 0.75     | 0.80             | 0.75          | 0.74           |
| GoogLeNet         | 0.87     | 0.89             | 0.87          | 0.87           |
| GoogLeNet Light   | 0.93     | 0.94             | 0.93          | 0.93           |
| VGG16             | 0.96     | 0.97             | 0.96          | 0.96           |
| VGG Light         | 0.92     | 0.94             | 0.92          | 0.92           |
| VGG Custom        | 0.99     | 0.99             | 0.99          | 0.99           |

![alt text](/assets/images/cnnproject_summary_accuracy.png)<br/>
Filter의 개수를 줄이거나 Layer를 제거한 모델들의 일반화 성능이 더욱 높게 나왔다.<br/>

![alt text](/assets/images/cnnproject_summary_flops.png)<br/>
오른쪽으로 갈수록 필요 계산량이 적고 위로 갈수록 성능이 좋다. Accuracy가 0.9 미만은 실사용 불가능이라고 판단하였다.<br/>
![alt text](/assets/images/cnnproject_summary_filter.png)<br/>
오른쪽으로 갈수록 Filter의 개수가 적고 위로 갈수록 성능이 좋다. Accuracy가 0.9 미만은 실사용 불가능이라고 판단하였다.<br/>
&nbsp;&nbsp;FLOPs vs Test Accuracy 표를 보면 왼쪽 아래로 갈수록 성능이 좋지 않고 계산에 필요한 자원이 많이 들들고 오른쪽 위로 갈수록 성능이 좋으면서 계산에 필요한 자원이 적다는 것이다. 'Rice Image Dataset'기준으로 VGG16 Custom이 가장 효율이 좋은 것으로 판단된다. GoogLeNet Light도 높은 성능과 낮은 계산량으로 효율면에서 좋다고 판단된다.<br/>
&nbsp;&nbsp;CNN Light의 경우 계산량이 0.056TFLOPs 수준으로 압도적으로 낮았으나 Test Accuracy가 0.9 미만이면 일반화 성능 부족으로 실사용 불가로 판단하였기에 정확도가 뒷받침 해주지 못하여 실사용 불가판정을 내렸다. Accuracy / TFLOPs 그래프를 시각화해보면 아래와 같다.<br/>
![alt text](/assets/images/cnnproject_summary_accpertflops.png)<br/>
&nbsp;&nbsp;CNN Light의 경우 Validation Accuracy가 0.9970, Validation Loss가 0.0108로 학습은 굉장이 잘 되었다. 따라서 'Rice Image Dataset'처럼 이미지 전처리가 잘 되어있고 어느정도 과적합이 괜찮은 상황에서 극단적으로 계산량을 줄이고 싶다면 사용해볼 수 있다고 생각한다. 따라서 일반적인 CNN모델도 어느정도 개선을 계산량은 조금만 올리고 Test Accuracy는 많이 올릴 수 있다고 생각한다.
&nbsp;&nbsp;Rice Image Dataset'기준으로 Filter의 개수가 꼭 성능에 큰 영향을 미치지 않는 것을 볼 수 있다. 성능비교 표를 보면 알 수 있듯이 오히려 Filter의 개수가 적을수록 일반화 성능이 더 앞서는 것을 보여준다. [서론](#4-back-propagation-관점)에서 이야기한 것과 같이 GoogLeNet처럼 Filter의 개수가 너무 많으면 'Rice Image Dataset'과 같이 복잡하지 않은 이미지에서는 학습이 제대로 이루어지지 않을 수 있는 것을 주의해야한다.<br/>

# 5. 결론
&nbsp;&nbsp;본 프로젝트에서는 모델 구조를 단순히 깊게 설계하거나 필터 수를 늘리는 방식보다는, 실제 학습에 도움이 되지 않는 불필요한 구조를 제거하는 방식의 경량화 전략이 오히려 더 좋은 성능을 낼 수 있음을 확인하였다.<br/>
&nbsp;&nbsp;Feature Map을 시각적으로 확인한 결과, 일부 모델에서는 이미지가 비교적 단순하고 전처리가 잘 되어 있음에도 불구하고 의미 있는 특성 추출에 기여하지 않는 필터들이 다수 존재하였다. 따라서 필터 수를 줄이거나 레이어를 삭제하는 방식으로 모델을 경량화한 결과, 실제 테스트 성능에서 오히려 일반화가 더 잘 되는 현상을 확인할 수 있었다.<br/>
&nbsp;&nbsp;실제로 '[4. 모델별 일반화 성능 비교](#4-모델별-일반화-성능비교)에서 확인할 수 있듯이 경량화된 모델들이 오히려 더 우수한 성능을 보이는 경우가 다수였으며, 이는 필터 수가 많다고 반드시 좋은 모델이 되는 것은 아님을 보여주는 결과였다.
&nbsp;&nbsp;이를 바탕으로, CNN Light의 압도적으로 낮은 계산량과 VGG Custom의 뛰어난 일반화 성능이라는 각각의 강점을 결합한 최종 개선 모델을 설계하였다. 이 모델은 CNN Light처럼 필터 개수를 8 → 16으로 제한하면서도, VGG Custom과 같이 두 번째 풀링 이전에 Convolution 층을 추가하여 표현력을 확보하였고, Global Average Pooling과 Dropout을 적용하여 경량화와 성능 모두를 고려하였다.

## (1) 최종 모델
- CNN Light와 같이 필터의 개수를 8 → 16으로 제한하여 CNN Light의 극단적으로 적은 계산량을 가져가고 VGG Custom과 같이 두 번째 풀링 이전에 Convolution 층을 추가하여 표현력을 확보하였다. 또한 Global Average Pooling과 Dropout을 적용하여 경량화와 성능 모두를 고려한 모델이다.
  
  ```python
  def final_model(input_shape, num_classes):
      inputs = tf.keras.Input(shape=input_shape)
      x = tf.keras.layers.Conv2D(8, (3,3), activation='relu', padding='same')(inputs)
      x = tf.keras.layers.MaxPooling2D((2,2), strides = (2,2))(x)
      x = tf.keras.layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)
      x = tf.keras.layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)
      x = tf.keras.layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)
      x = tf.keras.layers.MaxPooling2D((2,2), strides = (2,2))(x)
      x = layers.GlobalAveragePooling2D()(x)
      x = tf.keras.layers.Flatten()(x)
      x = tf.keras.layers.Dense(512, activation='relu')(x)
      x = layers.Dropout(0.3)(x)
      outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
      model = tf.keras.Model(inputs, outputs)
      return model

  final_model = final_model(input_shape=img_size + (3,), num_classes=5)
  final_model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
  final_model.summary()
  ```

- Final Model Summary
  
  ```
  ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
  ┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
  ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
  │ input_layer_10 (InputLayer)          │ (None, 224, 224, 3)         │               0 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ conv2d_32 (Conv2D)                   │ (None, 224, 224, 8)         │             224 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ max_pooling2d_20 (MaxPooling2D)      │ (None, 112, 112, 8)         │               0 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ conv2d_33 (Conv2D)                   │ (None, 112, 112, 16)        │           1,168 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ conv2d_34 (Conv2D)                   │ (None, 112, 112, 16)        │           2,320 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ conv2d_35 (Conv2D)                   │ (None, 112, 112, 16)        │           2,320 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ max_pooling2d_21 (MaxPooling2D)      │ (None, 56, 56, 16)          │               0 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ global_average_pooling2d_6           │ (None, 16)                  │               0 │
  │ (GlobalAveragePooling2D)             │                             │                 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ flatten_10 (Flatten)                 │ (None, 16)                  │               0 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ dense_20 (Dense)                     │ (None, 512)                 │           8,704 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ dropout_3 (Dropout)                  │ (None, 512)                 │               0 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ dense_21 (Dense)                     │ (None, 5)                   │           2,565 │
  └──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
  Total params: 17,301 (67.58 KB)
  Trainable params: 17,301 (67.58 KB)
  Non-trainable params: 0 (0.00 B)
  ```

- Model Evaluation 시각화
  
  ![alt text](/assets/images/cnnproject_final_accuracy.png)<br/>
  ![alt text](/assets/images/cnnproject_final_loss.png)<br/>

  ```
  Restoring model weights from the end of the best epoch: 34.
  ```
  Best score인 34번 째 epoch의 evaluation은 아래와 같다.<br/>
  Train Accuracy: 0.9899<br/>
  Train Loss: 0.0310<br/>
  Validation Accruacy: 0.9897<br/>
  Validation Loss: 0.0305<br/>
  
- feature map 시각화

  ![alt text](/assets/images/cnnproject_final_layer1.png)<br/>
  ![alt text](/assets/images/cnnproject_final_layer2.png)<br/>
  ![alt text](/assets/images/cnnproject_final_layer3.png)<br/>
  ![alt text](/assets/images/cnnproject_final_layer4.png)<br/>

- 총 필터 수: 56개

- 이미지 1장 당 FLOPs : 167,860,766 (약 0.17 GFLOPs)<br/>
  이미지 7,500장 : 약 1.26 TFLOPs

- Test Accuracy : 0.9533

- Confusion Matrix<br/>
  ![alt text](/assets/images/cnnproject_final_confusionmatrix.png)<br/>

  | Class        | Precision | Recall | F1-Score | Support |
  |--------------|-----------|--------|----------|---------|
  | **Arborio**   | 0.91      | 0.97   | 0.94     | 1500    |
  | **Basmati**   | 0.98      | 0.93   | 0.96     | 1500    |
  | **Ipsala**    | 0.99      | 1.00   | 0.99     | 1500    |
  | **Jasmine**   | 0.93      | 0.92   | 0.92     | 1500    |
  | **Karacadag** | 0.96      | 0.95   | 0.95     | 1500    |
  |              |           |        |          |         |
  | **Accuracy**  |           |        | **0.95** | 7500    |
  | **Macro Avg** | 0.95      | 0.95   | 0.95     | 7500    |
  | **Weighted Avg** | 0.95   | 0.95   | 0.95     | 7500    |

## (2) 최종 성능 비교표

![alt text](/assets/images/cnnproject_final_summary.png)<br/>
![alt text](/assets/images/cnnproject_final_flops.png)<br/>
![alt text](/assets/images/cnnproject_final_filter.png)<br/>

&nbsp;&nbsp;결론적으로 본 프로젝트 결과를 종합하여 계산량은 줄이면서 성능을 끌어올린 모델을 제작할 수 있었다. 따라서 해당 모델을 바탕으로 Feature Map 기반의 시각적 분석을 통해 불필요한 필터를 제거하고 구조적으로 꼭 필요한 요소만을 유지하는 방식은 충분히 타당하며 실제로도 효과적임을 입증하였다. 향후 후속 연구를 통해 보다 정량적인 기준과 자동화된 필터 최적화 방법이 도입된다면 다양한 모델 구조에 널리 적용될 수 있을 것으로 기대된다.

# 6. Reference
1. Koklu, Murat. “Rice Image Dataset.” Kaggle. Accessed March 26, 2025. https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset/data.

2. Joshi, Shardul. “Rice Classification Using VGG16 - 99% Accuracy.” Kaggle, September 8, 2022. https://www.kaggle.com/code/sharduljoshi29/rice-classification-using-vgg16-99-accuracy.

3. 최진영. “산업인공지능 수업자료”. 아주대학교 산업공학과. 2024.