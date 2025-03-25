---
title: CNN 연구 개인프로젝트
date: 2025-03-22
categories: [Project, KTB]
tags: [python, langchain, fastapi]
math: true
---

# 서론
&nbsp;&nbsp;최근 다양한 CNN(Convolutional Neural Network) 기반의 이미지 분류 모델들이 제안되며 이미지 인식 분야에서 뛰어난 성능을 보이고 있다. 하지만 이러한 모델들은 구조의 복잡성이나 파라미터 수에 따라 연산 비용과 메모리 사용량에 큰 차이를 보이며, 특히 데이터셋의 특성에 따라 성능 및 효율성이 달라질 수 있다.<br/>
&nbsp;&nbsp;실제 응용 환경에서는 모델의 경량화 또한 중요한 과제로 떠오르고 있다. 모바일 기기, 임베디드 시스템, 자동화 기계 등에서는 모델의 정확도뿐만 아니라 처리 속도와 자원 효율성이 중요한 요소로 작용하기 때문이다.<br/>
&nbsp;&nbsp;따라서 본 프로젝트에서는 'Rice Image Dataset'을 활용하여 여러 CNN 기반 모델들이 해당 데이터셋에서 어떤 성능을 보이는지 비교하고 모델별 특징 및 효율성을 분석하고자 한다. 특히, 이미지 분류 과정에서 생성되는 Feature map을 시각화함으로써 각 모델이 어떤 방식으로 이미지를 인식하고 구분하는지 직관적으로 이해하고자 하였다.<br/>
&nbsp;&nbsp;이를 통해 더 이상 특징을 제대로 추출하지 못 하는 Layer를 일부 제거하거나 Filter의 수를 줄임 성능저하 없이 예측 효율성을 개선할 수 있는 가능성도 살펴보고자한다. 이러한 분석을 통해 단순 정확도 비교를 넘어서 실제 응용에 적합한 효율적인 모델을 선정하기 위한 방법을 살펴보고자 한다.<br/>

## 모델 경량화
&nbsp;&nbsp;CNN 계열 모델의 이론적 추론시간은 아래와 같다.
$$
Inference\,Time = \frac{\sum_{l=1}^{L}FLOPs_l}{Device\,Performance\,(FLOPs/sec)}
$$

&nbsp;&nbsp;FLOPs란 FLoating point Operations의 약자로 부동소수점 연산을 의미하며 주로 모델의 계산 복잡성을 측정하는데 사용된다. Device Performance는 대개 FLOPS(FLoating point Operations Per Second)로 측정하고 있으며 추론 시간은 Device의 성능이 높을수록 계산해야하는 FLOPs가 낮을수록 추론시간은 짧아진다.<br/>
&nbsp;&nbsp;일반적으로 Convolution Lyaer의 FLOPs의 계산은 아래와 같이 계산된다.<br/>

$$
FLOPs_{conv} = 2 \times C_{in} \times K^2 \times H_{out} \times W_{out} \times C_{out}
$$

$C_{in}$: 입력 채널 수<br/>
$K$: 커널크기<br/>
$H_{out}, W_{out}$: Feature map의 height, width<br/>
$C_{out}$: 출력 채널 수<br/>

&nbsp;&nbsp;여기서 Layer를 제거하거나 Filter의 개수를 줄여 다음 Layer에서의 $C_{in}$이 줄어든다. 따라서 계산량이 줄어들어 상당한 FLOPs의 이득을 볼 수 있다.

## CNN의 기본 구조
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

&nbsp;&nbsp;Convolution → ReLU → MaxPooling 형태의 CNN 층을 수식으로 표현하면 아래와 같다.<br/>
$X \in \mathbb{R}^{C_{in}\times H \times W}$<br/>
$W \in \mathbb{R}^{C_{out}\times C_{in} \times K' \times K}$<br/>
$b_i \in \mathbb{R}$<br/>
&nbsp;&nbsp;출력 채널 $i$, 위치 $(m, n)$에서의 convolution 출력은 아래의 식과 같다.
$$
Z_i(m, n) = \sum_{c=1}^{C_{\text{in}}} \sum_{u=1}^{K} \sum_{v=1}^{K} W_{i,c,u,v} \cdot X_c(m + u, n + v) + b_i
$$
&nbsp;&nbsp;ReLU는 음수를 0으로 만들고 양수는 그대로 유지하므로 아래의 식과 같다.
$$
\text{ReLU}(Z_i(m, n)) = \max(0, Z_i(m, n))
$$
&nbsp;&nbsp; $2 \times 2$ 커널에서의 Max Pooling은 아래의 식과 같다.
$$
P_i(p, q) = \max_{\substack{0 \leq m < k \\ 0 \leq n < k}} A_i(s \cdot p + m,\, s \cdot q + n)
$$

&nbsp;&nbsp; 따라서 모든 Feature Map의 값이 0인 경우를 보면 아래와 같이 출력된다.
$$
Z_i(m,n) = 0 \,\,\, \forall i,m,n \\
ReLU(Z_{i}(m,n))\max(0, Z_i(m,n)) = 0 \\
P_i(p, q) = \max(0, 0, 0, 0) = 0
$$

&nbsp;&nbsp;이와 같이 Feature Map의 값이 모두 0이면 ReLU함수와 Pooling 층을 통과해도 그 결과는 0이 나온다.<br/>
&nbsp;&nbsp;Fully Connected Layer는 $z = Wx + b$와 같이 표현되고 모든 $x = 0$이라면 학습결과는 bias($b$)에 의존할 수 밖에 없으며 이는 추출된 특징으로 학습할 수 없다. 따라서 학습할 정보가 사라지므로 해당 모델은 찍기의 형태를 띌 수 밖에 없다.<br/>

&nbsp;&nbsp;실제 해당 프로젝트를 진행하면서 GoogLeNet을 학습시킬 때, weight를 초기화하며 학습하다보면 학습이 되지 않는 현상이 생겼다.<br/>
![alt text](/assets/images/cnnproject_learningerror.png)<br/>
![alt text](/assets/images/cnnproject_learningerroraccuracy.png)<br/>
![alt text](/assets/images/cnnproject_learningerrorloss.png)<br/>

[학습 불가 Colab Link](https://colab.research.google.com/drive/1Wt_dMel9Vv8HwzGIlkuvochkIFNWLsv0?usp=sharing), [프로젝트 Colab Link](https://colab.research.google.com/drive/1WgjlYTLEkDyacbimKDHpdKw0LBZn4aTi?usp=sharing) 두 링크를 통해 비교해보면 코드는 같은 걸 알 수 있다.<br/>

&nbsp;&nbsp;따라서 본 프로젝트는 위의 결과를 토대로 시각적으로 Feature Map을 살펴보고 불필요한 Filter가 생성되는 Layer는 제거하거나 Filter의 개수를 줄여보면서 모델을 경량화하고 성능을 비교해보고자 한다.

# 데이터셋 설명
&nbsp;&nbsp;본 프로젝트에서 사용한 데이터셋은 Kaggle의 'Rice Image Dataset'으로 Murat Koklu에 의해 제공되었다.<br/>
&nbsp;&nbsp;해당 데이터셋은 Arborio, Basmati, Ipsala, Jasmine, Karacadag로 총 5가지 class로 구분되어 있으며 각 데이터는 15,000개로 총 75,000개의 `.jpg` 이미지로 구성되어있다.<br/>
&nbsp;&nbsp; 각 이미지는 250x250 픽셀 크기를 가지며 검은 배경 위에 단일 쌀알이 위치한 형태로 구성되어 있다. 이미지들은 쌀알 이외의 잡음은 없고 배경과 객체가 명확히 구분되도록 전처리 되어있어 이미지 분류 모델 학습에 적합한 데이터이다.

## Data Load
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

# 모델 구성
## CNN
- 일반적인 CNN으로 Convolution → ReLu → MaxPooling 으로 3개 층으로 쌓아보았으며 필터의 갯수는 32 → 64 → 128 개를 사용하여 Feature map을 시각화 하였다.
  
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
  Model: "functional"
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

  부분적으로 filter을 거친 후 feature map이 검은색인 경우가 많다. 이는 계산에 큰 영향을 미치지 않을 것이라 판단되어 filter의 개수를 줄여 개선을 경량화를 해보고자 한다.

## CNN 경량화
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
  Model: "functional_2"
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

## 모델 비교

|Model|Parameter|Validation Accuracy|Validation Loss|
|---|--------|--|--|
|CNN|12,939,077 (49.36 MB)|0.9963|0.0132|
|CNN 경량화|**51,669** (201.83 KB)|**0.9970**|**0.0108**|

Parameter 수는 각 $12,939,077$와 $51,669$로 $99.60(\%)$ 경량화 감소하였으며 Accuracy와 Loss를 보았을 때, 성능차이는 오차범위 이내로 소폭 증가하였다.

## GoogLeNet
- GoogLeNet은 LeNet을 활용한 어쩌구... tensorflow로 구현해보았다.

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
  Model: "functional_115"
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

## GoogLeNet 경량화
- GoogLeNet은 Stem Network의 filter의 개수를 줄이고 각 Inception Module의 filter 개수를 줄여주었다. 또한 2단계의 Inception Module하나를 삭제하므로써 연산량을 줄였다. Inception Module의 구조는 GoogLeNet의 특징이므로 건드리지 않았다.
  
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
  Model: "functional_24"
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

## 모델 비교

|Model|Parameter|Validation Accuracy|Validation Loss|
|---|--------|--|--|
|GoogLeNet|5,978,677 (22.81 MB)|0.9975|0.0111|
|GoogLeNet 경량화|**580,905** (2.22 MB)|**0.9981**|**0.0070**|

Parameter 수는 각 $5,978,677$와 $580,905$로 $90.28(\%)$ 경량화 감소하였으며 Accuracy와 Loss를 보았을 때, 성능차이는 오차범위 이내로 소폭 증가하였다.

## VGG16
- VGG16 어쩌구 구조는 어쩌구 저쩌구
  
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
  Model: "functional_177"
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


## VGG16 경량화
- Pre-trained 되어있는 Filter를 그대로 사용하고 block_4와 block_5는 특징 추출이 되지 않은 형태로 보이므로 삭제하여 경량화

  ```python
  x = vgg16.get_layer('block3_pool').output

  # 분류기 추가
  x = GlobalAveragePooling2D()(x)
  x = Dense(256, activation='relu')(x)
  x = Dropout(0.2)(x)
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
  Model: "functional_56"
  ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
  ┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
  ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
  │ input_layer_49 (InputLayer)          │ (None, 224, 224, 3)         │               0 │
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
  │ global_average_pooling2d_6           │ (None, 256)                 │               0 │
  │ (GlobalAveragePooling2D)             │                             │                 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ dense_11 (Dense)                     │ (None, 256)                 │          65,792 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ dropout_5 (Dropout)                  │ (None, 256)                 │               0 │
  ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
  │ dense_12 (Dense)                     │ (None, 5)                   │           1,285 │
  └──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
  Total params: 1,802,565 (6.88 MB)
  Trainable params: 67,077 (262.02 KB)
  Non-trainable params: 1,735,488 (6.62 MB)
  ```

- Model Evaluation 시각화
  
  ![alt text](/assets/images/cnnproject_vgg16_accuracy.png)<br/>
  ![alt text](/assets/images/cnnproject_vgg16_loss.png)<br/>

  ```
  Restoring model weights from the end of the best epoch: 26.
  ```
  Best score인 11번 째 epoch의 evaluation은 아래와 같다.<br/>
  Train Accuracy: 0.9970<br/>
  Train Loss: 0.0102<br/>
  Validation Accruacy: 0.9979<br/>
  Validation Loss: 0.0070<br/>
  
- feature map 시각화<br/>

  ![alt text](/assets/images/cnnproject_vgg16_layer1.png)<br/>
  ![alt text](/assets/images/cnnproject_vgg16_layer2.png)<br/>
  ![alt text](/assets/images/cnnproject_vgg16_layer3.png)<br/>

  사전 학습된 모델 그대로 가져왔으므로 Filter를 통과한 Feature map은 동일하다.

## 모델 비교

|Model|Parameter|Validation Accuracy|Validation Loss|
|---|--------|--|--|
|VGG16|14,979,909 (57.14 MB)|0.9979|0.0099|
|VGG 경량화|**1,802,565** (6.88 MB)|**0.9979**|**0.0070**|

Parameter 수는 각 $5,978,677$와 $580,905$로 $87.97(\%)$ 경량화 감소하였으며 Accuracy와 Loss를 보았을 때, 성능차이는 오차범위 이내로 소폭 증가하였다.

## MobileNet

## 결론

## Reference