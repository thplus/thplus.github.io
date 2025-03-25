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
&nbsp;&nbsp;이를 통해 모델 경량화 여부를 판단하여 보고 모델 경량화 기법을 일부 적용해 봄으로써 성능저하 없이 예측 효율성을 개선할 수 있는 가능성도 살펴보고자한다. 이러한 분석을 통해 단순 정확도 비교를 넘어서 실제 응용에 적합한 효율적인 모델을 선정하기 위한 방법을 살펴보고자 한다.<br/>

## 모델 경량화

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

### 두 모델 비교

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



## MobileNet

## 결론

## Reference