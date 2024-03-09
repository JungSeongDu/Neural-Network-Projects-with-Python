import matplotlib
matplotlib.use("TkAgg")
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import numpy as np
import random
import os


image_generator = ImageDataGenerator(rotation_range = 30,
                                     width_shift_range = 0.2,
                                     height_shift_range = 0.2,
                                     zoom_range = 0.2,
                                     horizontal_flip=True,
                                     fill_mode='nearest')

# rotation_range: 이미지 회전의 범위를 지정
# width_shift_range 및 height_shift_range: 이미지를 수평 및 수직으로 무작위로 이동시키는 범위를 지정
# zoom_range: 이미지를 확대 또는 축소하는 범위를 지정
# horizontal_flip: 수평으로 무작위로 이미지를 뒤집음, 좌우 대칭을 이용하여 데이터 다양성을 늘리는 데 도움
# fill_mode: 이미지 이동이나 회전으로 인해 새로운 픽셀이 생성될 경우, 이를 어떻게 채울지 지정
# 'nearest'는 주변 픽셀 값을 사용하여 채우라는 의미

fig, ax = plt.subplots(2,3, figsize=(20,10))
all_images = []

_, _, dog_images = next(os.walk('Chapter04/Dataset/kagglecatsanddogs_5340/PetImages/Train/Dog/'))
random_img = random.sample(dog_images, 1)[0]
print(random_img)
random_img = plt.imread('Chapter04/Dataset/kagglecatsanddogs_5340/PetImages/Train/Dog/'+random_img)
all_images.append(random_img)

random_img = random_img.reshape((1,) + random_img.shape)
# random_img.shape는 원래 이미지 배열의 모양(shape)을 나타내는 튜플
#새로운 형태의 튜플을 생성하고, 그 첫 번째 요소로 1을 추가하는 역할
"""
1을 추가하는 이유는 Keras에서 모델에 입력되는 데이터의 배치 차원을 나타내기 위해서, 
Keras의 대부분의 모델은 데이터를 배치로 처리하는데, 이는 모델에 여러 입력 샘플을 동시에 전달하여 학습하거나 예측하는 데 도움

원래 이미지 배열의 shape이 (height, width, channels)일 때, Keras 모델은 보통 (batch_size, height, width, channels) 형태의 입력을 기대
여기서 batch_size는 한 번에 처리되는 이미지의 개수를 나타냄

따라서 random_img.reshape((1,) + random_img.shape)를 통해 원래 이미지 배열에 추가적인 차원을 첫 번째로 넣어주어,
이 이미지를 하나의 배치로 처리할 수 있도록 함, 최종적으로 변환된 형태는 (1, height, width, channels)가 됨

이렇게 1을 추가함으로써, 단일 이미지도 배치 형태로 모델에 전달할 수 있게 되며, 
Keras에서 데이터를 일괄 처리(batch processing)하는 방식과 일치시킬 수 있음
"""



sample_augmented_images = image_generator.flow(random_img)
# flow : flow 메서드는 이미지 데이터를 지속적으로 생성하는 제너레이터를 반환
# 이러한 제너레이터를 사용하면 데이터를 증강한 후에도 원래 이미지와 동일한 형태로 사용가능






for _ in range(5):
	augmented_imgs = sample_augmented_images.next()
	for img in augmented_imgs:
		all_images.append(img.astype('uint8'))

for idx, img in enumerate(all_images):
	ax[int(idx/3), idx%3].imshow(img)
	ax[int(idx/3), idx%3].axis('off')
	if idx == 0:
		ax[int(idx/3), idx%3].set_title('Original Image')
	else:
		ax[int(idx/3), idx%3].set_title('Augmented Image {}'.format(idx))


plt.show()