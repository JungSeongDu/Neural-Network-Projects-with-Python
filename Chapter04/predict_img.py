import os
import numpy as np
from keras.preprocessing import image
from keras.models import load_model



# 이미지 전처리 및 예측
def predict_image(model, image_path):
    img = image.load_img(image_path, target_size=(32, 32))  # 이미지를 (64, 64) 크기로 조정합니다.
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # 배치 차원을 추가합니다.
    img_array = img_array / 255.0  # 이미지를 [0, 1] 범위로 정규화합니다.

    result = model.predict(img_array)
    if result[0][0] > 0.5:
        print("강아지입니다.")
    else:
        print("고양이입니다.")


# 모델 로드
model = load_model('main_basic_cnn.h5')  # 모델 경로 지정

# 이미지 경로
image_path = 'Chapter04/14.jpg'  # 이미지 경로 지정

# 이미지 예측
predict_image(model, image_path)