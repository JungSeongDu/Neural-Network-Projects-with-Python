from keras.datasets import mnist
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
import random

def create_basic_autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units=hidden_layer_size, input_shape=(784,), activation='relu'))
    model.add(Dense(units=784, activation='sigmoid'))
    return model
"""
if __name__ == "__main__":
    #MNIST 데이터 셋을 가져온다
    training_set, testing_set = mnist.load_data()
    X_train, y_train = training_set
    X_test, y_test = testing_set

    #신경망에 사용할 수 있게 백터 형태를 바꾼다
    X_train_reshaped = X_train.reshape((X_train.shape[0],X_train.shape[1] * X_train.shape[2]))
"""
    #X_train.shape[0]은 훈련 세트의 이미지 수
    #X_train.shape[1]은 각 이미지의 세로 픽셀 수(높이)
    #X_train.shape[2]는 각 이미지의 가로 픽셀 수(너비)
"""
    X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1]*X_test.shape[2]))

    #0부터 255까지의 값을 0과1 사이의 값으로 정규화 한다.
    X_train_reshaped = X_train_reshaped/255.
    X_test_reshaped = X_test_reshaped/255.


    model = create_basic_autoencoder(hidden_layer_size=1)
    model.compile(optimizer='adam',loss='mean_squared_error')
    model.fit(X_train_reshaped,X_train_reshaped,epochs=10)

    output=model.predict(X_test_reshaped)

      # 결과 출력
    fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10)) = plt.subplots(2, 5, figsize=(20,7))

    #이미지 5개를 무작위로 고른다
    randomly_selected_imgs = random.sample(range(output.shape[0]),5)

    # 첫 번째 줄에는 원본 이미지를 그린다
    for i, ax in enumerate([ax1,ax2,ax3,ax4,ax5]):
        ax.imshow(X_test[randomly_selected_imgs[i]], cmap='gray')
        if i == 0:
            ax.set_ylabel("INPUT",size=40)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

    # 두 번째 줄에는 노이즈를 추가한 이미지를 그린다
    for i, ax in enumerate([ax6,ax7,ax8,ax9,ax10]):
        ax.imshow(output[randomly_selected_imgs[i]].reshape(28,28), cmap='gray')
        if i == 0:
            ax.set_ylabel("OUTPUT",size=40)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()
"""

if __name__ == '__main__':
    # MNIST 데이터셋을 가져온다
    training_set, testing_set = mnist.load_data()
    X_train, y_train = training_set
    X_test, y_test = testing_set

    # 신경망에 사용할 수 있게 벡터 형태를 바꾼다
    X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1]*X_train.shape[2]))
    X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1]*X_test.shape[2]))

    # 0부터 255까지의 값을 0과 1 사이 값으로 정규화한다
    X_train_reshaped = X_train_reshaped/255.
    X_test_reshaped = X_test_reshaped/255.

    # 은닉 레이어 크기가 다른 오토인코더를 만든다
    hiddenLayerSize_1_model = create_basic_autoencoder(hidden_layer_size=1)
    hiddenLayerSize_2_model = create_basic_autoencoder(hidden_layer_size=2)
    hiddenLayerSize_4_model = create_basic_autoencoder(hidden_layer_size=4)
    hiddenLayerSize_8_model = create_basic_autoencoder(hidden_layer_size=8)
    hiddenLayerSize_16_model = create_basic_autoencoder(hidden_layer_size=16)
    hiddenLayerSize_32_model = create_basic_autoencoder(hidden_layer_size=32)

    # 각 오토인코더를 훈련시킨다
    hiddenLayerSize_1_model.compile(optimizer='adam', loss='mean_squared_error')
    hiddenLayerSize_1_model.fit(X_train_reshaped, X_train_reshaped, epochs=10, verbose=0)

    hiddenLayerSize_2_model.compile(optimizer='adam', loss='mean_squared_error')
    hiddenLayerSize_2_model.fit(X_train_reshaped, X_train_reshaped, epochs=10, verbose=0)

    hiddenLayerSize_4_model.compile(optimizer='adam', loss='mean_squared_error')
    hiddenLayerSize_4_model.fit(X_train_reshaped, X_train_reshaped, epochs=10, verbose=0)

    hiddenLayerSize_8_model.compile(optimizer='adam', loss='mean_squared_error')
    hiddenLayerSize_8_model.fit(X_train_reshaped, X_train_reshaped, epochs=10, verbose=0)

    hiddenLayerSize_16_model.compile(optimizer='adam', loss='mean_squared_error')
    hiddenLayerSize_16_model.fit(X_train_reshaped, X_train_reshaped, epochs=10, verbose=0)

    hiddenLayerSize_32_model.compile(optimizer='adam', loss='mean_squared_error')
    hiddenLayerSize_32_model.fit(X_train_reshaped, X_train_reshaped, epochs=10, verbose=0)

    # 훈련시킨 모델을 사용해 테스트셋의 결과를 예측한다
    output_1_model = hiddenLayerSize_2_model.predict(X_test_reshaped)
    output_2_model = hiddenLayerSize_2_model.predict(X_test_reshaped)
    output_4_model = hiddenLayerSize_4_model.predict(X_test_reshaped)
    output_8_model = hiddenLayerSize_8_model.predict(X_test_reshaped)
    output_16_model = hiddenLayerSize_16_model.predict(X_test_reshaped)
    output_32_model = hiddenLayerSize_32_model.predict(X_test_reshaped)

    # 각 모델의 결과를 그려 비교한다
    fig, axes = plt.subplots(7, 5, figsize=(15,15))

    randomly_selected_imgs = random.sample(range(output_2_model.shape[0]),5)
    outputs = [X_test, output_1_model, output_2_model, output_4_model, output_8_model, output_16_model, output_32_model]

    # 서브차트를 하나씩 그린다
    for row_num, row in enumerate(axes):
      for col_num, ax in enumerate(row):
        ax.imshow(outputs[row_num][randomly_selected_imgs[col_num]].reshape(28,28), cmap='gray')
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()