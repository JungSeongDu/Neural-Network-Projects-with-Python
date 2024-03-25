import matplotlib
matplotlib.use("TkAgg")
from keras.preprocessing.image import load_img, img_to_array
import os
import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Lambda, Input
from keras import backend as K

def create_shared_network(input_shape):
    model = Sequential(name='Shared_Conv_Network')
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D())
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(units=128, activation='sigmoid'))
    return model

def euclidean_distance(vectors):
    vector1, vector2 = vectors
    sum_square = K.sum(K.square(vector1 - vector2), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

faces_dir = 'Chapter07/att_faces/'

X_train, Y_train = [], []
X_test, Y_test = [], []

# faces_dir 아래의 하위 디렉터리 목록을 가져온다
# 각 하위 디렉터리는 대상의 이미지를 담았다
subfolders = sorted([file.path for file in os.scandir(faces_dir) if file.is_dir()])

# 하위 디렉터리 목록을 대상으로 반복문을 실행한다
# idx를 대상으로 ID로 사용한다
for idx, folder in enumerate(subfolders):
    for file in sorted(os.listdir(folder)):
        img = load_img(folder+"/"+file, color_mode='grayscale')
        img = img_to_array(img).astype('float32')/255
        img = img.reshape(img.shape[0], img.shape[1],1)
        if idx < 35:
            X_train.append(img)
            Y_train.append(idx)
        else:
            X_test.append(img)
            Y_test.append(idx-35)

X_train = np.array(X_train)
X_test = np.array(X_test)
Y_train = np.array(Y_train)
Y_test = np.array(Y_test)


input_shape = X_train.shape[1:]
shared_network = create_shared_network(input_shape)

input_top = Input(shape=input_shape)
input_bottom = Input(shape=input_shape)

output_top = shared_network(input_top)
output_bottom = shared_network(input_bottom)

distance = Lambda(euclidean_distance,output_shape=(1,))([output_top,output_bottom])
model = Model(inputs=[input_top,input_bottom],outputs=distance)

print(model.summary())

"""
subject_idx = 4
fig, ((ax1,ax2,ax3),(ax4,ax5,ax6),(ax7,ax8,ax9)) = plt.subplots(3,3,figsize=(10,10))
subject_img_idx = np.where(Y_train==subject_idx)[0].tolist()
for i, ax in enumerate([ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9]):
    img = X_train[subject_img_idx[i]]
    img = img.reshape(img.shape[0], img.shape[1])
    ax.imshow(img, cmap='gray')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.show()

# 첫 9명의 사진을 그린다
subjects = range(10)
fig, ((ax1,ax2,ax3),(ax4,ax5,ax6),(ax7,ax8,ax9)) = plt.subplots(3,3,
figsize=(10,12))
subject_img_idx = [np.where(Y_train==i)[0].tolist()[0] for i in subjects]
for i, ax in enumerate([ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9]):
    img = X_train[subject_img_idx[i]]
    img = img.reshape(img.shape[0], img.shape[1])
    ax.imshow(img, cmap='gray')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Subject {}".format(i))
plt.tight_layout()
plt.show()
"""

