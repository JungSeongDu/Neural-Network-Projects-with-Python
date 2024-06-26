import os
import random
import warnings
warnings.filterwarnings("ignore")
from utils import train_test_split


src = 'Chapter04/Dataset/kagglecatsanddogs_5340/PetImages/'

# 데이터셋이 있는지 먼저 확인한다. 없다면 내려 받아야 한다
if not os.path.isdir(src):
    print("""
          Dataset not found in your computer.
          Please follow the instructions in the link below to download the dataset:
          https://raw.githubusercontent.com/PacktPublishing/Neural-Network-Projects-with-Python/master/chapter4/how_to_download_the_dataset.txt
          """)
    quit()

# train/test 폴더가 없다면 생성한다
if not os.path.isdir(src+'train/'):
    train_test_split(src)

from keras.models import Sequential
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

# 초매개변수 정의
FILTER_SIZE = 3
NUM_FILTERS = 32
INPUT_SIZE  = 32
MAXPOOL_SIZE = 2
BATCH_SIZE = 16
STEPS_PER_EPOCH = 20000//BATCH_SIZE
EPOCHS = 10

model = Sequential()

model.add(Conv2D(NUM_FILTERS, (FILTER_SIZE, FILTER_SIZE), 
                 input_shape = (INPUT_SIZE, INPUT_SIZE, 3), 
                 activation = 'relu'))
model.add(MaxPooling2D(pool_size = (MAXPOOL_SIZE, MAXPOOL_SIZE)))
model.add(Conv2D(NUM_FILTERS, (FILTER_SIZE, FILTER_SIZE), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (MAXPOOL_SIZE, MAXPOOL_SIZE)))
model.add(Flatten())
model.add(Dense(units = 128, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(units = 1, activation = 'sigmoid'))


model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

training_data_generator = ImageDataGenerator(rescale = 1./255)
testing_data_generator = ImageDataGenerator(rescale = 1./255)


training_set = training_data_generator.flow_from_directory('Chapter04/Dataset/kagglecatsanddogs_5340/PetImages/Train/',
                                                target_size = (INPUT_SIZE, INPUT_SIZE),
                                                batch_size = BATCH_SIZE,
                                                class_mode = 'binary')

test_set = testing_data_generator.flow_from_directory('Chapter04/Dataset/kagglecatsanddogs_5340/PetImages/Test/',
                                             target_size = (INPUT_SIZE, INPUT_SIZE),
                                             batch_size = BATCH_SIZE,
                                             class_mode = 'binary')


model.fit(training_set, steps_per_epoch = STEPS_PER_EPOCH, epochs=EPOCHS, verbose=1)
score = model.evaluate(test_set, steps=100)

for idx, metric in enumerate(model.metrics_names):
    print("{}: {}".format(metric, score[idx]))


# 모델 저장
model.save('main_basic_cnn.h5')

