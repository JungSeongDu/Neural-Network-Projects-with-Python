from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Dense
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

#imdb 데이터셋을 가져온다
training_set,testing_set = imdb.load_data(num_words=10000)
X_train,y_train = training_set
X_test,y_test = testing_set

#print("Number of training samples = {}".format(X_train.shape[0]))
#print("Number of testing samples = {}".format(X_test.shape[0]))

#제로패딩
X_train_padded = sequence.pad_sequences(X_train, maxlen=100)
X_test_padded = sequence.pad_sequences(X_test, maxlen=100)

print("X_train vector shape = {}".format(X_train_padded.shape))
print("X_test vector shape = {}".format(X_test_padded.shape))

# 모델 구성
def train_model(Optimizer, X_train, y_train, X_val, y_val):
    model = Sequential()
    model.add(Embedding(input_dim=10000, output_dim=128))
    model.add(LSTM(units=128))
    model.add(Dense(units=1, activation="sigmoid"))
    #model.summary()
    model.compile(loss="binary_crossentropy",optimizer=Optimizer,metrics=['accuracy'])
    scores = model.fit(X_train,y_train,batch_size=128,epochs=10,validation_data=(X_val,y_val))

    return scores,model

#모델 훈련
"""
SGD_score,SGD_model = train_model(Optimizer='sgd',
                                   X_train=X_train_padded,
                                   y_train=y_train,
                                   X_val=X_test_padded,
                                   y_val=y_test)
"""
RMSprop_score, RMSprop_model = train_model(Optimizer = 'RMSprop', 
                                           X_train=X_train_padded, 
                                           y_train=y_train, 
                                           X_val=X_test_padded, 
                                           y_val=y_test)

"""
Adam_score,Adam_model = train_model(Optimizer='adam',
                                    X_train=X_train_padded,
                                    y_train=y_train,
                                    X_val=X_test_padded,
                                    y_val=y_test)
"""

#훈련 및 검증 정확도 차트를 그린다
"""
plt.plot(range(1,11), SGD_score.history['accuracy'],label='Training Accuracy')
# SGD_scores.history['acc'] : 모델의 각 훈련 단계(epoch)에서의 훈련 정확도를 담고 있는 리스트
plt.plot(range(1,11), SGD_score.history['val_accuracy'],label='Validation Accuracy')
plt.axis([1,10,0,1])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Train and Validation Accuracy using SGD Optimizer')
plt.legend()
plt.show()
"""

plt.plot(range(1,11), RMSprop_score.history['accuracy'],label='Training Accuracy')
plt.plot(range(1,11), RMSprop_score.history['val_accuracy'],label='Validation Accuracy')
plt.axis([1,10,0,1])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Train and Validation Accuracy using RMSprop Optimizer')
plt.legend()
plt.show()

"""
plt.plot(range(1,11), Adam_score.history['accuracy'],label='Training Accuracy')
plt.plot(range(1,11), Adam_score.history['val_accuracy'],label='Validation Accuracy')
plt.axis([1,10,0,1])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Train and Validation Accuracy using Adam Optimizer')
plt.legend()
plt.show()
"""

#혼동행렬을 그린다
#plt.figure(gigsize=(10,7))
#sns.set(font_scale=2)
y_test_pred = RMSprop_model.predict_classes(X_test_padded)
c_matrix = confusion_matrix(y_test,y_test_pred)
ax = sns.heatmap(c_matrix, annot=True, xticklabels=['Negative Sentiment', 'Positive Sentiment'], 
                 yticklabels=['Negative Sentiment', 'Positive Sentiment'], cbar=False, cmap='Blues', fmt='g')
ax.set_xlabel("Prediction")
ax.set_ylabel("Actual")
plt.show()