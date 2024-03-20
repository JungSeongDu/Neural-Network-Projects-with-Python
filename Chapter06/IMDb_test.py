from keras.datasets import imdb

training_set,testing_set = imdb.load_data(index_from = 3)

X_train, y_train = training_set
X_test, y_train = testing_set

#print(X_train[0])

word_to_id = imdb.get_word_index()
print(word_to_id["happy"])
word_to_id = {key:(value + 3) for key,value in word_to_id.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1

# word_to_id 딕셔너리를 사용하여 정수로 된 단어 인덱스를 단어로 변환하는 id_to_word 딕셔너리를 생성
# word_to_id 딕셔너리의 키와 값을 서로 뒤집어서 새로운 딕셔너리를 생성
id_to_word = {value:key for key,value in word_to_id.items()}
print(id_to_word[654])

print(' '.join(id_to_word[id] for id in X_train[159]))
# X_train[159]에 있는 정수 시퀀스를 텍스트로 변환하여 출력
# X_train[159] => 훈련 데이터 중에서 159번째 리뷰에 해당하는 정수 시퀀스
# id_to_word[id] for id in X_train[159] => X_train[159]에 있는 각 정수를 id_to_word 딕셔너리를 사용하여 해당하는 단어로 변환한 리스트를 생성
#join() 함수 => 이 리스트를 공백으로 구분하여 하나의 문자열로 합칩니다

#긍정리뷰
print(y_train[159])

print(' '.join(id_to_word[id] for id in X_train[6]))

