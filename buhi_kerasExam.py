from tensorflow.python.keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation

# 데이터셋 생성
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') / 255.0
x_test = x_test.reshape(10000, 784).astype('float32') / 255.0
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# 모델 구성
model = Sequential()

# 은닉층 구성
model.add(Dense(128, activation="relu"))
model.add(Dense(64, activation="relu"))

# 출력층 구성
model.add(Dense(10, activation="softmax"))

# 모델 학습(10번 학습시킴)
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=["acc"])

hist = model.fit(x_train, y_train, epochs=10, batch_size=32)

# 학습과정
print('## training loss and acc ##')
print(hist.history['loss'])
print(hist.history['acc'])

# 모델 학습 평가
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=32)
print('## evalution loss and metrics ##')
print(loss_and_metrics)

# 모델 테스트(사용)
xhat = x_test[0:1]
yhat = model.predict(xhat)
print('## yhat ##')
print(yhat)

