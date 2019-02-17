from keras.datasets import imdb
from keras.preprocessing import sequence
'''获得训练和测试数据，要比rnn的多花一点时间'''
max_features = 10000 #只考虑最常使用的前一万个单词
maxlen = 500 #一篇文章只考虑最多500个单词
batch_size = 32
print("Loading data....")
(input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=max_features)
print(len(input_train), 'train sequence')  # 25000的训练数据
print(len(input_test), 'test sequence')  # 25000的测试数据

print('Pad sequences (samples , time)')
input_train = sequence.pad_sequences(input_train, maxlen = maxlen)
input_test = sequence.pad_sequences(input_test, maxlen=maxlen)
print('input_train shape:', input_train.shape)  # (25000, 500)
print('input_test shape: ', input_test.shape)  # (25000, 500)

from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Embedding, Dense

model = Sequential()
model.add(Embedding(max_features, 32))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(input_train, y_train, epochs=10, batch_size=128, validation_split=0.2)