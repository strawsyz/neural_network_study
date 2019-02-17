from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN
'''第一个使用rnn的网络结构
大概要运行几分钟，
验证数据的准确率最高可以达到0.87左右
SimpleRNN无法记忆过长的单词串'''
def pre():
    # SimpleRNN网络层有一个特点是，他必须同时接收一批输入，而不能像我们以为那样一条条的把数据传入网络，而是要一下子把一批数据传进去。
    # 也就是我们在使用SimpleRNN层时，需要把数据集合成一个3维向量(batch_size, timesteps, inputput_features),当数据处理完毕后，
    # 它可以一下子输出一批结果例如(batch_size, timesteps, output_features)，要不然就输出最后一条结果(batch_size, output_features)。
    # 我们看一个具体例子，下面代码使得网络输出最后一条结果：
    model = Sequential()
    model.add(Embedding(10000, 32))
    model.add(SimpleRNN(32))
    model.summary()
    # 下面代码使得网络一下子输出一批结果：
    model = Sequential()
    model.add(Embedding(10000, 32))
    model.add(SimpleRNN(32, return_sequences=True))
    model.summary()

from keras.datasets import imdb
from keras.preprocessing import sequence
'''获得训练和测试数据'''
max_features = 10000 #只考虑最常使用的前一万个单词
maxlen = 500 #一篇文章只考虑最多500个单词
batch_size = 32
epochs = 10
print("Loading data....")
(input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=max_features)
print(len(input_train), 'train sequence')  # 25000的训练数据
print(len(input_test), 'test sequence')  # 25000的测试数据

print('Pad sequences (samples , time)')
input_train = sequence.pad_sequences(input_train, maxlen = maxlen)
input_test = sequence.pad_sequences(input_test, maxlen=maxlen)
print('input_train shape:', input_train.shape)  # (25000, 500)
print('input_test shape: ', input_test.shape)  # (25000, 500)

'''构造一个RNN网络，把数据输入网络进行训练：'''
from keras.layers import Dense
model = Sequential()
model.add(Embedding(max_features, 32))
model.add(SimpleRNN(32))
model.add(Dense(1, activation='sigmoid'))  # 输出一个字符判断是否是正能量

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(input_train, y_train, epochs = epochs, batch_size = 128, validation_split=0.2)

'''绘制训练结果'''
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label = 'Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()

plt.show()