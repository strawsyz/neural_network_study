from keras.datasets import imdb
from keras.preprocessing import sequence

'''大概要运行几个小时
Train on 20000 samples, validate on 5000 samples
Epoch 1/10
20000/20000 [==============================] - 77s 4ms/step - loss: 0.6187 - acc: 0.6672 - val_loss: 0.4182 - val_acc: 0.8374
Epoch 2/10
20000/20000 [==============================] - 76s 4ms/step - loss: 0.3784 - acc: 0.8321 - val_loss: 0.3928 - val_acc: 0.8382
Epoch 3/10
20000/20000 [==============================] - 76s 4ms/step - loss: 0.3022 - acc: 0.7488 - val_loss: 0.7596 - val_acc: 0.6148
Epoch 4/10
20000/20000 [==============================] - 77s 4ms/step - loss: 0.2451 - acc: 0.6546 - val_loss: 1.2835 - val_acc: 0.4584
Epoch 5/10
20000/20000 [==============================] - 76s 4ms/step - loss: 0.1807 - acc: 0.5944 - val_loss: 0.6607 - val_acc: 0.5356
Epoch 6/10
20000/20000 [==============================] - 76s 4ms/step - loss: 0.1409 - acc: 0.5066 - val_loss: 0.7663 - val_acc: 0.4618
Epoch 7/10
20000/20000 [==============================] - 76s 4ms/step - loss: 0.1142 - acc: 0.4019 - val_loss: 0.8708 - val_acc: 0.3738
Epoch 8/10
20000/20000 [==============================] - 76s 4ms/step - loss: 0.1061 - acc: 0.2960 - val_loss: 0.9799 - val_acc: 0.3154
Epoch 9/10
20000/20000 [==============================] - 76s 4ms/step - loss: 0.1025 - acc: 0.2288 - val_loss: 1.0578 - val_acc: 0.2922
Epoch 10/10
20000/20000 [==============================] - 76s 4ms/step - loss: 0.0976 - acc: 0.1918 - val_loss: 1.2339 - val_acc: 0.2682
25000/25000 [==============================] - 27s 1ms/step
'''
max_features = 10000
max_len = 500
print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

'''使用cnn网络进行影评正负能量的判断'''
model = Sequential()
model.add(layers.Embedding(max_features, 128, input_length=max_len))
# layers.Conv1D(32, 7, activation=‘relu’)，其中的7表示把文本序列中，每7个单词当做一个切片进行识别，
model.add(layers.Conv1D(32, 7, activation='relu'))  # 一维卷积层
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(32, 7, activation='relu'))  # 一维卷积层
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(1))

model.summary()

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
# 训练网络
history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)
# 用测试数据评估
results = model.evaluate(x_test, y_test)

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
plt.show()

import numpy as np

'''
假设现在是1点，我们要预测2点时的气温，由于当前数据记录的是每隔10分钟时的气象数据，1点到2点
间隔1小时，对应6个10分钟，这个6对应的就是delay

要训练网络预测温度，就需要将气象数据与温度建立起对应关系，我们可以从1点开始倒推10天，从过去
10天的气象数据中做抽样后，形成训练数据。由于气象数据是每10分钟记录一次，因此倒推10天就是从
当前时刻开始回溯1440条数据，这个1440对应的就是lookback

我们无需把全部1440条数据作为训练数据，而是从这些数据中抽样，每隔6条取一条，
因此有1440/6=240条数据会作为训练数据，这就是代码中的lookback//step

于是我就把1点前10天内的抽样数据作为训练数据，2点是的气温作为数据对应的正确答案，由此
可以对网络进行训练
'''
def generator(data, lookback, delay, min_index, max_index, shuffle=False,
             batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)
        samples = np.zeros((len(rows), lookback//step, data.shape[-1]))
        targets = np.zeros((len(rows), ))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets


'''预测天气'''
lookback = 1440
step = 6
delay = 144
batch_size = 128

import os
data_dir = 'D:/alab/net_data/jena_climate_2009_2016.csv'
fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')

f = open(fname)
data = f.read()
f.close()

lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]

print(header)
print(len(lines))

import numpy as np
float_data = np.zeros((len(lines), len(header) - 1))
for i, line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]
    float_data[i, :] = values

from matplotlib import pyplot as plt
temp = float_data[:, 1]
plt.plot(range(len(temp)), temp)

mean = float_data[:200000].mean(axis=0)
float_data -= mean
std = float_data[:200000].std(axis=0)
float_data /= std

train_gen = generator(float_data, lookback=lookback,
                      delay=delay, min_index=0,
                      max_index=200000, shuffle=True,
                      step=step,batch_size=batch_size)
val_gen = generator(float_data, lookback=lookback,
                   delay=delay, min_index=200001,
                   max_index=300000,
                   step=step, batch_size=batch_size)
test_gen = generator(float_data, lookback=lookback,
                   delay=delay, min_index=300001,
                   max_index=400000,
                   step=step, batch_size=batch_size)

val_steps = (300000 - 200001 - lookback) // batch_size
test_steps = (len(float_data) - 300001 - lookback) //batch_size

from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

model = Sequential()
#在顶层加1个卷积网络
model.add(layers.Conv1D(32, 5, activation='relu',
                        input_shape=(None, float_data.shape[-1])))
model.add(layers.MaxPooling1D(3))
#添加一个有记忆性的GRU层
model.add(layers.GRU(32, dropout=0.1, recurrent_dropout=0.5))
model.add(layers.Dense(1))
model.summary()

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen, steps_per_epoch=500, epochs=20,
                             validation_data=val_gen,
                             validation_steps=val_steps)