import os
data_dir = 'D:/alab/net_data/jena_climate_2009_2016.csv'
fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')

f = open(fname)
data = f.read()
f.close()

lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]

import numpy as np
float_data = np.zeros((len(lines), len(header) - 1))
for i, line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]  # 取除了日期以外的数据
    float_data[i, :] = values  # values 的格式 [1000.32, -4.09, 269.05, -7.23, 78.6, 4.51, 3.54, 0.96, 2.21, 3.54, 1293.37, 1.25, 1.6, 199.2]

def pre():
    print(header)  # 每一列的列名
    print(len(lines))  # 数据总共有420551个条目
    '''画出'''
    from matplotlib import pyplot as plt
    temp = float_data[:, 1]  # 将温度的数据取出来，作为y轴
    plt.plot(range(len(temp)), temp)
    plt.show()
    plt.plot(range(1440), temp[:1440])  # 显示10天的温度变化
    plt.show()

'''将数据归一化'''
# mean 求平均值
mean = float_data[:200000].mean(axis=0)  # 求前200000个数据的平均值，每一个项目一个平均值
float_data -= mean
# std 求标准平方差
std = float_data[:200000].std(axis=0)
float_data /= std


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

lookback = 1440
step = 6
delay = 144
batch_size = 128

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

def  evaluate_naive_method():
    '''模拟人计算的，预测下一个小时的温度和现在一样'''
    batch_maes = []
    for step in range(val_steps):
        samples, targets = next(val_gen)
        #preds是当前时刻温度，targets是下一小时温度
        preds = samples[:, -1, 1]
        mae = np.mean(np.abs(preds - targets))
        batch_maes.append(mae)
    print(np.mean(batch_maes))  # 0.2897359729905486
    print(np.mean(batch_maes)*std[1])  # 2.564887434980494
# evaluate_naive_method()

from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
# '''要运行几分钟直接使用全连接网络，效果很差，还不如人的预测'''
# model = Sequential()
# model.add(layers.Flatten(input_shape=(lookback // step,
#                                       float_data.shape[-1])))
# model.add(layers.Dense(32, activation='relu'))
# model.add(layers.Dense(1))
#
# model.compile(optimizer=RMSprop(), loss='mae')
# history = model.fit_generator(train_gen, steps_per_epoch=500,
#                              epochs=20,
#                              validation_data=val_gen,
#                              validation_steps=val_steps)

'''大概要20分钟，第四次val_loss最小，0.26左右,比人预测的0.29好很多
使用反复性神经网络GRU，LSTM的变种，对LSTM优化，使得可以变得更快'''
# model = Sequential()
# model.add(layers.GRU(32, input_shape=(None, float_data.shape[-1])))
# model.add(layers.Dense(1))
#
# model.compile(optimizer=RMSprop(), loss='mae')
# history = model.fit_generator(train_gen, steps_per_epoch=500,
#                        epochs = 20, validation_data=val_gen,
#                        validation_steps=val_steps)


'''因为训练了40次，大概要40分钟
增加了随机将权重清零，来处理过拟合
基本在0.26和0.27之间
'''
model = Sequential()
# 使用 dropout=0.2, recurrent_dropout=0.2, 随机将权重清零
# 来处理GRU的过度拟合
model.add(layers.GRU(32, dropout=0.2, recurrent_dropout=0.2,
                    input_shape=(None, float_data.shape[-1])))
model.add(layers.Dense(1))
model.compile(optimizer=RMSprop(), loss='mae')
#  为了看过拟合现象，把训练次数增加
history = model.fit_generator(train_gen, steps_per_epoch=500,
                             epochs = 40, validation_data=val_gen,
                             validation_steps = val_steps)

import matplotlib.pyplot as plt
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation loss')

plt.show()

