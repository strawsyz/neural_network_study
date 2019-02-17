from keras.datasets import reuters
from keras.utils.np_utils import to_categorical

'''将新闻数据按照46个不同话题进行划分
属于典型的“单标签，多类别划分”的文本分类应用'''
# 加载数据
(train_data, train_label), (test_data, test_labels) = reuters.load_data(num_words=10000)
# 格式化数据
import numpy as np
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# 调用keras框架提供的接口一次性方便简单的完成
one_hot_train_labels = to_categorical(train_label)
one_hot_test_labels = to_categorical(test_labels)

# 划分测试和训练数据
x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

# from keras import models
# from keras import layers
# # 搭建神经网络
# model = models.Sequential()
# model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
# model.add(layers.Dense(64, activation='relu'))
# #当结果是输出多个分类的概率时，用softmax激活函数,它将为46个分类提供不同的可能性概率值
# model.add(layers.Dense(46, activation='softmax'))
#
# #对于输出多个分类结果，最好的损失函数是categorical_crossentropy
# model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
#
# history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512,
#                    validation_data=(x_val, y_val))
#
# # 画图展示训练过程
# import matplotlib.pyplot as plt
# loss = history.history['loss']
# val_loss = history.history['val_loss']
#
# epochs = range(1, len(loss) + 1)
#
# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
#
# # 用测试数据测试
# results = model.evaluate(x_test, one_hot_test_labels)
# print(results)  # [1.2092601955733124, 0.778717720444884]
#
# predictions = model.predict(x_test)
# print(predictions[0])
# print(np.sum(predictions[0]))
# print(np.argmax(predictions[0]))
# print(one_hot_test_labels[0])

'''下面是测试，中间层小于46时，对结果造成的影响'''
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(4, activation='relu'))
#当结果是输出多个分类的概率时，用softmax激活函数,它将为46个分类提供不同的可能性概率值
model.add(layers.Dense(46, activation='softmax'))

#对于输出多个分类结果，最好的损失函数是categorical_crossentropy
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(partial_x_train, partial_y_train, epochs=9, batch_size=512,
                   validation_data=(x_val, y_val))
results = model.evaluate(x_test, one_hot_test_labels)
print(results)