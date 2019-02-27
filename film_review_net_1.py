import numpy as np
from keras.datasets import imdb
from keras import models
from keras import layers
from keras import losses
from keras import metrics
from keras import optimizers




'''
使用深度神经网络
判断是正能量的影评还是负能量的影评


总结
1，将原始数据进行加工，使其变成数据向量以便输入网络。 
2，根据问题的性质选用不同的损失函数和激活函数，如果网络的目标是将数据区分成两类，那么损失函数最好选择binary_crossentropy，输出层的神经元如果选用sigmoid激活函数，那么它会给出数据属于哪一种类型的概率。 
3，选取适当的优化函数，几乎所有的优化函数都以梯度下降法为主，只不过在更新链路权重时，有些许变化。 
4，网络的训练不是越多越好，它容易产生“过度拟合”的问题，导致训练的越多，最终效果就越差，所以在训练时要密切关注网络对检验数据的判断准确率。'''

# 数据中的评论是用英语拟写的文本，我们需要对数据进行预处理，把文本变成数据结构后才能提交给网络进行分析。
# 我们当前下载的数据条目中，包含的已经不是原来的英文，而是对应每个英语单词在所有文本中的出现频率，
# 我们加载数据时，num_words=10000，表示数据只加载那些出现频率 前一万位 高的单词。
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
def preinfo():
    # label为1表示正能量，0表示负能量
    print(train_data[0])
    print(train_labels[0])

    #频率与单词的对应关系存储在哈希表word_index中,它的key对应的是单词，value对应的是单词的频率
    word_index = imdb.get_word_index()
    #我们要把表中的对应关系反转一下，变成key是频率，value是单词
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

    '''
    在train_data所包含的数值中，数值1，2，3对应的不是单词，而用来表示特殊含义，1表示“填充”，2表示”文本起始“，
    3表示”未知“，因此当我们从train_data中读到的数值是1，2，3时，我们要忽略它，从4开始才对应单词，如果数值是4，
    那么它表示频率出现最高的单词
    '''
    decoded_review = ' '.join([reverse_word_index.get(i-3, '?') for i in train_data[0]])
    print(decoded_review)

def vectorize_sequence(sequences, dimension=10000):
    '''
    sequences 是包含所有评论的序列，一条评论对应到一个长度为10000的数组，因此我们要构建一个二维矩阵，
    矩阵的行对应于评论数量，列对应于长度为10000
    '''
    results = np.zeros((len(sequences), dimension))
    for i, sequences in enumerate(sequences):
        results[i, sequences] = 1.0
    return results

x_train = vectorize_sequence(train_data)
x_test = vectorize_sequence(test_data)

print(x_train[0])

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')


'''下面是构建一个神经网络，进行训练'''
# model = models.Sequential()
# #构建第一层和第二层网络，第一层有10000个节点，第二层有16个节点
# #Dense的意思是，第一层每个节点都与第二层的所有节点相连接
# #relu 对应的函数是relu(x) = max(0, x)
# model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
# #第三层有16个神经元，第二层每个节点与第三层每个节点都相互连接
# model.add(layers.Dense(16, activation='relu'))
# #第四层只有一个节点，输出一个0-1之间的概率值，用于标记文本含有正能量的可能性。
# model.add(layers.Dense(1, activation='sigmoid'))
#
# # optimizer参数指定的是如何优化链路权重，事实上各种优化方法跟我们前面讲的梯度下降法差不多，
# # 只不过在存在一些微小的变化，
# # 特别是在更新链路权值时，会做一些改动，但算法主体还是梯度下降法。
# # 当我们的网络用来将数据区分成两种类型时，损失函数最好使用binary_crossentroy
# model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss='binary_crossentropy',
#               metrics=['accuracy'])
# # 训练数据共25000条
# # 前10000条作为校验数据
# x_val = x_train[:10000]
# # 剩余15000条数据作为训练数据
# partial_x_train = x_train[10000:]
#
# y_val = y_train[:10000]
# partial_y_train = y_train[10000:]
# # fit函数返回的history对象记录了训练过程中，网络的相关数据，通过分析这些数据，
# # 我们可以了解网络是如何改进自身的，它是一个哈希表
# history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512,
#                     validation_data = (x_val, y_val))
#
# history_dict = history.history
# # print(history_dict.keys())  结果是这样：dict_keys(['val_loss', 'val_acc', 'loss', 'acc'])
#
# # 网络对训练数据识别的精确度和对校验数据识别的精确度，画出来的成像图
# import matplotlib.pyplot as plt
#
# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
#
# epochs = range(1, len(acc) + 1)
# #绘制训练数据识别准确度曲线,点图
# plt.plot(epochs, loss, 'bo', label='Trainning loss')
# #绘制校验数据识别的准确度曲线，折线图
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Trainning and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()


'''下面是另一个主函数
训练网络之后进行测试'''
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
# 训练网络
model.fit(x_train, y_train, epochs=3, batch_size=512)
# 用测试数据评估
results = model.evaluate(x_test, y_test)
print(results)  # [0.295116818857193, 0.88348] 第二值 表示准确值
# 看看网络对每一条测试文本得出它是正能量的几率：
temp = model.predict(x_test)
print(temp)

