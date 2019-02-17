import os

'''使用预先训练好的单词向量识别影评的正负能量'''
'''验证数据正确率有0.7左右
原因这个问题的原因主要就是我们的训练数据量太少，只有两万条，因此没能重复发挥预先训练向量的作用，
我们现在使用的单词向量维度很大，达到了100，
但维度变大，但是训练数据量没有等量级的增加时，过度拟合就出现了。'''
imdb_dir = 'D:/alab/net_data/aclImdb_v1/aclImdb/'

train_dir = os.path.join(imdb_dir, 'train')  # 训练用数据的路径
labels = []
texts = []

for label_type in ['neg', 'pos']:
    '''
    遍历两个文件夹下的文本，将文本里面的单词连接成一个大字符串，从neg目录下读出的文本赋予一个标签0，
    从pos文件夹下读出的文本赋予标签1
    '''
    dir_name = os.path.join(train_dir, label_type)
    for file_name in os.listdir(dir_name):  # 获得文件夹下的所有文件
        if file_name[-4:] == '.txt':  # 如果是txt文件
            file = open(os.path.join(dir_name, file_name), encoding='utf-8')
            texts.append(file.read())
            file.close()
            if label_type == 'neg':  # 如果是消极的影评
                labels.append(0)
            else:
                labels.append(1)  # 如果是积极的影评

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

maxlen = 100  # 最多读取影评的前100个单词
training_samples = 20000  # 前20000个影评作为训练数据
validation_samples = 2500 #用2500个影评作为校验数据
max_words = 10000  #只考虑出现频率最高的10000个单词


#下面代码将单词转换为one-hot-vector
tokenizer = Tokenizer(num_words = max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('总共有 %s 个不同的单词' % len(word_index))  # 总共有 88582 个不同的单词
data = pad_sequences(sequences, maxlen=maxlen)  # 只截取100个字，如果不够就用0填充

labels = np.asarray(labels)
print("数据向量的格式为：", data.shape)  # 数据向量的格式为： (25000, 100)
print("标签向量的格式为：", labels.shape)  # 标签向量的格式为： (25000,)

'''
将数据分成训练集合校验集，同时把数据打散，让正能量影评和负能量影评随机出现
'''
indices = np.arange(data.shape[0])  # data.shape[0] 是 25000
np.random.shuffle(indices)  # 打乱顺序， indices是一个一维数组
data = data[indices]
labels = labels[indices]

x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]

glove_dir = "D:/alab/net_data/glove.6B"  # 存放预先训练好的单词向量数据的文件夹
embedding_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'), encoding='utf-8')  # 该文件用100个向量代表一个单词
for line in f:
    #依照空格将一条数据分解成数组
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embedding_index[word] = coefs
f.close()
print("总共有 %s 个单词向量."%len(embedding_index))  # 总共有 400000 个单词向量.

'''由此可见数据量还是不小的。我们把加载进来的四十万条单词向量集合在一起形成一个矩阵，
我们从影评中抽取出每个单词，并在四十万条单词向量中找到对应单词的向量，'''
# 由于影评中的单词最多10000个，于是我们就能形成维度为(10000, 100)的二维矩阵
embedding_dim = 100  # 100就是单词向量的元素个数
embedding_matrix = np.zeros((max_words, embedding_dim))  # 10000， 100
# print(type(word_index))  # <class 'dict'>
# print(type(word_index.items()))  # <class 'dict_items'>
# print(word_index.items()) ([单词],[index]),([单词],[index]),([单词],[index]),([单词],[index])
# 下面的embedding_matrix就是一个Emdedding层
for word, i in word_index.items():
    embedding_vector = embedding_index.get(word)  # 根据单词获得单词向量
    if i < max_words:
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector


from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.layers[0].set_weights([embedding_matrix])  # 把之前设置的Embedding层放入神经网络中
model.layers[0].trainable = False  # 防止破坏之前训练好的单词向量

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(x_val, y_val))
model.save_weights('pre_trained_glove_model.h5')

'''绘制训练结果如下'''
import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

'''用测试数据进行测试'''
test_dir = os.path.join(imdb_dir, 'test')
labels = []
texts = []

for label_type in ['neg', 'pos']:
    dir_name = os.path.join(test_dir, label_type)
    for fname in sorted(os.listdir(dir_name)):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname), encoding='utf-8')
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)

sequences = tokenizer.texts_to_sequences(texts)
x_test = pad_sequences(sequences, maxlen=maxlen)
y_test = np.asarray(labels)

result = model.evaluate(x_test, y_test)
print(result)