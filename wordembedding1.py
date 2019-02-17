import numpy as np

'''使用Embedding来训练数据，可以准确率达到0.96以上，
运行要几分钟
还有一种方式是只取影评前20个字，精度应该会降低，还没有试过'''
def pre1():
    samples = ['The cat jump over the dog', 'The dog ate my homework']

    # 将每个单词放到哈希表中
    token_index = {}
    for sample in samples:
        # 将一个句子分解成多个单词
        for word in sample.split():
            if word not in token_index:
                token_index[word] = len(token_index) + 1

    # 设置句子的最大长度
    max_length = 10
    results = np.zeros((len(samples), max_length, max(token_index.values()) + 1))
    for i, samples in enumerate(samples):
        for j, word in list(enumerate(samples.split()))[: max_length]:
            index = token_index.get(word)
            results[i, j, index] = 1
            print("{0} ====>> {1}".format(word, results[i, j]))

def pre2():
    '''使用keras的话，the和The是作为同一个字符来判断的'''
    from keras.preprocessing.text import Tokenizer
    samples = ['The cat jump over the dog', 'The dog ate my homework']
    #只考虑给的句子样本中使用最频繁的前1000个单词
    tokenizer = Tokenizer(num_words = 1000)
    tokenizer.fit_on_texts(samples)
    #把句子分解成单词数组，每个单词用一个数字来表示
    sequences = tokenizer.texts_to_sequences(samples)
    print(sequences)
    one_hot_vecs = tokenizer.texts_to_matrix(samples, mode='binary')

    word_index = tokenizer.word_index
    print("当前总共有%s个不同单词"%len(word_index))
    # one_hot_vecs对应两个含有1000个元素的向量，第一个向量的第1，3，4，5个元素为1，
    # 其余全为0，第二个向量第1，2，6，7，8个元素为1，其余全为0.

'''准备影评数据'''
from keras.datasets import imdb
#  因为是频率前10000的，所以，data二维数组中最大的数字应该是10000也许是9999？先不管
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

def vectorize_sequence(sequences, dimension=10000):
    '''
    sequences 是包含所有评论的序列，一条评论对应到一个长度为10000的数组，因此我们要构建一个二维矩阵，
    矩阵的行对应于评论数量，列对应于长度为10000
    '''
    results = np.zeros((len(sequences), dimension))
    for i, sequences in enumerate(sequences):
        results[i, sequences] = 1.0
    return results
# print(train_data[0])  # [1, 14, 22, 16, 43, 530, 973, 1622, 1385, 65, 458, 4468, 66, 3941, ....}
# print(len(train_data[0]))  # 长度是句子里的单词个数
x_train = vectorize_sequence(train_data)
x_test = vectorize_sequence(test_data)

print(len(x_train[0]))

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

'''word embedding
用非零向量来表示每一个单词。one-hot-vector对单词进行编码有很多缺陷，
一是冗余过多，一大堆0，然后只有一个1，二是向量的维度过高，有多少个单词，向量就有多少维度
这会给计算带来很多麻烦，word-embedding把原来高维度的冗余向量转换为低纬度的，
信息量强的向量，转换后的向量，无论单词量多大，向量的维度一般只有256维到1024维。

单词向量化的一个关键目标是，意思相近的单词，他们对应的向量之间的距离要接近，
例如”good”,”fine”都表示“好”的意思，因此这两个单词对应的向量在空间上要比较接近的
，也就是说意思相近的单词，他们对应的向量在空间上的距离应该比较小。
'''
from keras.layers import Embedding
#Embedding对象接收两个参数，一个是单词量总数，另一个是单词向量的维度
# embedding_layer = Embedding(1000, 64)

from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential()
maxlen = 10000
#在网络中添加Embedding层，专门用于把单词转换成向量
model.add(Embedding(10000, 8, input_length=maxlen))

'''
我们给Embeding层输入长度不超过maxlen的单词向量，它为每个单词构造长度为8的向量
它会输出格式为(samples, maxlen, 8)的结果,然后我们把它转换为(samples, maxlen*8)的
二维格式
'''
model.add(Flatten())

#我们在顶部加一层只含有1个神经元的网络层，把Embedding层的输出结果对应成两个类别
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics = ['acc'])
model.summary()

history = model.fit(x_train, y_train, epochs = 10, batch_size = 32, validation_split=0.2)

'''画出图像'''
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# 绘制模型对训练数据和校验数据判断的准确率
plt.plot(epochs, acc, 'bo', label='training acc')
plt.plot(epochs, acc, 'b', label='validation acc')
plt.title('Trainning and validation accuary')
plt.legend()

plt.show()
plt.figure()

#绘制模型对训练数据和校验数据判断的错误率
plt.plot(epochs, loss, 'bo', label = 'Trainning loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
plt.title('Trainning and validation loss')
plt.legend()

plt.show()