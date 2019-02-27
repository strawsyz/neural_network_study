import keras
import numpy as np

'''自动写文章的网络，输入60个字符，然后自动生成后面的400个字符
原理：训练一个LSRM网络，更具前面的60个字符，预测下一个字符，
最新的60个字符再作为下一个循序的输入，循环400次，即可生成400个字符'''

# 加载数据
path = keras.utils.get_file('nietzche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
text = open(path).read().lower()  # 全都变成小写字符
# print('length is ', len(text))  # 600893个字符，共有57个不同的元素

maxlen = 60
step = 3
setences = []

# next_chars 对应下一个字符，便于训练网络
next_chars = []

# 告诉网络前60字符，预测61个字符
for i in range(0, len(text) - maxlen, step):
    setences.append(text[i:i+maxlen])  # maxlen个单词作为一个句子
    next_chars.append(text[i + maxlen])

print('Number of sentences is ', len(setences))  #  200278

chars = sorted(list(set(text)))
len_chars = len(chars)
# print('Unique characters is ', len(chars))  # 字符的个数 58
# print(chars)  # ['\n', ' ', '!', '"', "'", '(', ')', ',', '-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '=', '?', '[', ']', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '忙', '毛', '盲', '脝', '茅']
# 为每个字符做编号
char_indices = dict((char, chars.index(char)) for char in chars)
print('Vectorization....')  # 矢量化
'''
整个文本中不同字符的个数为chars, 对于当个字符我们对他进行one-hot编码，
也就是构造一个含有chars个元素的向量，根据字符对于的编号，
我们把向量的对应元素设置为1，
一个句子含有maxlen个字符，因此一行句子对应一个二维句子(maxlen, chars)，
矩阵的行数是maxlen，列数是len(chars)
'''
# 	np.bool布尔型数据类型（True 或者 False）
x = np.zeros((len(setences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(setences), len(chars)), dtype=np.bool)
for i, setence in enumerate(setences):
    for t, char in enumerate(setence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

from keras import layers
from keras.models import Sequential
# Sequential ：序贯模型是多个网络层的线性堆叠。
model = Sequential()
model.add(layers.LSTM(128, input_shape=(maxlen, len_chars)))
model.add(layers.Dense(len_chars, activation='softmax'))
optimizer = keras.optimizers.RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
model.summary()

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds/np.sum(exp_preds)
    '''
     由于preds含有57个元素，每个元素表示对应字符出现的概率，我们可以把这57个元素看成一个含有57面的骰子，
     骰子第i面出现的概率由preds[i]决定，然后我们模拟丢一次这个57面骰子，看看出现哪一面，这一面对应的字符作为
     网络预测的下一个字符
     '''
    probas = np.random.multinomial(1, preds, 1)  # 多项分布
    # np.argmax 返回最大数的索引
    return np.argmax(probas)

import random
import sys

for epoch in range(1,60):
    print('epoch:',epoch)
    # 训练一次
    model.fit(x, y, batch_size=128, epochs=1)
    # 随机从文本中选取开始的输入网络的字符串
    start_index = random.randint(0, len(text) - maxlen - 1)
    generated_text = text[start_index: start_index+maxlen]
    print('---Generating with seed:"' + generated_text + '"')

    for temperature in [0.2, 0.5, 1.0, 1.2]:
        print('---temperature:', temperature)
        sys.stdout.write(generated_text)
        '''
          根据原文，我们让网络创作接着原文后面的400个字符组合成的段子
        '''
        for i in range(400):
            sampled = np.zeros((1, maxlen, len_chars))
            for t, char in enumerate(generated_text):
                sampled[0, t, char_indices[char]] = 1.

            # 让网络根据当前输入的字符预测下一个字符
            preds = model.predict(sampled, verbose=0)[0]
            # print(preds)
            next_index = sample(preds, temperature)
            next_char = chars[next_index]

            generated_text += next_char  # 添加生成的字符
            generated_text = generated_text[1:]  # 截取下一次训练的字符

            sys.stdout.write(next_char)
            sys.stdout.flush()
