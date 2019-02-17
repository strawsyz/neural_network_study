import keras
from keras import models

'''使用tensorboard'''
def pre():
    callbacks_list = {
        # 停止训练流程，一旦网络对校验数据的判断率不再提升，
        # patience表示在两次循环间判断率没改进时就停止
        keras.callbacks.EarlyStopping(monitor='acc', patience=1),

    '''
        在每次训练循环结束时将当前参数存入文件tensorboard_model.h5,
        后两个参数表明当网络判断率没有提升时，不存储参数
        ''',
        keras.callbacks.ModelCheckpoint(filepath='tensorboard_model.h5',
                                        monitor='val_loss',
                                        save_best_only=True),
        '''
        如果网络对校验数据的判断率在10次训练循环内一直没有提升，
        下面回调将修改学习率
        ''',
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                          factor='0.1',
                                          patience=10)
    }

import keras;
from keras import layers
from keras.datasets import imdb
from keras.preprocessing import sequence

max_features = 2000
max_len = 500

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = max_features)
x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen = max_len)

model = keras.models.Sequential()
model.add(layers.Embedding(max_features, 128, input_length = max_len,
                          name = 'embed'))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(1))
model.summary()
model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy',
             metrics = ['acc'])

callbacks = [
    keras.callbacks.TensorBoard(log_dir='tensorboard_log_dir',
                                #每隔一个训练循环就用柱状图显示信息
                               histogram_freq = 1,
                               embeddings_freq = 1)
]

history = model.fit(x_train, y_train,
                   epochs = 20,
                   batch_size = 128,
                   validation_split = 0.2,
                   callbacks = callbacks)