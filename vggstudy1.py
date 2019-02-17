from keras.applications import VGG16
# weight参数告诉程序将网络的卷积层和max pooling层对应的参数传递过来，
# 并将它们初始化成对应的网络层次。include_top表示是否也要把Flatten()后面的网络层也下载过来，
# VGG16对应的这层网络用来将图片划分到1000个不同类别中，由于我们只用来区分猫狗两个类别，
# 因此我们去掉它这一层。input_shape告诉网络，我们输入图片的大小是150*150像素，
# 每个像素由[R, G, B]三个值表示
conv_base = VGG16(weights = 'imagenet', include_top = False, input_shape=(150, 150, 3))

conv_base.summary()

import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

base_dir = '/Users/chenyi/Documents/人工智能/all/cats_and_dogs_small'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

datagen = ImageDataGenerator(rescale = 1. / 255)
batch_size = 20

def extract_features(directory, sample_count):
    '''用于特征提取'''
    features = np.zeros(shape = (sample_count, 4, 4, 512))
    labels = np.zeros(shape = (sample_count))
    generator = datagen.flow_from_directory(directory, target_size = (150, 150),
                                            batch_size = batch_size,
                                            class_mode = 'binary')
    i = 0
    for inputs_batch, labels_batch in generator:
        #把图片输入VGG16卷积层，让它把图片信息抽取出来
        features_batch = conv_base.predict(inputs_batch)
        #feature_batch 是 4*4*512结构
        features[i * batch_size : (i + 1)*batch_size] = features_batch
        labels[i * batch_size : (i+1)*batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count :
            #for in 在generator上的循环是无止境的，因此我们必须主动break掉
            break
    return features, labels

#features 的数据格式为(samples, 4, 4, 512)，samples是样本的数量
train_features, train_labels = extract_features(train_dir, 2000)
validation_features, validation_labels = extract_features(validation_dir, 1000)
test_features, test_labels = extract_features(test_dir, 1000)

train_features = np.reshape(train_features, (2000, 4 * 4 * 512))
validation_features = np.reshape(validation_features, (1000, 4 * 4 * 512))
test_features = np.reshape(test_features, (1000, 4 * 4* 512))

from keras import models
from keras import layers
from keras import optimizers

#构造我们自己的网络层对输出数据进行分类
model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim = 4 * 4 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation = 'sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr = 2e-5), loss = 'binary_crossentropy', metrics = ['acc'])
history = model.fit(train_features, train_labels, epochs = 30, batch_size = 20,
                    validation_data = (validation_features, validation_labels))

'''画出结果'''
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label = 'Train_acc')
plt.plot(epochs, val_acc, 'b', label = 'Validation acc')
plt.title('Trainning and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


'''参数调优的步骤如下：

1，将我们自己的网络层添加到VGG16的卷积层之上。 
2，固定VGG16的卷积层保持不变。 
3，用数据训练我们自己添加的网络层 
4，将VGG16的卷积层最高两层放开 
5，用数据同时训练放开的那两层卷积层和我们自己添加的网络层
'''

model = models.Sequential()
#将VGG16的卷积层直接添加到我们的网络
model.add(conv_base)
#添加我们自己的网络层
model.add(layers.Flatten())
model.add(layers.Dense(256, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))
model.summary()

conv_base.trainable = True
set_trainable = False
#一旦读取到'block5_conv1'时，意味着来到卷积网络的最高三层
#可以使用conv_base.summary()来查看卷积层的信息
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        #当trainable == True 意味着该网络层可以更改，要不然该网络层会被冻结，不能修改
        layer.trainable = True
    else:
        layer.trainable = False

#把图片数据读取进来
test_datagen = ImageDataGenerator(rescale = 1. / 255)
train_generator = test_datagen.flow_from_directory(train_dir, target_size = (150, 150), batch_size = 20,
                                                   class_mode = 'binary')
validation_generator = test_datagen.flow_from_directory(validation_dir, target_size = (150,150),
                                                       batch_size = 20,
                                                       class_mode = 'binary')
model.compile(loss = 'binary_crossentropy', optimizer = optimizers.RMSprop(2e-5),
             metrics = ['acc'])

history = model.fit_generator(train_generator, steps_per_epoch = 100, epochs = 30,
                              validation_data = validation_generator,
                              validation_steps = 50)