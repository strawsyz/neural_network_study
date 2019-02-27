from keras.layers import Dense, Input
from keras.layers import Conv2D, Flatten
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
from keras.datasets import mnist
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt

'''
使用自动编解码器网络实现图片噪音去除
训练60000个样本应该要几天时间，使用colab只要20分钟就能训练好
实验效果：
Train on 60000 samples, validate on 10000 samples
Epoch 1/10
60000/60000 [==============================] - 126s 2ms/step - loss: 0.0355 - val_loss: 0.0202
Epoch 2/10
60000/60000 [==============================] - 126s 2ms/step - loss: 0.0193 - val_loss: 0.0181
Epoch 3/10
60000/60000 [==============================] - 126s 2ms/step - loss: 0.0177 - val_loss: 0.0171
Epoch 4/10
60000/60000 [==============================] - 125s 2ms/step - loss: 0.0169 - val_loss: 0.0167
Epoch 5/10
60000/60000 [==============================] - 126s 2ms/step - loss: 0.0164 - val_loss: 0.0163
Epoch 6/10
60000/60000 [==============================] - 126s 2ms/step - loss: 0.0160 - val_loss: 0.0161
Epoch 7/10
60000/60000 [==============================] - 127s 2ms/step - loss: 0.0157 - val_loss: 0.0159
Epoch 8/10
60000/60000 [==============================] - 127s 2ms/step - loss: 0.0154 - val_loss: 0.0158
Epoch 9/10
60000/60000 [==============================] - 126s 2ms/step - loss: 0.0152 - val_loss: 0.0157
Epoch 10/10
60000/60000 [==============================] - 126s 2ms/step - loss: 0.0151 - val_loss: 0.0158
'''

#加载手写数字图片数据
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train[:1000]
x_test = x_test[:1000]
image_size = x_train.shape[1]  # 图片宽的像素个数
latent_dim = 16  # 解码器输出的向量长度
input_shape = (image_size, image_size, 1)
batch_size = 32
kernel_size = 3
layer_filters = [32, 64]
print(x_train.shape)
#把图片大小统一转换成28*28,并把像素点值都转换为[0,1]之间
x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
print(x_train.shape)  #(1000, 28, 28, 1)


# 使用高斯分布产生图片噪音
np.random.seed(1337)
# 使用高斯分布函数生成随机数,均值0.5，方差0.5
noise = np.random.normal(loc=0.5, scale=0.5, size=x_train.shape)
x_train_noisy = x_train + noise

noise = np.random.normal(loc=0.5, scale=0.5, size=x_test.shape)
x_test_noisy = x_test + noise

# 把像素点取值范围转换到[0,1]间,小于0的变成0，大于1的变成1
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)
# 上面的代码先使用高斯函数产生随机数，然后加到像素点上从而形成图片噪音。

# 构造编码器实现图片去噪
inputs = Input(shape=input_shape, name='encoder_input')
x = inputs
for filters in layer_filters:
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=2,
               activation='relu', padding='same')(x)

shape = K.int_shape(x)  # 返回张量或变量的尺寸，以 int或None 项的元组 形式。
x = Flatten()(x)
latent = Dense(latent_dim, name='latent_vector')(x)  # 输出层是latent_dim个元素
encoder = Model(inputs, latent, name='encoder')

# 构造解码器
latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
x = Dense(shape[1] * shape[2] * shape[3])(latent_inputs)
x = Reshape((shape[1], shape[2], shape[3]))(x)  # Flatten层的相反动作

for filters in layer_filters[::-1]:  # layer_filters[::-1]将layer_filters[::-1]倒着排序
    x = Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=2,
                        activation='relu', padding='same')(x)

outputs = Conv2DTranspose(filters=1, kernel_size=kernel_size, padding='same',
                          activation='sigmoid', name='decoder_output')(x)
decoder = Model(latent_inputs, outputs, name='decoder')

# 将编码器encoder和解码器decoder前后相连
autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
autoencoder.compile(loss='mse', optimizer='adam')
# 输入数据是有噪音图片，对应结果是无噪音图片
autoencoder.fit(x_train_noisy, x_train, validation_data=(x_test_noisy, x_test),
                epochs=10, batch_size=batch_size)

# 将测试图片输入网络查看训练效果
x_decoded = autoencoder.predict(x_test_noisy)

# 把输出结果转化成图片
rows, cols = 3, 9
num = rows * cols
imgs = np.concatenate([x_test[:num], x_test_noisy[:num],
                      x_decoded[:num]])
print(imgs.shape)
imgs = imgs.reshape((rows * 3, cols, image_size, image_size))
imgs = np.vstack(np.split(imgs, rows, axis=1))
imgs = imgs.reshape((rows * 3, -1, image_size, image_size))
imgs = np.vstack([np.hstack(i) for i in imgs])
imgs = (imgs * 255).astype(np.uint8)
plt.figure()
plt.axis('off')
plt.title('Original images: top rows'
          'Corrupted images: middle rows'
         'Denoised Input: third rows')
plt.imshow(imgs, interpolation='none', cmap='gray')
plt.show()