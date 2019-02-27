from keras import layers
from keras import models

'''使用卷积网络训练手写识别
大概要运行两三个小时
在谷歌colab的训练结果，训练了1分钟左右
Epoch 1/5
60000/60000 [==============================] - 13s 222us/step - loss: 0.1737 - acc: 0.9456
Epoch 2/5
60000/60000 [==============================] - 10s 161us/step - loss: 0.0485 - acc: 0.9849
Epoch 3/5
60000/60000 [==============================] - 10s 161us/step - loss: 0.0334 - acc: 0.9894
Epoch 4/5
60000/60000 [==============================] - 10s 162us/step - loss: 0.0255 - acc: 0.9921
Epoch 5/5
60000/60000 [==============================] - 10s 160us/step - loss: 0.0199 - acc: 0.9940
10000/10000 [==============================] - 1s 102us/step
0.9884
第二次训练结果
Epoch 1/5
60000/60000 [==============================] - 10s 163us/step - loss: 0.1735 - acc: 0.9446
Epoch 2/5
60000/60000 [==============================] - 10s 158us/step - loss: 0.0493 - acc: 0.9851
Epoch 3/5
60000/60000 [==============================] - 10s 160us/step - loss: 0.0348 - acc: 0.9895
Epoch 4/5
60000/60000 [==============================] - 10s 165us/step - loss: 0.0263 - acc: 0.9918
Epoch 5/5
60000/60000 [==============================] - 10s 158us/step - loss: 0.0205 - acc: 0.9938
10000/10000 [==============================] - 1s 92us/step
0.9922
'''

model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.summary()

# 。卷积网络主要作用是对输入数据进行一系列运算加工，它输出的是中间形态的结果，
# 该结果不能直接用来做最终结果，要得到最终结果，我们需要为上面的卷积网络添加一层输出层，代码如下：
model.add(layers.Flatten())
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(10, activation='softmax'))
model.summary()

from keras.datasets import mnist
from keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs = 5, batch_size=64)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(test_acc)