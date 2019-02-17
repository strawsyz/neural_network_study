import matplotlib.pyplot as plt
from keras.datasets import mnist
# 先引入用于训练神经网络的数据集
# train_images是用于训练系统的手写数字图片，train_labels是用于标志图片的信息，test_images是用于检测系统训练效果的图片，test_labels是test_images图片对应的数字标签
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# print(train_images.shape)
# print(train_labels)
# print(test_images.shape)
# print(test_labels)

def test():
    # 打印测试用的第一张图
    digit = test_images[0]
    plt.imshow(digit, cmap=plt.cm.binary)
    plt.show()

# 使用Keras迅速搭建一个有效识别图案的神经网络
from keras import models
from keras import layers
# layers表示的就是神经网络中的一个数据处理层。models.Sequential()
# 表示我们要把每一个数据处理层串联起来.神经网络的数据处理层之间的组合方式有多种，
# 串联是其中一种，也是最常用的一种。
network = models.Sequential()
# layers.Dense(…)就是构造一个数据处理层。
# input_shape(28*28,)表示当前处理层接收的数据格式必须是长和宽都是28的二维数组，
# 后面的“,“表示数组里面的每一个元素到底包含多少个数字都没有关系，
network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))  #relu其实一种张量操作，也就是max(x,0),
                                                                        # 假设x是[1, -1, -2], 那么作为relu操作后，x变为[1, 0, 0],
                                                                            # 也就是把x中小于0的元素全部变成0，大于0则保持不变
network.add(layers.Dense(10, activation='softmax'))
# 代码中的输入参数optimizer(y优化器), loss（损失函数）都对应着神经网络的相关组件
network.compile(optimizer='rmsprop', loss='categorical_crossentropy',
               metrics=['accuracy'])
# 其中reshape(60000, 28*28) 的意思是，train_images数组原来含有60000个元素，
# 每个元素是一个28行，28列的二维数组，
# 现在把每个二维数组转变为一个含有28*28个元素的一维数组。
train_images = train_images.reshape((60000, 28*28))
# 由于数字图案是一个灰度图，图片中每个像素点值的大小范围在0到255之间，
# 代码train_images.astype(“float32”)/255 把每个像素点的值从范围0-255转变为范围
# 在0-1之间的浮点值。
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32') / 255

# 接着我们把图片对应的标记也做一个更该，目前所有图片的数字图案对应的是0到9，
# 例如test_images[0]对应的是数字7的手写图案，那么其对应的标记test_labels[0]的值就是7，
# 我们需要把数值7变成一个含有10个元素的数组，然后在低7个元素设置为1，其他元素设置为0，
# 例如test_lables[0] 的值由7转变为数组[0,0,0,0,0,0,0,1,0,0,]
from keras.utils import to_categorical
print("before change:" ,test_labels[0])
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print("after change: ", test_labels[0])

# 把数据输入网络进行训练
# train_images是用于训练的手写数字图片，train_labels对应的是图片的标记，
# batch_size 的意思是，每次网络从输入的图片数组中随机选取128个作为一组进行计算，
# 每次计算的循环是五次，
network.fit(train_images, train_labels, epochs=5, batch_size = 128)

# 网络经过训练后，我们就可以把测试数据输入，检验网络学习后的图片识别效果了
test_loss, test_acc = network.evaluate(test_images, test_labels, verbose=1)
print(test_loss)
print('test_acc', test_acc)

# 使用神经网络识别数字
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
digit = test_images[1]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()
test_images = test_images.reshape((10000, 28*28))
res = network.predict(test_images)

for i in range(res[1].shape[0]):
    if (res[1][i] == 1):
        print("the number for the picture is : ", i)
        break