import numpy as np
import pandas as pd
from scipy.misc import imread
import os
import pylab
from keras.models import Sequential
from keras.layers import Dense, Flatten, Reshape, InputLayer
from keras.regularizers import L1L2  # regularizers 正则
import matplotlib.pyplot as plt

'''
使用gan生成手写数字
经过千辛万苦的调bug终于可以训练了！！
训练10次没什么效果，训练100次终于有点效果了
训练100次大概要几十分钟'''

# 设置路径
root_dir = os.path.abspath('D:/alab/net_data/')
data_dir = os.path.join(root_dir, 'numberdata')

# 加载数据
train = pd.read_csv(os.path.join(data_dir, 'Train', 'train.csv'))
test = pd.read_csv(os.path.join(data_dir, 'Train', 'test.csv'))
# 处理数据
temp = []
for img_name in train.filename:
    image_path = os.path.join(data_dir, "Train", 'train', img_name)
    img = imread(image_path, flatten=True)
    img = img.astype('float32')
    temp.append(img)

train_x = np.stack(temp)
train_x = train_x/255
def pre():
    '''绘制一张图'''
    # 设置一个seed，使确定性的随机可复现
    seed = 128
    rng = np.random.RandomState(seed)
    img_name = rng.choice(train.filename)
    filepath = os.path.join(data_dir, 'Train', 'train', img_name)
    img = imread(filepath, flatten=True)
    pylab.imshow(img, cmap='gray')
    pylab.axis('on')  # 显示坐标轴
    pylab.show()

def gan():
    # define variables
    # 初始化一些参数
    g_input_shape = 100  # 生成器输入层节点数
    d_input_shape = (28, 28)  # 辨别器输入层节点数
    hidden_1_num_units = 500
    hidden_2_num_units = 500
    g_output_num_units = 784  # 生成器输出层节点数28*28
    d_output_num_units = 1  # 辨别器输出层节点数1个，辨别是否是真实图片
    epochs = 100
    batch_size = 128

    # 定义生成器，用于生成图片
    model_g = Sequential([
    Dense(units=hidden_1_num_units, input_dim=g_input_shape, activation='relu',
          kernel_regularizer=L1L2(1e-5, 1e-5)),
    Dense(units=hidden_2_num_units, activation='relu', kernel_regularizer=L1L2(1E-5, 1E-5)),
    Dense(units=g_output_num_units, activation='sigmoid', kernel_regularizer=L1L2(1E-5, 1E-5)),
    Reshape(d_input_shape)])


    # 定义分辨器，用于辨别图片
    model_d = Sequential([
    InputLayer(input_shape=d_input_shape),
    Flatten(),
    Dense(units=hidden_1_num_units,activation='relu',kernel_regularizer=L1L2(1E-5, 1E-5)),
    Dense(units=hidden_2_num_units,activation='relu',kernel_regularizer=L1L2(1E-5, 1E-5)),
    Dense(units=d_output_num_units, activation='sigmoid', kernel_regularizer=L1L2(1E-5, 1E-5))
    ])
    # model_g.summary()
    # model_d.summary()

    from keras_adversarial import AdversarialModel, simple_gan, gan_targets
    from keras_adversarial import AdversarialOptimizerSimultaneous, normal_latent_sampling
    # 开始训练gan网络
    gan = simple_gan(model_g, model_d, normal_latent_sampling((100,)))
    # gan.summary()
    # 在keras2.2.x版本中，下面的代码会报错，keras2.1.2中不会
    model = AdversarialModel(base_model=gan, player_params=[model_g.trainable_weights, model_d.trainable_weights])
    model.adversarial_compile(adversarial_optimizer=AdversarialOptimizerSimultaneous(),
                              player_optimizers=['adam', 'adam'], loss='binary_crossentropy')
    # 使用训练数据进行训练
    # 把keras_adversarial clone到了本地，然后替换掉了pip安装的keras_adversarial
    # 解决了这个报错AttributeError: 'AdversarialModel' object has no attribute '_feed_output_shapes'
    history = model.fit(x=train_x, y=gan_targets(train_x.shape[0]), epochs=epochs, batch_size=batch_size)
    # 保存为h5文件
    model_g.save_weights('gan1_g.h5')
    model_d.save_weights('gan1_d.h5')
    model.save_weights('gan1.h5')

    # 绘制训练结果的loss
    plt.plot(history.history['player_0_loss'],label='player_0_loss')
    plt.plot(history.history['player_1_loss'],label='player_1_loss')
    plt.plot(history.history['loss'],label='loss')
    plt.show()

    # 训练之后100次之后生成的图像
    # 随机生成10组数据，生成10张图像
    zsample = np.random.normal(size=(10, 100))
    pred = model_g.predict(zsample)
    print(pred.shape)  # (10,28,28)
    for i in range(pred.shape[0]):
        plt.imshow(pred[i, :], cmap='gray')
        plt.show()

