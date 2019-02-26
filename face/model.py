from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD 
from keras import backend as K

img_rows, img_cols = 57, 47         # 输入图片的大小
nb_filters1, nb_filters2 = 20, 40   # 两层卷积核的数目（即输出的维度）


def cnn_model(lr=0.005,decay=1e-6,momentum=0.9):
    '''构建cnn网络结构'''
    model = Sequential()

    # 卷积层
    # K.image_data_format() 返回默认的图片维度顺序
    if K.image_data_format() == 'channels_first':
        model.add(Conv2D(nb_filters1, kernel_size=(3, 3), 
            input_shape = (1, img_rows, img_cols)))
    else:
        model.add(Conv2D(nb_filters1, kernel_size=(2, 2), 
            input_shape = (img_rows, img_cols, 1)))
    model.add(Activation('tanh'))  # 使用tanh激励函数
    # 池化层
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # 卷积层
    model.add(Conv2D(nb_filters2, kernel_size=(3, 3)))
    model.add(Activation('tanh'))  
    # 池化层
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # flatten层
    model.add(Flatten())  
    # 全连接层
    model.add(Dense(1000))       #Full connection
    model.add(Activation('tanh'))  
    model.add(Dropout(0.5))  
    # 全连接层，分类器
    model.add(Dense(40))
    model.add(Activation('softmax'))  # 使用softmax激励函数来分类

    # 设置梯度下降算法
    sgd = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)  
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    return model  