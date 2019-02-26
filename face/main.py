import numpy as np
from PIL import Image
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD  
from keras.utils import np_utils
from keras import backend as K

import train
import test
import model
import load_data

'''人脸识别的示例
使用olivettifaces.gif'''


epochs = 35          # 训练的轮数
img_rows, img_cols = 57, 47         # 输入图片的大小

input_path = 'olivettifaces.gif'
if __name__ == '__main__':  
    # 加载数据
    (X_train, y_train), (X_val, y_val),(X_test, y_test) = load_data.load_data(input_path)
    
    if K.image_data_format() == 'channels_first':    # 1为图像像素深度
        X_train = X_train.reshape(X_train.shape[0],1,img_rows,img_cols)
        X_val = X_val.reshape(X_val.shape[0], 1, img_rows, img_cols)  
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)  
        input_shape = (1, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)  
        X_val = X_val.reshape(X_val.shape[0], img_rows, img_cols, 1)  
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)  
        input_shape = (img_rows, img_cols, 1)
    
    print('X_train shape:', X_train.shape)
    # np_utils.to_categorical将整型标签转为onehot。
    # 在这里将向量转成了矩阵
    Y_train = np_utils.to_categorical(y_train, 40)
    Y_val = np_utils.to_categorical(y_val, 40)
    Y_test = np_utils.to_categorical(y_test, 40)

    model = model.cnn_model()
    # 训练模型
    train.train_model(model, X_train, Y_train, X_val, Y_val, epochs)
    # 测试模型
    score = test.test_model(model, X_test, Y_test)
    print(score)
    # 加载训练好的模型
    model.load_weights('model_weights.h5')
    # 计算预测的类别
    classes = model.predict_classes(X_test, verbose=0)  
    # 计算正确率
    test_accuracy = np.mean(np.equal(y_test, classes))
    print("last accuarcy:", test_accuracy)
    error_num = 0
    for i in range(0,40):
        if y_test[i] != classes[i]:
            error_num += 1
            print(y_test[i], '被错误分成', classes[i]);
    print("共有" + str(error_num) + "张图片被识别错了")