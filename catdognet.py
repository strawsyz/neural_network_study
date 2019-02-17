import os, shutil

'''识别狗和猫的网络
由于没有训练数据，所以只是搭建了网络结构'''
def init():
    '''用来移动图片'''
    #数据包被解压的路径
    original_dataset_dir = '/Users/chenyi/Documents/人工智能/all/train'
    #构造一个专门用于存储图片的路径
    base_dir = 'data/cats_and_dogs_small'
    os.makedirs(base_dir, exist_ok=True)
    #构造路径存储训练数据，校验数据以及测试数据
    train_dir = os.path.join(base_dir, 'train')
    os.makedirs(train_dir, exist_ok = True)
    test_dir = os.path.join(base_dir, 'test')
    os.makedirs(test_dir, exist_ok = True)
    validation_dir = os.path.join(base_dir, 'validation')
    os.makedirs(validation_dir, exist_ok = True)

    #构造专门存储猫图片的路径，用于训练网络
    train_cats_dir = os.path.join(train_dir, 'cats')
    os.makedirs(train_cats_dir, exist_ok = True)
    #构造存储狗图片路径，用于训练网络
    train_dogs_dir = os.path.join(train_dir, 'dogs')
    os.makedirs(train_dogs_dir, exist_ok = True)

    #构造存储猫图片的路径，用于校验网络
    validation_cats_dir = os.path.join(validation_dir, 'cats')
    os.makedirs(validation_cats_dir, exist_ok = True)
    #构造存储狗图片的路径，用于校验网络
    validation_dogs_dir = os.path.join(validation_dir, 'dogs')
    os.makedirs(validation_dogs_dir, exist_ok = True)

    #构造存储猫图片路径，用于测试网络
    test_cats_dir = os.path.join(test_dir, 'cats')
    os.makedirs(test_cats_dir, exist_ok = True)
    #构造存储狗图片路径，用于测试网络
    test_dogs_dir = os.path.join(test_dir, 'dogs')
    os.makedirs(test_dogs_dir, exist_ok = True)


    #把前1000张猫图片复制到训练路径
    fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(train_cats_dir, fname)
        shutil.copyfile(src, dst)

    #把接着的500张猫图片复制到校验路径
    fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(validation_cats_dir, fname)
        shutil.copyfile(src, dst)

    #把接着的500张猫图片复制到测试路径
    fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(test_cats_dir, fname)
        shutil.copyfile(src, dst)

    #把1000张狗图片复制到训练路径
    fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(train_dogs_dir, fname)
        shutil.copyfile(src, dst)

    #把接下500张狗图片复制到校验路径
    fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(validation_dogs_dir, fname)
        shutil.copyfile(src, dst)

    #把接下来500张狗图片复制到测试路径
    fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(test_dogs_dir, fname)
        shutil.copyfile(src, dst)

    print('total trainning cat images: ', len(os.listdir(train_cats_dir)))

    print('total training dog images', len(os.listdir(train_dogs_dir)))

    print('total validation cat images', len(os.listdir(validation_cats_dir)))

    print('total validation dogs images', len(os.listdir(validation_dogs_dir)))

    print('total test cat images:', len(os.listdir(test_cats_dir)))

    print('total test dog images:', len(os.listdir(test_dogs_dir)))


'''搭建神经网络架构'''
from keras import layers
from keras import models
from keras import optimizers

model = models.Sequential()
#输入图片大小是150*150 3表示图片像素用(R,G,B)表示
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4),
             metrics=['acc'])

model.summary()

'''将数据读入内存'''
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./ 255) #把像素点的值除以255，使之在0到1之间
test_datagen = ImageDataGenerator(rescale = 1. / 255)

#generator 实际上是将数据批量读入内存，使得代码能以for in 的方式去方便的访问
# class_mode的作用，由于我们只有猫狗两种图片，因此该标签值不是0就是1
# 由于train_dir路径下只有两个文件夹，它会为从这两个文件夹中读取的图片分别赋值0和1。
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150, 150),
                                                   batch_size=20,class_mode = 'binary')
validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                        target_size = (150, 150),
                                                       batch_size = 20,
                                                       class_mode = 'binary')
#calss_mode 让每张读入的图片对应一个标签值，我们上面一下子读入20张图片，因此还附带着一个数组(20, )
#标签数组的具体值没有设定，由我们后面去使用
for data_batch, labels_batch in train_generator:
    print('data batch shape: ', data_batch.shape)
    print('labels batch shape: ', labels_batch.shape)
    break

'''训练数据'''
# 网络模型支持直接将generator作为参数输入，由于我们构造的generator一次批量读入20张图片
# 总共有2000张图片，所以我们将参数steps_per_epoch = 100,
# 这样每次训练时，模型会用for…in… 在train_generator上循环100次，将所有2000张图片全部读取，
# 指定循环训练模型30次
history = model.fit_generator(train_generator, steps_per_epoch = 100,
                             epochs = 30, validation_data = validation_generator,
                             validation_steps = 50)
'''绘制型的训练准确率和校验准确率'''
model.save('cats_and_dogs_small_1.h5')
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

#绘制模型对训练数据和校验数据判断的准确率
plt.plot(epochs, acc, 'bo', label = 'trainning acc')
plt.plot(epochs, val_acc, 'b', label = 'validation acc')
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

'''为防止过度拟合，进行数据拓展'''
# rotation_range表示对图片进行旋转变化， width_shift 和 height_shift对图片的宽和高进行拉伸，
# shear_range指定裁剪变化的程度，zoom_range是对图片进行放大缩小，
# horizaontal_flip将图片在水平方向上翻转，fill_mode表示当图片进行变换后产生多余空间时，如何去填充
datagen = ImageDataGenerator(rotation_range = 40, width_shift_range = 0.2, height_shift_range = 0.2,
                            shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True, fill_mode = 'nearest')