#加载intception卷积网络层
from keras.applications import inception_v3
from keras import backend as K
from keras.preprocessing import image
import numpy as np

'''
风格移植网络
改了一些会报错的代码
但是，输出的图片总是绿绿的，
原版代码里的使用scipy保存图片，由于会报错
所以改成了opencv，还不清楚为什么报错
'''
# 把inceptionV3网络层的参数从网上读下来，并构建相应的网络层模型
K.set_learning_phase(0)
model = inception_v3.InceptionV3(weights='imagenet', include_top=True)
from keras.models import load_model
#我们前几节在训练网络后，曾经以下面名字把训练后的网络存储起来，现在我们重新将它加载
# model = load_model('D:/alab/net_data/inception_v3_weights_tf_dim_ordering_tf_kernels.h5')
# model.summary()

# 对图片进行预处理
def preprocess_image(img_path):
    img = image.load_img(img_path)
    img = image.img_to_array(img)
    # print(img.shape)  #(404, 540, 3)
    img = np.expand_dims(img, axis=0)  # 沿着某个轴增加值
    # print(img.shape)  # (1,404, 540, 3)
    img = inception_v3.preprocess_input(img)  # 向inception_v3网络中输入 图片
    return img
import keras.applications.imagenet_utils as utils
def pre():
    img = preprocess_image('data/flower.png')
    preds = model.predict(img)
    #  解析测试结果然后输出
    print(utils.decode_predictions(preds))  # 这个格式[[('n11939491', 'daisy', 0.026205944), ('n03930313', 'picket_fence', 0.023571707),
    # ('n03991062', 'pot', 0.022323348), ('n13044778', 'earthstar', 0.016001847), ('n03457902', 'greenhouse', 0.014344326)]]
    for n, label , prob in utils.decode_predictions(preds)[0]:  #
        print(label, prob)
    #     daisy 0.026205944
    # picket_fence 0.023571707
    # pot 0.022323348
    # earthstar 0.016001847
    # greenhouse 0.014344326

    # 我们使用下面代码看看inceptionV3网络层结构：
    print(model.summary())


# 定义要刺激的网络哪一层，获得那一层的output
def get_layer_to_stimulate(model, layer_num):
    # 选中的网络层名字
    layer = "activation_" + str(layer_num)
    activation = model.get_layer(layer).output
    return activation


def define_stimulation(activation):
    '''
    假设网络层的输出是x1,x2...xn
    刺激函数的定义为（x1^2 + x2^2 + .... xn^2) / (x1 * x2 .... *xn)
    '''
    # 先把对应网络层的输出元素相乘
    # K.shape返回张量或变量的符号尺寸。
    # keras.backend.cast(x, dtype)
    # 将张量转换到不同的 dtype 并返回。
    # 你可以转换一个 Keras 变量，但它仍然返回一个 Keras 张量。
    # keras.backend.prod(x, axis=None, keepdims=False)
    # 在某一指定轴，计算张量中的值的乘积。
    # 参数
    # x: 张量或变量。
    # axis: 一个整数需要计算乘积的轴。
    # keepdims: 布尔值，是否保留原尺寸。 如果 keepdims 为 False，则张量的秩减 1。 如果 keepdims 为 True，缩小的维度保留为长度 1。
    # 返回
    # x 的元素的乘积的张量。
    # 整句话就是把所有元素相乘
    scaling = K.prod(K.cast(K.shape(activation), 'float'))
    # 2:-2作用是排除图像边界点对应的网络层输出
    # square
    # keras.backend.square(x)
    # 元素级的平方操作。
    # 参数
    # x: 张量或变量。
    # 返回
    # 一个张量。
    # keras.backend.sum(x, axis=None, keepdims=False)
    # 计算张量在某一指定轴的和。
    # 参数
    # x: 张量或变量。
    # axis: 一个整数，需要加和的轴。
    # keepdims: 布尔值，是否保留原尺寸。 如果 keepdims 为 False，则张量的秩减 1。 如果 keepdims 为 True，缩小的维度保留为长度 1。
    # 返回
    # x 的和的张量。
    stimulate = K.sum(K.square(activation[:, 2:-2, 2:-2, :])) / scaling
    return stimulate


# 输入图片像素点
dream = model.input
# 刺激第41层
activation = get_layer_to_stimulate(model, 41)
# 对每个像素点求偏导，实现刺激函数最大化
stimulate = define_stimulation(activation)
# keras.backend.gradients(loss, variables)
# 返回 variables 在 loss 上的梯度。
# 参数
# loss: 需要最小化的标量张量。
# variables: 变量列表。
# 返回
# 一个梯度张量。
grads = K.gradients(stimulate, dream)[0]
print(grads)  # Tensor("gradients/conv2d_1/convolution_grad/Conv2DBackpropInput:0", shape=(?, ?, ?, 3), dtype=float32)
# 对每个偏导数做正规化处理
# keras.backend.mean(x, axis=None, keepdims=False)
# 张量在某一指定轴的均值。
# 参数
# x: A tensor or variable.
# axis: 整数或列表。需要计算均值的轴。
# keepdims: 布尔值，是否保留原尺寸。 如果 keepdims 为 False，则 axis 中每一项的张量秩减 1。 如果 keepdims 为 True，则缩小的维度保留为长度 1。
# 返回
# x 元素的均值的张量
# maximum
# keras.backend.maximum(x, y)
# 逐个元素比对两个张量的最大值。
# 参数
# x: 张量或变量。
# y: 张量或变量。
# 返回
# 一个张量。
grads /= K.maximum(K.mean(K.abs(grads)), 1e-7)  # 1e-7表示0.1的7次方
# function
# keras.backend.function(inputs, outputs, updates=None)
# 实例化 Keras 函数。
# 参数
# inputs: 占位符张量列表。
# outputs: 输出张量列表。
# updates: 更新操作列表。
# **kwargs: 需要传递给 tf.Session.run 的参数。
# 返回
# 输出值为 Numpy 数组。
iterate_grad_ac_step = K.function([dream], [stimulate, grads])
# 上面代码定义了刺激函数，要刺激的网络层，接下来我们对图片进行缩放等相关操作：

import scipy


# 把二维数组转换为图片格式
def deprocess_image(x):
    # keras.backend.image_data_format()
    # 返回默认图像数据格式约定 ('channels_first' 或 'channels_last')。
    # 返回
    # 一个字符串，'channels_first' 或 'channels_last'
    # 例子
    # >>> keras.backend.image_data_format()
    # 'channels_first'
    if K.image_data_format() == 'channel_first':  # "channels_first”或“channels_last”，则代表数据的通道维的位置
        # channels_first”应将数据组织为（3,128,128,128），而“channels_last”应将数据组织为（128,128,128,3）
        x = x.reshape((3, x.shape[2], x.shape[3]))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((x.shape[1], x.shape[2], 3))

    x /= 2.
    x += 0.5
    x *= 255
    x = np.clip(x, 0, 255).astype('int8')
    return x


def resize_img(img, size):
    img = np.copy(img)
    factors = (1, float(size[0]) / img.shape[1], float(size[1]) / img.shape[2], 1)
    return scipy.ndimage.zoom(img, factors, order=1)
import cv2

def save_img(img, fname):
    pil_img = deprocess_image(np.copy(img))  # 对图片像素进行处理
    #  使用scipy会报错
    # scipy.misc.imsave(fname, pil_img)
    # 使用opence保存
    cv2.imwrite(fname, pil_img)

#将图片进行4次缩放处理
num_octave = 4
#定义每次缩放的比例为1.4
octave_scale = 1.4
base_image_path = path = 'data/sky.jpg'
#将图片转换为二维数组
img = preprocess_image(base_image_path)
original_shape = img.shape[1:3]  # 获得图片的[长，宽]
successive_shapes = [original_shape]
#以比例1.4缩小图片
for i in range(1 , num_octave):
    shape = tuple(int(dim / (octave_scale ** i)) for dim in original_shape)
    successive_shapes.append(shape)
#将图片比率由小到达排列
successive_shapes = successive_shapes[::-1]  # 放入需要达到的分辨率
original_img = np.copy(img)
#将图片按照最小比率压缩
shrunk_original_img = resize_img(img, successive_shapes[0])
print(successive_shapes)

#像素调整次数
#每次对图片进行20次求偏导
MAX_ITRN = 20
#限制刺激强度不超过20，如果刺激强度太大，产生的图片效果就好很难看
MAX_STIMULATION = 20
#像素点的调整比率
learning_rate = 0.01

def gradient_ascent(x, iterations, step, max_loss=None):
    '''
    通过对输入求偏导的方式求取函数最大值
    '''
    for i in range(iterations):
        loss_value, grad_values = iterate_grad_ac_step([x])
        if max_loss is not None and loss_value > max_loss:
            break
        #根据偏导数调整输入值
        x += step * grad_values
    return x

for shape in successive_shapes:
    print('Processing image shape, ', shape)
    #变换图片比率
    img = resize_img(img, shape)
    img = gradient_ascent(img, MAX_ITRN, step=learning_rate,
                         max_loss = MAX_STIMULATION)
    #把调整后的图片等比例还原
    upscaled_shrunk_original_img = resize_img(shrunk_original_img, shape)
    same_size_original = resize_img(original_img, shape)
    '''
    图片缩小后再放大回原样会失去某些像素点的信息,我们把原图片和缩小再放大回原样的图片相减
    就能得到失去的像素点值
    '''
    lost_detail = same_size_original - upscaled_shrunk_original_img
    #把失去的像素点值加回到放大图片就相当于调整后的图片与原图片的结合
    img += lost_detail
    #按照比率缩放图片
    shrunk_original_img = resize_img(original_img, shape)
    file_name = fname='dream_at_scale_' + str(shape) + '.png'
    print('save file as : ', file_name)
    save_img(img, "data/" + file_name)

save_img(img, 'data/final_dream.png')