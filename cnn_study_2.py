from keras.models import load_model

'''仅仅用来了解cnn结构'''
#重新加载之前保存的模型
model = load_model('cats_and_dogs_small_2.h5')
model.summary()

img_path = 'data/cats_and_dogs_small/test/cats/cat.1700.jpg'

from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
#把图片缩小为150*150像素
img = image.load_img(img_path, target_size = (150,150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis = 0)
#把像素点取值变换到[0,1]之间
img_tensor /= 255.
print(img_tensor.shape)
plt.figure()
plt.imshow(img_tensor[0])


from keras import models
import matplotlib.pyplot as plt
'''
我们把网络的前8层，也就是含有卷积和max pooling的网络层抽取出来，
下面代码会把前八层网络(就是flatten前面的那几层)的输出结果放置到数组layers_outputs中
'''
layer_outputs = [layer.output for layer in model.layers[:8]]
activation_model = models.Model(inputs=model.input, outputs = layer_outputs)
#执行下面代码后，我们能获得卷积层和max pooling层对图片的计算结果
activations = activation_model.predict(img_tensor)
#我们把第一层卷积网络对图片信息的识别结果绘制出来
first_layer_activation = activations[0]
print(first_layer_activation.shape)  # (1,148,148,32)
plt.figure()  # 。网络层获得的信息表示是148*148*32，也就是抽取出的图像大小是148*148个像素，
# 其中每个像素对应一个含有32个元素的向量，我们只把向量中前4个元素表示的信息视觉化，
plt.matshow(first_layer_activation[0, :, : , 4], cmap = 'viridis')

layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)

images_per_row = 16
for layer_name, layer_activation in zip(layer_names, activations):
    #layer_activation的结构为(1, width, height, array_len)
    #向量中的元素个数,(32)
    n_features = layer_activation.shape[-1]
    #获得切片的宽和高
    size = layer_activation.shape[1]
    #在做卷积运算时，我们把图片进行3*3切片，然后计算出一个含有32个元素的向量，
    # 这32个元素代表着网络从3*3切片中抽取的信息
    #我们把这32个元素分成16列，绘制在一行里
    n_cols = n_features // images_per_row  # (32/16=2),一共2行，一行16个图
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0, : , :, col * images_per_row  + row]
            #这32个元素中，不一定每个元素对应的值都能绘制到界面上，所以我们对它做一些处理
            # ，使得它能画出来
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = channel_image

    scale = 1. / size
    plt.figure(figsize = (scale * display_grid.shape[1], scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect = 'auto', cmap = 'viridis')