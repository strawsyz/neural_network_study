import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

img_height = 28
img_width = 28
batch_size = 100
out_height = 28
out_width = 28
c_dim = 1
y_dim = 10
df_dim = 64
dfc_dim = 1024
gf_dim = 64
gfc_dim = 1024
max_epoch = 300
z_dim = 100 # 噪声维度
save_path = './dcgan_output/'


def lrelu(x,leak=0.2):
    # leakyrelu
    '''参考Rectier Nonlinearities Improve Neural Network Acoustic Models'''
    return tf.maximum(x,leak*x)  # 返回结果维度不变

def conv2d(input_,output_dim,name,k_h=5,k_w=5,s_h=2,s_w=2,stddev=0.02):
    '''普通的卷积层'''
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal(stddev=stddev,shape=[k_h, k_w, input_.shape.as_list()[-1], output_dim]))
        conv = tf.nn.conv2d(input_,w,strides=[1,s_h,s_w,1],padding='SAME')
        b = tf.Variable(tf.zeros([output_dim]))
        return tf.reshape(tf.nn.bias_add(conv,b),conv.shape)

def conv_cond_concat(xb,yb):
    '''把label条件附加在输入上,DCGAN用上了条件GAN'''
    # 输入的x第一个参数默认为batch_size
    xb_shape = xb.shape.as_list()
    yb_shape = yb.shape.as_list()
    yb = tf.reshape(yb,[yb_shape[0],1,1,yb_shape[-1]])
    return tf.concat([xb,yb*tf.ones([xb_shape[0],xb_shape[1],xb_shape[2],yb_shape[-1]])],3) # 连接最后一维

def batch_norm(x,name,train = True, epsilon=1e-5, momentum=0.9):
    '''如名字所示'''
    # 这里面也有可训练的变量
    return tf.contrib.layers.batch_norm(x, decay=momentum,updates_collections=None,epsilon=epsilon,scale=True,is_training=train,scope = name)

def linear(input_, output_dim,name,stddev=0.02):
    '''相当于全连接层，做矩阵的相乘'''
    with tf.name_scope(name):  # 作用于操作
        matrix = tf.Variable(tf.random_normal(shape=[input_.shape.as_list()[-1],output_dim],stddev=stddev,dtype=tf.float32))
        bias = tf.Variable(tf.zeros([output_dim]))
        return tf.matmul(input_, matrix) + bias

def deconvolution(input_,output_dim,name,k_h=5,k_w=5,s_h=2,s_w=2,stddev=0.02):
    '''反卷积，放大'''
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal(shape=[k_h,k_w,output_dim[-1],input_.shape.as_list()[-1]],stddev=stddev))
        deconv = tf.nn.conv2d_transpose(input_,w,output_shape=output_dim,strides=[1,s_h,s_w,1])
        b = tf.Variable(tf.zeros([output_dim[-1]]))
        return tf.reshape(tf.nn.bias_add(deconv,b),deconv.shape)

def get_z(shape):
    '''生成随机噪声，作为G的输入'''
    #  numpy.random.uniform(low,high,size)
    #  从一个均匀分布[low,high)中随机采样，注意定义域是左闭右开，即包含low，不包含high.
    return np.random.uniform(-1.,1.,size=shape).astype(np.float32)

def discriminator(x,x_generated,y):
    # 这里遇到大坑，调用两次就跪了，除非用tf.get_variable()替代tf.Variable()
    # 因为x和x_generated要公用一套判别式的权值，如果调用两个discriminator会导致结果不一样，这里就合在一起了
    x = tf.concat([x,x_generated],0)
    # 因此y也要做相应调整
    y = tf.concat([y,y],0)
    # 把条件和x连在一起
    x = conv_cond_concat(x,y)

    h0 = lrelu(conv2d(x,c_dim+y_dim,name='d_c'))
    h0 = conv_cond_concat(h0,y)

    h1 = lrelu(batch_norm(conv2d(h0,df_dim+y_dim,name='d_c'),name='d_cb1'))
    h1 = tf.reshape(h1,[batch_size+batch_size,-1])
    h1 = tf.concat([h1,y],1)

    h2 = lrelu(batch_norm(linear(h1,dfc_dim,name='d_c'),name='d_cb2'))
    h2 = tf.concat([h2,y],1)

    h3 = linear(h2,1,name='d_fc')

    # 把得到的结果按原来的逆步骤分成两个
    y_data = tf.nn.sigmoid(tf.slice(h3, [0, 0], [batch_size, -1], name=None))
    y_generated = tf.nn.sigmoid(tf.slice(h3, [batch_size, 0], [-1, -1], name=None))

    return y_data,y_generated

def generator(z,y):
    s_h,s_w = out_height,out_width
    s_h2,s_w2 = int(s_h/2),int(s_w/2)
    s_h4,s_w4 = int(s_h/4),int(s_w/4)

    # 噪声也要连接标签
    z = tf.concat([z,y],1)

    h0 = tf.nn.relu(batch_norm(linear(z,gfc_dim,name='g_fc'),name='g_fcb1'))
    h0 = tf.concat([h0,y],1)

    h1 = tf.nn.relu(batch_norm(linear(h0,gf_dim*2*s_h4*s_w4,name='g_fc'),name='g_fcb2'))
    h1 = tf.reshape(h1,[batch_size,s_h4,s_w4,gf_dim*2])
    h1 = conv_cond_concat(h1,y)

    h2 = tf.nn.relu(batch_norm(deconvolution(h1,[batch_size,s_h2,s_w2,gf_dim*2],name='g_dc'),name='g_dcb'))
    h2 = conv_cond_concat(h2,y)
    # 原文这里用的是tanh，不过要输出图片的话建议用sigmoid
    return tf.nn.sigmoid(deconvolution(h2,[batch_size,s_h,s_w,c_dim],name='g_dc'))

def save(samples, index,shape):
    '''只是用来把图片保存到本地，和训练无关'''
    x,y=shape  # 保存图片的宽高（每个单位一张生成数字）
    fig = plt.figure(figsize=(x,y))
    gs = gridspec.GridSpec(x,y)
    gs.update(wspace=0.05,hspace=0.05)

    for i,sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample[:,:,0],cmap='Greys_r')
    plt.savefig(save_path+'{}.png'.format(str(index).zfill(3)))
    plt.close(fig)

mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)  # 加载数据集

z = tf.placeholder(tf.float32,shape=[None,z_dim])
x = tf.placeholder(tf.float32,shape=[batch_size,img_height,img_width,c_dim])
y = tf.placeholder(tf.float32,shape=[batch_size,y_dim])

x_generated = generator(z,y)  # 假图
d_real,d_fake = discriminator(x,x_generated,y)  # 真、假图各自概率

d_loss = -tf.reduce_mean(tf.log(d_real+1e-30) + tf.log(1.-d_fake+1e-30))  # 不加这个1e-30会出现log(0)
g_loss = -tf.reduce_mean(tf.log(d_fake+1e-30))  # tf有内置的sigmoid_cross_entropy_with_logits可以解决这个问题，但我没用它

# 这一步很关键，主要是用来取出一切可以训练的参数，命名前缀决定了这个参数属于谁（建层的时候特地写的）
t_vars = tf.trainable_variables()  # 所有可训练变量的列表
d_vars = [var for var in t_vars if var.name.startswith('d_')]
g_vars = [var for var in t_vars if var.name.startswith('g_')]

d_optimizer = tf.train.AdamOptimizer(0.0002,beta1=0.5)  # beta1是momentum
g_optimizer = tf.train.AdamOptimizer(0.0002,beta1=0.5)

# update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
# with tf.control_dependencies(update_ops):
d_solver = d_optimizer.minimize(d_loss,var_list = d_vars)
g_solver = g_optimizer.minimize(g_loss,var_list = g_vars)



sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists(save_path):
    os.makedirs(save_path)  # 保存图片的位置

iteration = int(50000/batch_size)
for epoch in range(max_epoch):

    # 以下几行和训练无关，只是把G的生成样本保存在本地save_path目录下
    labels = [i for i in range(10) for _ in range(10)]  # 我要让他生成的数字，每行相同，每列从0到1递增
    cond_y = sess.run(tf.one_hot(np.array(labels),depth=10))  # 喂的字典不能是tensor，我run成np array
    samples = sess.run(x_generated, feed_dict = {z:get_z([100,z_dim]),y:cond_y})
    shape = [10,10]  # 维度和labels的宽高匹配
    save(samples, epoch, shape)  # 保存图片
    # 以上几行和训练无关，去掉也可以，但就没有可视化结果了。

    # 主要的训练步骤
    for idx in range(iteration):
        # 提取及转换数据
        x_mb,y_mb = mnist.train.next_batch(batch_size)
        z_mb = get_z([batch_size,z_dim])
        x_mb = np.reshape(x_mb,[batch_size,out_height,out_width,1])
        # 判别器训练
        _,d_loss_ = sess.run([d_solver,d_loss],feed_dict={x:x_mb,z:z_mb,y:y_mb.astype(np.float32)})
        # 生成器训练
        _,g_loss_ = sess.run([g_solver,g_loss],feed_dict={x:x_mb,z:z_mb,y:y_mb.astype(np.float32)})
        if idx % 100 == 0:
            print('epoch:[%d/%d][%d/%d], d_loss: %.3f, g_loss: %.3f\n' % (epoch,max_epoch,idx+1,iteration,d_loss_,g_loss_))
sess.close()