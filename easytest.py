from __future__ import print_function

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
arr = [[1, 2],
       [22, 3]]
       # [[3, 4],
       #  [4, 5]]]

# arr = np.asarray(arr)
# print(arr[0, 0])
# print(arr[0][0])
# print(arr)
# print(arr.T)
# arr = np.transpose(arr)
# print(arr)
def gaosu():
    np.random.seed()
    # 使用高斯产生随机数
    temp = np.random.normal(loc=0.5, scale=0.5)
    print(temp)

if __name__ == '__main__':
    temp = np.cos(0)
    print(temp)
    temp = np.cos(np.pi * 1/4)
    print(temp)
    temp = np.cos(np.pi * 1/2)
    print(temp)
    temp = np.cos(np.pi * 3/4)
    print(temp)
    temp = np.cos(np.pi * 1)
    print(temp)
    print(6.123233995736766e-17)

    # gaosu()
    # a = np.array([1,2,3,4,5,6,7,8])
    # a = np.array([1,2,3])
    # temp = np.std(a)  # 计算标准平方差
    # print(temp)
    # print(0.816496580927726 * 0.816496580927726)
    #
    # temp = np.random.normal(size=(10, 10))
    # print(temp)
    #
    # print(-1.63031758e+00)
    #
    # temp = np.arange(12)
    # temp = temp.reshape([2,3,2])
    # print(temp)
    # print(tf.reduce_sum(temp, axis=0))
    # print(tf.reduce_sum(temp, axis=0).shape)

    # temp = np.random.uniform(0., 1., size=(10,)).astype(np.float32)
    # print(temp)
    #
    # count, bins, ignored = plt.hist(temp, 10, normed=True)
    """
    hist原型：
            matplotlib.pyplot.hist(x, bins=10, range=None, normed=False, weights=None,
            cumulative=False, bottom=None, histtype='bar', align='mid', 
            orientation='vertical',rwidth=None, log=False, color=None, label=None, 
            stacked=False, hold=None,data=None,**kwargs)
    输入参数很多，具体查看matplotlib.org,本例中用到3个参数，分别表示：s数据源，bins=12表示bin 
    的个数，即画多少条条状图，normed表示是否归一化，每条条状图y坐标为n/(len(x)`dbin),整个条状图积分值为1
    输出：count表示数组，长度为bins，里面保存的是每个条状图的纵坐标值
         bins:数组，长度为bins+1,里面保存的是所有条状图的横坐标，即边缘位置
         ignored: patches，即附加参数，列表或列表的列表，本例中没有用到。
   """
    # plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')
    # plt.show()

# temp = np.zeros((1,2,23))
# print(temp)
