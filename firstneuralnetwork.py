import numpy as np
from scipy.special import expit
from scipy.special import logit
import matplotlib.pyplot as plt

'''实现了简单的数字识别用的神经网络'''
class NeuralNetWork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        '''初始化网络，设置输入层，中间层，输出层节点，学习效率'''
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        #  设置学习效率
        self.lr = learningrate

        #  初始化权重矩阵,权重是-0.5到0.5的随机数
        self.wih = np.random.rand(self.hnodes, self.inodes) - 0.5
        self.who = np.random.rand(self.onodes, self.hnodes) - 0.5

        #  激活函数 sigmod 函数
        # pylint: disable=no-member
        # self.activation_function = lambda x:scipy.special.expit(x) # 用这种方式会报错，原因不明
        self.activation_function = lambda x: expit(x)  # 这种方式就不会报错了
        self.inverse_activation_function = lambda  x: logit(x)

    def train(self, inputs_list, targets_list):
        '''用于训练'''
        # 根据输入的训练数据更新节点链路权重
        #  先把input_list,target_list转成numpy支持的二维矩阵,并转置
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        # 计算信号经过输入层后产生的信号量f
        hidden_inputs = np.dot(self.wih, inputs)
        #  中间层神经元对输入的信号做激活函数后得到输出信号
        hidden_outputs = self.activation_function(hidden_inputs)
        # 计算信号经过中间层后产生的信号量
        final_inputs = np.dot(self.who, hidden_outputs)
        # 输出层对信号量进行激活函数后得到最终输出信号
        final_outputs = self.activation_function(final_inputs)

        # 反馈误差，优化权重
        # 计算误差
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)
        # 根据误差计算链路权重的更新量，然后把更新加到原来的链路权重上
        self.who += self.lr * np.dot((output_errors * final_outputs * (1 - final_outputs)),
                                     np.transpose(hidden_outputs))
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)),
                                     np.transpose(inputs))

        pass

    def query(self, inputs):
        '''用于测试数据的时候调用'''
        #  根据输入数据计算并输出答案
        #  计算中间层从输入层得到的信号量
        hidden_inputs = np.dot(self.wih, inputs)
        #  计算中间层经过激活函数计算后，得到的输出信号量
        hidden_inputs = self.activation_function(hidden_inputs)
        #  计算最外层接收到的信号量
        final_inputs = np.dot(self.who, hidden_inputs)
        final_outputs = self.activation_function(final_inputs)
        # print(final_outputs)
        return final_outputs

    def backQuery(self,target_lists):
        '''将结果向量转置以便反解出输入信号量'''
        # 将结果向量转置，以便反解出输入信号量
        final_outputs = np.array(target_lists, ndmin=2).T

        # 通过激活函数的反函数得到输出层的输入信号
        final_inputs = self.inverse_activation_function(final_outputs)
        # 获取中间层的输出信号
        hidden_outputs =np.dot(self.who.T, final_inputs)
        # 将信号量调整到0.01到0.99之间
        hidden_outputs -= np.min(hidden_outputs)
        hidden_outputs /= np.max(hidden_outputs)
        hidden_outputs *= 0.98
        hidden_outputs += 0.01

        # 通过激活函数的反函数计算中间层获得的输入信号量
        hidden_inputs = self.inverse_activation_function(hidden_outputs)

        # 计算输入层的输出信号量
        inputs = np.dot(self.wih.T, hidden_inputs)
        # 将信号量调整到 0.01到0.99之间
        inputs -= np.min(inputs)
        inputs /= np.max(inputs)
        inputs *= 0.98
        inputs += 0.01
        # input对应的就是输入神经网络的图像像素数组
        return inputs

if __name__ == '__main__':
    # 初始化网络
    input_nodes = 784  # 输入28*28的灰度图像的每个像素的灰度值
    hidden_nodes = 100
    output_nodes = 10  # 输出结果,如：[0.01,0.01,0.01,0.99,0.01,0.01,0.01,0.01,0.01,0.01]，表示数字4
    learning_rate = 0.3
    n = NeuralNetWork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    # 读取训练数据
    training_data_file = open("data/mnist_train.csv")
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    # 加入epocs,设定网络的训练循环次数
    # 。一般来说，epochs 的数值越大，网络被训练的就越精准，但如果超过一个阈值，
    # 网络就会引发一个过渡拟合的问题，
    # 也就是网络会对老数据识别的很精准，但对新数据识别的效率反而变得越来越低
    epochs = 1  # 训练7次能到 0.9508

    for e in range(epochs):
        # 用,分开数据，分别读入
        for record in training_data_list:
            all_values = record.split(',')
            # 如果inputs为0，会这样会导致链路权重更新出问题,所以 0.01到1
            inputs = (np.asfarray(all_values[1:]))/255.0 * 0.99 + 0.01
            # 设置图片与数字的对应关系
            targets = np.zeros(output_nodes) + 0.01
            targets[int(all_values[0])] = 0.99  # 原来是0.99，好像改成1也没有问题
                                                # 0.99的原因可能是因为用了sigmoid函数不可能输出1
            n.train(inputs, targets)

    test_data_file = open('data/mnist_test.csv')
    test_data_list = test_data_file.readlines()
    test_data_file.close()
    scores = []
    for record in test_data_list:
        all_values = record.split(',')
        correct_number = int(all_values[0])
        # 预处理数字图片
        inputs = (np.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
        # 让网络判断图片对应的数字
        outputs = n.query(inputs)
        # 找到数值最大的神经元对应编号
        label = np.argmax(outputs)
        if label == correct_number:
            scores.append(1)
        else:
            scores.append(0)
    scores_array = np.asarray(scores)
    print('result is ', scores_array.sum() / scores_array.size)

    label = 8
    targets = np.zeros(output_nodes) + 0.01
    targets[label] = 0.99
    print(targets)
    image_data = n.backQuery(targets)
    print(image_data.reshape(28, 28))
    plt.imshow(image_data.reshape(28, 28), cmap="Greys", interpolation="None")
    plt.show()