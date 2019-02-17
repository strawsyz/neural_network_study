import numpy as np
import random
from matplotlib import pyplot

class NeuralNet(object):

    # sizes是每层的神经元个数
    def __init__(self, sizes):
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.w_ = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.b_ = [np.random.randn(y, 1) for y in sizes[1:]]

    # Sigmoid函数，S行曲线，范围(0,1)
    def sigmoid(z):
        return 1.0/(1.0+np.exp(-z))

    # Sigmoid函数的导函数
    def sigmoid_prime(self, z):
        return self.sigmoid(z)*(1-self.sgimoid(z))

    @classmethod
    def test1(self):
        net = NeuralNet([3, 4, 2])
        print('权重：', net.w_)
        print('biases:', net.b_)

    @classmethod
    def test2(self):
        x = np.linspace(-8.0, 8.0, 2000)
        y = self.sigmoid(self,x)
        pyplot.plot(x, y)
        pyplot.show()

    def feedforward(self, x):
        for b, w in zip(self.b_, self.w_):
            x = self.sigmoid(np.dow(w,x)+b)
        return x

    # training_data是训练数据，epochs是训练次数，mini_batch_size是每次训练的样本数，eta是学习效率
    def SGD (self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data:
            n_test = len(test_data)

        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch ", j, ": ", self.evaluate(test_data), "/ ", n_test)
            else:
                print("Epoch ", j, "complete")

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.b_]
        nabla_w = [np.zeros(w.shape) for w in self.w_]

        activation = x
        activations = [x]
        zs = []

        for b, w  in zip(nabla_b,nabla_w):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = self.sgimoid(z)
            activations.append(activation)

        delta = self.cost_derivations(activations[-1], y) * self.sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.sigmoid_prime(z)
            delta = np.dot(self.w_[-l+1].transpose(),delta) * sp
            nabla_b[-1] = delta
            nabla_w[-1] = np.dot(delta, activations[-l-1].transpose())
        return nabla_b, nabla_w

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.b]
        nabla_w = [np.zeros(w.shape) for w in self.w]
        for x, y in mini_batch:
            delta_nabla_b,delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(self.b_, nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(self.w_, nabla_w)]
        self.w_ = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.w_, nabla_w)]
        self.b_ = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.b_, nabla_b)]
    def evaluate(self, test_data) :
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return output_activations - y
if __name__ == '__main__':
    NeuralNet.test2()