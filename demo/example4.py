from matplotlib import pylab
import pylab as plt
import numpy as np

def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

mySamples = []
mySigmoid = []

x = plt.linspace(-10, 10, 10)
y = plt.linspace(-10, 10, 100)

plt.plot(x, sigmoid(x), 'r', label = 'linspace(-10, 10, 10)')
plt.plot(y, sigmoid(y), 'r', label='linspace(-10, 10, 1000)')

plt.grid()

plt.title('Sigmoid function')
plt.suptitle('Sigmoid')

plt.legend(loc='lower right')

plt.text(4, 0.8, r'$\sigma(x)=\frac{1}{1+e^(-x)}$', fontsize=15)

plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))
plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.1))

plt.xlabel('X Axis')
plt.ylabel('Y Axis')

plt.show()