from functools import partial
import numpy as np
from matplotlib import pyplot as plt

x_samples = np.arange(-3, 3.01,0.01)
# print(x_samples)
# print(len(x_samples))  # 601
PDF = np.empty(x_samples.shape)
print(PDF)
# np.round() 方法返回浮点数x的四舍五入值。
PDF[x_samples < 0] = np.round(x_samples[x_samples<0] + 3.5)/3
print(PDF)
PDF[x_samples >= 0] = 0.5 * np.cos(np.pi * x_samples[x_samples>=0] + 0.5)
print(PDF)
PDF /= np.sum(PDF)

