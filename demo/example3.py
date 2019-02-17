import matplotlib.pyplot as plt
x = [1, 2, 3, 4]
y = [0, 1, 44, 30]
plt.step(x, y)
plt.show()
'''
当x小于1的时候，y是0
当x大于等于1的时候，y就到1
当x大于等于2的时候，y就到44
当x大于等于3的时候，y就到30
'''