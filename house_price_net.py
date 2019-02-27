from keras.datasets import boston_housing

'''波士顿房价预测网络'''

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
def pre():
    print(train_data.shape)  # (404, 13)
    print(test_data.shape)  # (102, 13)

# 数据规格化的做法是，计算每一个种类数据的均值
# 然后每个数据实例减去该均值，然后在除以数据的标准差，
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

# 测试数据要做和训练数据一样的处理
test_data -= mean
test_data /= std

from keras import models
from keras import layers

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    # 损失函数叫’mse’，中文名叫均方误差，其公式如下：
    # MSE = 1/n * (Y - Y’)^2，
    # 其中Y对应的是正确结果，Y’对应的是网络预测结果。
    # 当我们需要预测某个数值区间时，我们就使用MSE作用损失函数
    # 在第三个参数是评价函数，使用mae，它表示平均绝对误差，
    # 它用来描述预测结果与正确结果之差的绝对值
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

def pre2():
    '''。目前有一个问题是，数据量太小，这导致的结果是，
    我们对数据划分的不同方式都会对校验结果产生不同影响，
    如此一来我们就不能对网络的训练情况有确切的掌握。
    处理这种情况的有效办法叫k-fold交叉检验,k一般取4到5，
    选其中的k-1分数据来训练，剩下1份来校验。网络总共训练k次，
    每一份数据都有机会作为校验集，
    最后把k次校验的结果做一次平均。'''
    import numpy as np
    k = 5
    num_val_samples = len(train_data) // k  # 整数除法
    num_epochs = 10
    all_scores = []
    for i in range(k):
        print('processing fold #', i)
        #依次把k分数据中的每一份作为校验数据集
        val_data = train_data[i * num_val_samples : (i+1) * num_val_samples]
        val_targets = train_targets[i* num_val_samples : (i+1) * num_val_samples]

        #把剩下的k-1分数据作为训练数据,如果第i分数据作为校验数据，那么把前i-1份和第i份之后的数据连起来
        partial_train_data = np.concatenate([train_data[: i * num_val_samples],
                                             train_data[(i+1) * num_val_samples:]], axis = 0)
        partial_train_targets = np.concatenate([train_targets[: i * num_val_samples],
                                                train_targets[(i+1) * num_val_samples: ]],
                                              axis = 0)
        print("build model")
        model = build_model()
        #把分割好的训练数据和校验数据输入网络
        model.fit(partial_train_data, partial_train_targets, epochs = num_epochs,
                  batch_size = 1, verbose = 0)
        print("evaluate the model")
        val_mse, val_mae = model.evaluate(val_data, val_targets, verbose = 0)
        all_scores.append(val_mae)

    print(all_scores)  # k=4 [2.4755041079946083, 2.5985907540462985, 2.552518864669422, 2.8074415981179417]
                        # k=5 [2.1732486724853515, 2.531587505340576, 2.315967559814453, 3.0191749572753905, 2.7616650104522704]

import numpy as np
k = 4
num_val_samples = len(train_data) // k #整数除法
num_epochs = 200
all_mae_histories = []

for i in range(k):
    print('processing fold #', i)
    #依次把k分数据中的每一份作为校验数据集
    val_data = train_data[i * num_val_samples : (i+1) * num_val_samples]
    val_targets = train_targets[i* num_val_samples : (i+1) * num_val_samples]

    #把剩下的k-1分数据作为训练数据,如果第i分数据作为校验数据，那么把前i-1份和第i份之后的数据连起来
    partial_train_data = np.concatenate([train_data[: i * num_val_samples],
                                         train_data[(i+1) * num_val_samples:]], axis = 0)
    partial_train_targets = np.concatenate([train_targets[: i * num_val_samples],
                                            train_targets[(i+1) * num_val_samples: ]],
                                          axis = 0)
    print("build model")
    model = build_model()
    #把分割好的训练数据和校验数据输入网络
    history = model.fit(partial_train_data, partial_train_targets,
              validation_data=(val_data, val_targets),
              epochs = num_epochs,
              batch_size = 1, verbose = 0)
    mae_history = history.history['val_mean_absolute_error']
    all_mae_histories.append(mae_history)

average_mae_history = [
    np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)
]

import matplotlib.pyplot as plt
plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()


# 指数滑动平均具有把反复跳动的数据进行平滑的作用，使得我们能从反复变动的数据看出它潜在的变化趋势
def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

smooth_mae_history = smooth_curve(average_mae_history[10:])

plt.plot(range(1, len(smooth_mae_history)+1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

model = build_model()
model.fit(train_data, train_targets, epochs = 30, batch_size = 16, verbose = 0)
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
print(test_mse_score)
print(test_mae_score)  # 训练30次的误差是2.9017364558051613