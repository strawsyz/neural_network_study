from keras.models import Model
from keras import layers
from keras.utils import plot_model
from keras import Input
'''非串行网络
需要有pydot,graphviz等，才能显示图片
这里展示了两种非串行网络，并把网络的结构图
保存在model.png和model2.png'''

text_vocabulary_size = 10000
question_vocabulary_size = 1000
answer_vocabulary_size = 500

text_input = Input(shape=(None, ), dtype='int32', name="text")
embedded_text = layers.Embedding(64, text_vocabulary_size)(text_input)
encoded_text = layers.LSTM(32)(embedded_text)

question_input = Input(shape=(None, ), dtype='int32', name='question')
embedded_question = layers.Embedding(32, question_vocabulary_size)(question_input)
encoded_question = layers.LSTM(16)(embedded_question)

concatenated = layers.concatenate([encoded_text, encoded_question], axis=-1)
answer = layers.Dense(answer_vocabulary_size, activation='softmax')(concatenated)
model = Model([text_input,question_input],answer)
plot_model(model, to_file='model.png', show_shapes=True)


'''读入个人数据，然后预测该人的年龄，收入以及性别'''
vocabulary_size = 50000
num_income_groups = 10

posts_input = Input(shape=(None, ), dtype = 'int32', name = 'posts')
embedded_posts = layers.Embedding(256, vocabulary_size)(posts_input)
x = layers.Conv1D(128, 5, activation='relu')(embedded_posts)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.GlobalMaxPooling1D()(x)
x = layers.Dense(128, activation='relu')(x)

age_prediction = layers.Dense(1, name='age')(x)
income_prediction = layers.Dense(num_income_groups, activation='softmax', name='income')(x)
gender_prediction = layers.Dense(1, activation='sigmoid', name = 'gender')(x)
model = Model(posts_input, [age_prediction, income_prediction, gender_prediction])
model.compile(optimizer='rmsprop', loss=['mse', 'categorical_crossentropy', 'binary_crossentropy'], loss_weights = [0.25, 1. , 10.])
plot_model(model, to_file='model2.png', show_shapes=True)