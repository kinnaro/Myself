#coding:utf-8


#导入各种用到的模块组件
import os, random, keras, cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Convolution2D, MaxPooling2D, Dense, Activation
from keras.optimizers import RMSprop
from keras.callbacks import Callback, EarlyStopping


TRAIN_DIR = 'S:/INPUT/NoMove/Train/huidu2/'
TEST_DIR = 'S:/INPUT/NoMove/Test/huidu2/'
# 把N和V分开读入
# 想全部读入的话，就把那个if去掉。
# 记得把label也存好（0/1）
train_V = [(TRAIN_DIR+i, 1) for i in os.listdir(TRAIN_DIR) if 'V' in i]
train_N = [(TRAIN_DIR+i, 0) for i in os.listdir(TRAIN_DIR) if 'N' in i]
test_V = [(TEST_DIR+i, 1) for i in os.listdir(TEST_DIR) if 'V' in i]
test_N = [(TEST_DIR+i, 0) for i in os.listdir(TEST_DIR) if 'N' in i]

train_images = train_N + train_V
test_images = test_N + test_V

def read_image(tuple_set):
	file_path = tuple_set[0]
	label = tuple_set[1]
	img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
	# 你这里的参数，可以是彩色或者灰度(GRAYSCALE)
	return img, label


###########
# 预处理####
###########
CHANNELS = 1
# 代表1个颜色频道


def prep_data(images):
	no_images = len(images)
	data = np.ndarray((no_images, CHANNELS, 28, 28), dtype=np.uint8)
	labels = []

	for i, image_file in enumerate(images):
		image, label = read_image(image_file)
		data[i] = image.T
		labels.append(label)
		return data, labels


#加载数据
x_train, y_train = prep_data(train_images)
x_test, y_test = prep_data(test_images)

#label为2个类别，keras要求格式为binary class matrices,转化一下，直接调用keras提供的这个函数
x_train = x_train.reshape(-1, 1, 28, 28)
x_test = x_test.reshape(-1, 1, 28, 28)
y_train = np_utils.to_categorical(y_train, 2)
y_test = np_utils.to_categorical(y_test, 2)


###############
#开始建立CNN模型
###############

optimizer = RMSprop(lr=1e-4)
objective = keras.losses.categorical_crossentropy

# 建造模型
model = Sequential()

model.add(Convolution2D(4, 3, 3, border_mode='same', input_shape=(1, 28, 28), activation='relu'))
model.add(Convolution2D(4, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(8, 3, 3, border_mode='same', activation='relu'))
model.add(Convolution2D(8, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(16, 3, 3, border_mode='same', activation='relu'))
model.add(Convolution2D(16, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu'))
model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(2))
model.add(Activation('sigmoid'))

model.compile(loss=objective, optimizer=optimizer, metrics=['accuracy'])

nb_epoch = 13
batch_size = 32


## 每个epoch之后，存下loss，便于画出图
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))


early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')

# 跑模型
history = LossHistory()

model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, shuffle=True, callbacks=[history, early_stopping])
'''

def train_model(model, X_train, Y_train):
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, shuffle=True,
              verbose=1, validation_data=0.2, callbacks=[history, early_stopping])
    model.save_weights('model_weights.h5', overwrite=True)
    return model


def test_model(model, X, Y):
    model.load_weights('model_weights.h5')
    score = model.evaluate(X, Y, show_accuracy=True, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    return score


train_model(model, x_train, y_train)
score = test_model(model, x_test, y_test)
model.load_weights('model_weights.h5')
classes = model.predict_classes(x_test, verbose=0)
test_accuracy = np.mean(np.equal(y_test, classes))










loss = history.losses
val_loss = history.val_losses
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('VGG-16 Loss Trend')
plt.plot(loss, 'blue', label='Training Loss')
plt.plot(val_loss, 'green', label='Validation Loss')
plt.xticks(range(0,nb_epoch)[0::2])
plt.legend()
plt.show()
print("Test loss: ")
print(score[0])
print("Test accuracy: ")
print(score[1])
'''