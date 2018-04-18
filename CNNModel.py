import os, cv2, random
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns

from keras.models import Sequential
from keras.layers import Input, Dropout, Flatten, Convolution2D, MaxPooling2D, Dense, Activation
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils import np_utils
from keras import backend as K

K.set_image_data_format('channels_first')

TRAIN_DIR = 'Q:/INPUT/train/'
TEST_DIR = 'Q:/INPUT/test/'

# 把猫和狗分开读入
# 想全部读入的话，就把那个if去掉。
# 记得把label也存好（0/1）
train_V =   [(TRAIN_DIR+i, 1) for i in os.listdir(TRAIN_DIR) if 'dog' in i]
train_N =   [(TRAIN_DIR+i, 0) for i in os.listdir(TRAIN_DIR) if 'cat' in i]

# 训练集还是得全盘放进来的
# 这个testset就随便放个label
test_images =  [(TEST_DIR+i, -1) for i in os.listdir(TEST_DIR)]

# 合成一个数据集（取个100个玩玩）
train_images = train_V[:100] + train_N[:100]
# 把数据散好
random.shuffle(train_images)
# 取个10个玩玩
test_images =  test_images[:10]

ROWS = 64
COLS = 64

def read_image(tuple_set):
    file_path = tuple_set[0]
    label = tuple_set[1]
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    # 你这里的参数，可以是彩色或者灰度(GRAYSCALE)
    return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC), label
    # 这里，可以选择压缩图片的方式，zoom（cv2.INTER_CUBIC & cv2.INTER_LINEAR）还是shrink（cv2.INTER_AREA）

###########
# 预处理####
###########
CHANNELS = 3


# 代表RGB三个颜色频道

def prep_data(images):
    no_images = len(images)
    data = np.ndarray((no_images, CHANNELS, ROWS, COLS), dtype=np.uint8)
    labels = []

    for i, image_file in enumerate(images):
        image, label = read_image(image_file)
        data[i] = image.T
        labels.append(label)
    return data, labels

x_train, y_train = prep_data(train_images)
x_test, y_shit = prep_data(test_images)

##############
# CNN模型构造##
##############
optimizer = RMSprop(lr=1e-4)
objective = 'binary_crossentropy'

# 建造模型
model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(3, ROWS, COLS), activation='relu'))
model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss=objective, optimizer=optimizer, metrics=['accuracy'])

############
# 训练与预测##
############
nb_epoch = 10
batch_size = 10


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
          validation_split=0.2, verbose=0, shuffle=True, callbacks=[history, early_stopping])

predictions = model.predict(x_test, verbose=0)

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
