import os, keras, cv2, random
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Convolution2D, MaxPooling2D, Dense, Activation
from keras.optimizers import RMSprop
from keras.callbacks import Callback, EarlyStopping


TRAIN_DIR = 'S:/INPUT/NoMove/Train-2/32/'
TEST_DIR = 'S:/INPUT/NoMove/Test-2/32/'

# 把N和V分开读入
# 想全部读入的话，就把那个if去掉。
# 记得把label也存好（0/1）
train_V = [(TRAIN_DIR+i, 1) for i in os.listdir(TRAIN_DIR) if '2 (' in i]
train_N = [(TRAIN_DIR+i, 0) for i in os.listdir(TRAIN_DIR) if '1 (' in i]


# 训练集还是得全盘放进来的
# 这个testset就随便放个label
test_V =  [(TEST_DIR+i, 1) for i in os.listdir(TEST_DIR) if '2 (' in i]
test_N = [(TEST_DIR+i, 0) for i in os.listdir(TEST_DIR) if '1 (' in i]


train_images = train_V + train_N
test_images = test_V + test_N
# 把数据散好
random.shuffle(train_images)
test_images =  test_images
ROWS = 32
COLS = 32

def read_image(tuple_set):
    file_path = tuple_set[0]
    label = tuple_set[1]
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    # 你这里的参数，可以是彩色或者灰度(GRAYSCALE)
    return img, label
    # return img, label
    # 这里，可以选择压缩图片的方式，zoom（cv2.INTER_CUBIC & cv2.INTER_LINEAR）还是shrink（cv2.INTER_AREA）


###########
# 预处理####
###########
CHANNELS = 1
# 代表1个颜色频道


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
x_test, y_test = prep_data(test_images)

X_train = x_train.reshape(-1, 1, 32, 32)
X_test = x_test.reshape(-1, 1, 32, 32)
y_train = np_utils.to_categorical(y_train, 2)
y_test = np_utils.to_categorical(y_test, 2)

###############
#开始建立CNN模型
###############

optimizer = RMSprop(lr=1e-4)
objective = keras.losses.categorical_crossentropy

# 建造模型
model = Sequential()

model.add(Convolution2D(4, 3, 3, border_mode='same', input_shape=(1, ROWS, COLS), activation='relu'))
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
model.add(Activation('softmax'))

print(model.summary())
model.compile(loss=objective, optimizer=optimizer, metrics=['accuracy'])

nb_epoch = 30
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

model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, shuffle=True, callbacks=[history, early_stopping]
          )

score = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

'''
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
'''
'''
classes = model.predict_classes(x_test, verbose=0)
test_accuracy = np.mean(np.equal(y_test, classes))
print(test_accuracy)
'''
