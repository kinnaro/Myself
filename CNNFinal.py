import os, cv2, random
import numpy as np
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers import Input, Dropout, Flatten, Convolution2D, MaxPooling2D, Dense, Activation
from keras.callbacks import EarlyStopping
from keras.utils import np_utils


TRAIN_DIR = 'S:/INPUT/SJTrain/'
TEST_DIR = 'S:/INPUT/SJTest/'

# 把N和V分开读入
# 想全部读入的话，就把那个if去掉。
# 记得把label也存好（0/1）
train_V = [(TRAIN_DIR+i, 1) for i in os.listdir(TRAIN_DIR) if 'V' in i]
train_N = [(TRAIN_DIR+i, 0) for i in os.listdir(TRAIN_DIR) if 'N' in i]
V = random.sample(train_V, 2000)
N = random.sample(train_N, 2000)

# 训练集还是得全盘放进来的
# 这个testset就随便放个label
test_images =  [(TEST_DIR+i, -1) for i in os.listdir(TEST_DIR)]
test_images = random.sample(test_images, 1000)

train_images = V + N
# 把数据散好
# random.shuffle(train_images)
test_images =  test_images
ROWS = 28
COLS = 28

def read_image(tuple_set):
    file_path = tuple_set[0]
    label = tuple_set[1]
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    # 你这里的参数，可以是彩色或者灰度(GRAYSCALE)
    return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC), label
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
# print(x_train.shape) (4000, 1, 28, 28)
# print(x_test.shape) (1000, 1, 28, 28)




X_train = x_train.reshape(-1, 1, 28, 28)
X_test = x_test.reshape(-1, 1, 28, 28)
y_train = np_utils.to_categorical(y_train, num_classes=2)
y_test = np_utils.to_categorical(y_test, num_classes=2)


##############
# CNN模型构造##
##############
optimizer = RMSprop(lr=1e-4)
objective = 'binary_crossentropy'

# 建造模型
model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(1, ROWS, COLS), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))

model.add(Dense(2))
model.add(Activation('softmax'))

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

############
# 训练与预测##
############

# 跑模型

# We add metrics to get more results you want to see
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
              )
print('Training ---------------')
# Another way to train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
model.fit(x_train, y_train,
          epochs=10,
          batch_size=10,
          validation_split=0.2,
          verbose=0,
          shuffle=True,
          callbacks=[early_stopping])

predictions = model.predict(x_test, verbose=0)

print('\nTesting --------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(x_test, y_test)

print('\ntest loss: ', loss)
print('\ntest accuracy ', accuracy)


