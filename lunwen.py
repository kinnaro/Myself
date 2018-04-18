# -*- coding: utf-8 -*-


from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
import random, os, cv2
import numpy as np
import matplotlib.pyplot as plt


TRAIN_DIR = 'S:/INPUT/NoMove/Train/huidu2/'
TEST_DIR = 'S:/INPUT/NoMove/Test/huidu2/'

# 把N和V分开读入
# 想全部读入的话，就把那个if去掉。
# 记得把label也存好（0/1）
train_V = [(TRAIN_DIR+i, 1) for i in os.listdir(TRAIN_DIR) if 'V' in i]
train_N = [(TRAIN_DIR+i, 0) for i in os.listdir(TRAIN_DIR) if 'N' in i]
V = random.sample(train_V, 2000)
N = random.sample(train_N, 2000)

# 训练集还是得全盘放进来的
# 这个testset就随便放个label
test_images = [(TEST_DIR+i, -1) for i in os.listdir(TEST_DIR)]
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



#对训练和测试数据处理，转为float
X_train = x_train.astype('float32')
X_test = x_test.astype('float32')
#对数据进行归一化到0-1 因为图像数据最大是255
X_train /= 255
X_test /= 255

#一共2类
nb_classes = 2

# 将标签进行转换为one-shot
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

#搭建网络
model = Sequential()
# 2d卷积核，包括32个3*3的卷积核  因为X_train的shape是【样本数，通道数，图宽度，图高度】这样排列的，而input_shape不需要（也不能）指定样本数。
model.add(Convolution2D(258, 3, 3, border_mode='same',
                        input_shape=X_train.shape[1:]))#指定输入数据的形状
model.add(Activation('relu'))#激活函数
model.add(MaxPooling2D(pool_size=(2, 2)))                #maxpool


model.add(Convolution2D(126, 4, 4, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(30, 4, 4, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dense(30))
model.add(Activation('relu'))
model.add(Dropout(0.25))                                 #dropout
model.add(Flatten())                                     #压扁平准备全连接
#全连接
model.add(Dense(20))                                    #添加512节点的全连接
model.add(Activation('relu'))                           #激活
model.add(Dropout(0.5))
model.add(Dense(nb_classes))                             #添加输出2个节点
model.add(Activation('softmax'))                         #采用softmax激活

#设定求解器
sgd = SGD(lr=3e-3, decay=1e-6, momentum=0.7, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
#进行训练
batch_size = 32
nb_epoch = 30
data_augmentation = False #是否数据扩充，主要针对样本过小方案
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')

if not data_augmentation:
    print('Not using data augmentation.')
    result=model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=(X_test, Y_test),
              shuffle=True,
              # callbacks=[early_stopping]
                     )
else:
    print('Using real-time data augmentation.')

    # this will do preprocessing and realtime data augmentation
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(X_train)

    # fit the model on the batches generated by datagen.flow()
    result=model.fit_generator(datagen.flow(X_train, Y_train,
                        batch_size=batch_size),
                        samples_per_epoch=X_train.shape[0],
                        nb_epoch=nb_epoch,
                        validation_data=(X_test, Y_test),
                        # callbacks=[early_stopping]
                               )

#model.save_weights(weights,accuracy=False)

# 绘制出结果
plt.figure
plt.plot(result.epoch,result.history['acc'],label="acc")
plt.plot(result.epoch,result.history['val_acc'],label="val_acc")
plt.scatter(result.epoch,result.history['acc'],marker='*')
plt.scatter(result.epoch,result.history['val_acc'])
plt.legend(loc='under right')
plt.show()
plt.figure
plt.plot(result.epoch,result.history['loss'],label="loss")
plt.plot(result.epoch,result.history['val_loss'],label="val_loss")
plt.scatter(result.epoch,result.history['loss'],marker='*')
plt.scatter(result.epoch,result.history['val_loss'],marker='*')
plt.legend(loc='upper right')
plt.show()