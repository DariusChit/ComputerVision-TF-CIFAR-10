import keras.utils.np_utils
from keras.backend import set_session
from keras.datasets import cifar10
import tensorflow as tf
import numpy as np
from keras.optimizer_experimental.adam import Adam
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.regularizers import l2
from tensorflow.python.client.session import Session
from tensorflow.core.protobuf.config_pb2 import ConfigProto
import keras

config = ConfigProto( device_count = {'GPU': 1 , 'CPU': 6} )
sess = Session(config=config)
keras.backend.set_session(sess)

cols, rows, channels, num_classes = 32, 32, 3, 10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
for i in range(0, 9):
    plt.subplot(330 + 1 + i)
    plt.imshow(x_train[i])
plt.show()

#img_rows, img_cols , channels= 32,32,3
#for i in range(0,9):
#    plt.subplot(330 + 1 + i)
#    plt.imshow(x_train[i])
#plt.show()

datagen = ImageDataGenerator(zoom_range=[0.85, 1.0], rotation_range=30, horizontal_flip=True, width_shift_range=0.1,
                             height_shift_range=0.1)
datagen.fit(x_train)

for x_bch, y_bch in datagen.flow(x_train, y_train, batch_size=9) :
    for i in range(0, 9):
        plt.subplot(330 + 1 + i)
        plt.imshow(x_bch[i].astype(np.uint8))
    plt.show()
    break

#it = datagen.flow(samples, batch_size=1)
#for i in range(9):
#    plt.subplot(330+ 1 + i)
#    batch = it.next()
#    image = batch[0].astype('uint8')
#    plt.imshow(image)
#plt.show()

#datagen.fit(x_train)
#for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=9):
#    for i in range(0, 9):
#        plt.subplot(330 + 1 + i)
#        plt.imshow(x_batch[i].astype(np.uint8))
#    plt.show()
#    break

#reshape
x_train = x_train.reshape(x_train.shape[0], rows, cols, channels)
x_test = x_test.reshape(x_test.shape[0], rows, cols, channels)
input_shape = (rows, cols, 1)
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

#float and normalisation
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
mean = np.mean(x_train)
sd = np.std(x_train)
x_test = (x_test - mean) / sd
x_train = (x_train - mean) / sd

num_classes = 10

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

def plothelp(hist):
    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.title('model_accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

reg=None
num_filters=32
ac='relu'
adm=Adam(lr=0.001,decay=0, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
opt=adm
drop_dense=0.5
drop_conv=0

model = Sequential()

model.add(Conv2D(num_filters, (3, 3), activation=ac, kernel_regularizer=reg, input_shape=(rows, cols, channels),padding='same'))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(num_filters, (3, 3), activation=ac,kernel_regularizer=reg,padding='same'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(drop_conv))

model.add(Conv2D(2*num_filters, (3, 3), activation=ac,kernel_regularizer=reg,padding='same'))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(2*num_filters, (3, 3), activation=ac,kernel_regularizer=reg,padding='same'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(drop_conv))

model.add(Conv2D(4*num_filters, (3, 3), activation=ac,kernel_regularizer=reg,padding='same'))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(4*num_filters, (3, 3), activation=ac,kernel_regularizer=reg,padding='same'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(drop_conv))

model.add(Flatten())
model.add(Dense(512, activation=ac,kernel_regularizer=reg))
model.add(BatchNormalization())
model.add(Dropout(drop_dense))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'],optimizer=opt)

model.summary()

history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=128),
                              steps_per_epoch= len(x_train) / 128, epochs=35, validation_data=(x_train, y_train))
train_acc = model.evaluate(x_train, y_train, batch_size=128)
plothelp(history)

model.save('second_attempt')
