from keras.datasets import cifar10
import keras
from keras import backend as K

def load_cifar10_data_metrics():
    num_classes = 10
    # input image dimensions
    img_rows, img_cols = 32, 32

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
        input_shape = (3, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
        input_shape = (img_rows, img_cols, 3)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return (x_train, y_train), (x_test, y_test)

from keras.preprocessing.image import ImageDataGenerator
def load_cifar10_data_from_directory_generator(dirpath, target_size = (256, 256), batch_size=32, color_mode='rgb', class_mode='categorical', classes_subdir_list=None):
    train_datagen = ImageDataGenerator(rescale=1 / 255.0)

    test_datagen = ImageDataGenerator(rescale=1 / 255.0)

    train_generator = train_datagen.flow_from_directory(
        '{0}/train'.format(dirpath),
        target_size=target_size,
        batch_size=batch_size,
        class_mode=class_mode,
        color_mode=color_mode,
        classes=classes_subdir_list)

    test_generator = test_datagen.flow_from_directory(
        '{0}/test'.format(dirpath),
        target_size=target_size,
        batch_size=batch_size,
        class_mode=class_mode,
        color_mode=color_mode,
        classes=classes_subdir_list)

    return train_generator, test_generator

if __name__ == '__main__':
    # (x_train, y_train), (x_test, y_test) = load_cifar10_data_metrics()
    # print(x_train.shape, y_train.shape)
    # print(x_test.shape, y_test.shape)

    import cv2
    import numpy as np

    train_generator, test_generator = load_cifar10_data_from_directory_generator(dirpath='E:/0-ML_database/0-mnist-full',
                                                                                 target_size=(28, 28),
                                                                                 batch_size=32)

    print(len(train_generator)) # len(train_generator) = train_samples_num / batch_size
    for data in train_generator:
        xs, ys = data
        for x, y in zip(xs, ys):
            cv2.imshow('x', x)
            print(y)
            cv2.waitKey()