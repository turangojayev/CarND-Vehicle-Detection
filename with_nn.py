import os
import zipfile

import cv2
import numpy
from keras import Input, optimizers
from keras.callbacks import ModelCheckpoint
from keras.engine import Model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Lambda, activations
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from sklearn.model_selection import train_test_split


def get_data():
    cars = [os.path.join('data', name) for name in zipfile.ZipFile('data/vehicles.zip').namelist() if
            name.endswith('png')]

    notcars = [os.path.join('data', name) for name in zipfile.ZipFile('data/non-vehicles.zip').namelist() if
               name.endswith('png')]

    paths = numpy.concatenate((notcars, cars))
    X = numpy.array(list(map(lambda x: cv2.cvtColor(cv2.imread(x), cv2.COLOR_BGR2RGB), paths)))
    y = numpy.array(([0] * len(notcars) + [1] * len(cars)))
    print(X.shape, y.shape)
    return X, y


def build_model(input_shape=None):
    img_input = Input(shape=input_shape)
    out = Lambda(lambda x: (x - 127.) / 128)(img_input)
    out = Conv2D(16, (3, 3))(out)
    out = BatchNormalization()(out)
    out = Activation(activation=activations.elu)(out)
    out = MaxPooling2D((2, 2))(out)

    out = Conv2D(32, (3, 3))(out)
    out = BatchNormalization()(out)
    out = Activation(activation=activations.elu)(out)
    out = MaxPooling2D((2, 2))(out)

    out = Flatten()(out)
    out = Dense(100, activation=activations.elu)(out)
    out = Dropout(0.7)(out)
    out = Dense(100, activation=activations.elu)(out)
    out = Dropout(0.7)(out)
    out = Dense(1, activation='sigmoid')(out)
    return Model(img_input, out)


if __name__ == '__main__':
    X, y = get_data()
    model = build_model(X[0].shape)
    model.compile(optimizer=optimizers.adam(0.001), loss='binary_crossentropy', metrics=['acc'])
    model.summary()

    model_name = 'car_models/model-{epoch:02d}-{val_acc:.3f}.h5'

    checkpoint = ModelCheckpoint(model_name)

    random_state = numpy.random.randint(0, 100000, 1)[0]
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, stratify=y, train_size=0.8, random_state=random_state)

    class_weights = {0: 1.5, 1: 1}

    weight_val = numpy.ones(len(y_valid), dtype=numpy.float32)
    weight_val[y_valid == 1] *= class_weights[1]
    weight_val[y_valid == 0] *= class_weights[0]

    model.fit(X_train, y_train,
              batch_size=256,
              epochs=30,
              callbacks=[checkpoint],
              class_weight=class_weights,
              validation_data=(X_valid, y_valid, weight_val))
