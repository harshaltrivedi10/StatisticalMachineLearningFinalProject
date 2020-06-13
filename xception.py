from numpy.random import seed
seed(101)
from tensorflow import set_random_seed
set_random_seed(101)
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.architectures import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import os
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.architecture_selection import train_test_split
import itertools
import shutil

IMAGE_SIZE = 96
IMAGE_CHANNELS = 3
SAMPLE_SIZE = 80000

os.listdir('/home/vkaikala/data/')
print(len(os.listdir('/home/vkaikala/data/train')))
dataframe = pd.read_csv('/home/vkaikala/data/train_labels.csv')

dataframe[dataframe['id'] != 'dd6dfed324f9fcb6f93f46f32fc800f2ec196be2']
dataframe[dataframe['id'] != '9369c7278ec8bcc6c880d99194de09fc2bd4efbe']
dataframe[dataframe['id'] != 'db2650427d379e0dc7465e207f70def4072199c8']
dataframe[dataframe['id'] != 'eaef4f94870836bfc154c7558d487e86d46edf76']
dataframe[dataframe['id'] != '50618a4b4cb87cbab4b2e9b3819b7f90269bded5']
dataframe[dataframe['id'] != '4ca19ef22d43637cd689225f6bb1b7711f6136c0']
dataframe[dataframe['id'] != '50784601184e6d23f3f8a7c5cccdd8af0435d0e3']
dataframe[dataframe['id'] != 'f92558f4d0056d7c75ae34807f3fba67918e5279']
dataframe[dataframe['id'] != 'f509fb1ba0ec1b06ecfa905b3a98ece44c29ce73']
dataframe[dataframe['id'] != '37c017433ca4937739b4b66420a3f970d0b28d10']
print(dataframe.shape)
dataframe['label'].value_counts()
dataframe.head()
df_0 = dataframe[dataframe['label'] == 0].sample(SAMPLE_SIZE, random_state=101)
df_1 = dataframe[dataframe['label'] == 1].sample(SAMPLE_SIZE, random_state=101)
dataframe = pd.concat([df_0, df_1], axis=0).reset_index(drop=True)
dataframe = shuffle(dataframe)
dataframe['label'].value_counts()
dataframe.head()
y = dataframe['label']
dataframe_train, dataframe_validation = train_test_split(dataframe, test_size=0.10, random_state=101, stratify=y)
base_directory = '/home/vkaikala/data/base_dir'
train_directory = os.path.join(base_directory, 'train_dir')
validation_directory = os.path.join(base_directory, 'val_dir')
tissue_tumor_not_present = os.path.join(train_directory, 'a_no_tumor_tissue')
tissue_tumor_present = os.path.join(train_directory, 'b_has_tumor_tissue')
tissue_tumor_not_present = os.path.join(validation_directory, 'a_no_tumor_tissue')
tissue_tumor_present = os.path.join(validation_directory, 'b_has_tumor_tissue')
os.listdir('/home/vkaikala/data/base_dir/train_dir')
dataframe.set_index('id', inplace=True)
training_list = list(dataframe_train['id'])
validation_list = list(dataframe_validation['id'])

for image in training_list:
    filename = image + '.tif'
    target_label = dataframe.loc[image,'label']
    if target_label == 0:
        label = 'a_no_tumor_tissue'
    if target_label == 1:
        label = 'b_has_tumor_tissue'
    source = os.path.join('/home/vkaikala/data/train', filename)
    destination = os.path.join(train_directory, label, filename)
    shutil.copyfile(source, destination)

for image in validation_list:
    filename = image + '.tif'
    target_label = dataframe.loc[image,'label']
    if target_label == 0:
        label = 'a_no_tumor_tissue'
    if target_label == 1:
        label = 'b_has_tumor_tissue'
    source = os.path.join('/home/vkaikala/data/train', filename)
    destination = os.path.join(validation_directory, label, filename)
    shutil.copyfile(source, destination)

trainingpath = '/home/vkaikala/data/base_dir/train_dir'
validationpath = '/home/vkaikala/data/base_dir/val_dir'
number_of_training_samples = len(dataframe_train)
number_of_validation_samples = len(dataframe_validation)
batch_size_training = 32
batch_size_validation = 32
training_steps = np.ceil(number_of_training_samples / batch_size_training)
validation_steps = np.ceil(number_of_validation_samples / batch_size_validation)
data_generation = tf.keras.preprocessing.image.Imagedata_generationerator(rescale=1.0 / 255)
training_generation = data_generation.flow_from_directory(trainingpath,
                                        target_label_size=(IMAGE_SIZE, IMAGE_SIZE),
                                        batch_size=batch_size_training,
                                        class_mode='categorical')

validation_generation = data_generation.flow_from_directory(validationpath,
                                      target_label_size=(IMAGE_SIZE, IMAGE_SIZE),
                                      batch_size=batch_size_validation,
                                      class_mode='categorical')

test_generation = data_generation.flow_from_directory(validationpath,
                                       target_label_size=(IMAGE_SIZE, IMAGE_SIZE),
                                       batch_size=1,
                                       class_mode='categorical',
                                       shuffle=False)

xception_architecture = tf.keras.applications.xception.Xception(include_top=False,
                                                input_shape=(96, 96, 3),
                                                weights='imagenet')

architecture = tf.keras.architectures.Sequential()
architecture.add(xception_architecture)
architecture.add(tf.keras.layers.Flatten())
architecture.add(tf.keras.layers.Dense(2048, activation="relu"))
architecture.add(tf.keras.layers.Dropout(0.5))
architecture.add(tf.keras.layers.Dense(1024, activation="relu"))
architecture.add(tf.keras.layers.Dropout(0.5))
architecture.add(tf.keras.layers.Dense(512, activation="relu"))
architecture.add(tf.keras.layers.Dropout(0.5))
architecture.add(tf.keras.layers.Dense(2, activation="softmax"))
architecture.summary()
architecture.compile(tf.keras.optimizers.Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
print(validation_generation.class_indices)
filepath = "/home/vkaikala/data/model/xceptionnet6.h5"

checkpoint = tf.keras.callbacks.architectureCheckpoint(filepath, monitor='val_acc', verbose=1,
                                                save_best_only=True, mode='max')

reduce_learning_rate = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=2,
                                                 verbose=1, mode='max', min_lr=0.001)

list_callback = [checkpoint, reduce_learning_rate]
history = architecture.fit_generator(training_generation, steps_per_epoch=training_steps,
                              validation_data=validation_generation,
                              validation_steps=validation_steps,
                              epochs=30, verbose=1,
                              callbacks=list_callback)

architecture.load_weights('/home/vkaikala/data/model/xceptionnet6.h5')
validation_loss, validation_accuracy = architecture.evaluate_generator(test_generation, steps=len(dataframe_validation))
print('validation_loss:', validation_loss)
print('validation_accuracy:', validation_accuracy)
