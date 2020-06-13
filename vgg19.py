from numpy.random import seed
seed(101)
from tensorflow import set_random_seed
set_random_seed(101)

import pandas as pd
import numpy as np


import tensorflow as tf
from tensorflow import keras
tf.keras.preprocessing.image.ImageDataGenerator
tf.keras.layers.Conv2D
tf.keras.layers.MaxPooling2D
tf.keras.layers.Dense
tf.keras.layers.Dropout
tf.keras.layers.Flatten
tf.keras.layers.Activation
tf.keras.models.Sequential
tf.keras.callbacks.EarlyStopping
tf.keras.callbacks.ReduceLROnPlateau
tf.keras.callbacks.ModelCheckpoint
tf.keras.optimizers.Adam
tf.keras.applications.vgg19.VGG19
tf.keras.applications.vgg19.preprocess_input

import os
import cv2

from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import itertools
import shutil

IMAGE_SIZE = 96
IMAGE_CHANNELS = 3

SAMPLE_SIZE = 80000
# the number of ima

os.listdir('/home/htrived2/data/')

print(len(os.listdir('/home/htrived2/data/train')))
# print(len(os.listdir('/home/htrived2/data/test')))

df_data = pd.read_csv('/home/htrived2/train_labels.csv')

# removing this image because it caused a training error previously
df_data[df_data['id'] != 'dd6dfed324f9fcb6f93f46f32fc800f2ec196be2']

# removing this image because it's black
df_data[df_data['id'] != '9369c7278ec8bcc6c880d99194de09fc2bd4efbe']


print(df_data.shape)

df_data['label'].value_counts()

df_data.head()

# take a random sample of class 0 with size equal to num samples in class 1
df_0 = df_data[df_data['label'] == 0].sample(SAMPLE_SIZE, random_state = 101)
# filter out class 1
df_1 = df_data[df_data['label'] == 1].sample(SAMPLE_SIZE, random_state = 101)

# concat the dataframes
df_data = pd.concat([df_0, df_1], axis=0).reset_index(drop=True)
# shuffle
df_data = shuffle(df_data)

df_data['label'].value_counts()

df_data.head()

y = df_data['label']

df_train, df_val = train_test_split(df_data, test_size=0.10, random_state=101, stratify=y)

training_path = '/home/htrived2/data/base_dir/train_dir'
validation_path = '/home/htrived2/data/base_dir/val_dir'


num_train_samples = len(df_train)
num_val_samples = len(df_val)
training_batch_size = 32
validation_batch_size = 32


training_steps = np.ceil(num_train_samples / training_batch_size)
validation_steps = np.ceil(num_val_samples / validation_batch_size)

datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255)

train_gen = datagen.flow_from_directory(training_path,
                                        target_size=(IMAGE_SIZE,IMAGE_SIZE),
                                        batch_size=training_batch_size,
                                        class_mode='categorical')

val_gen = datagen.flow_from_directory(validation_path,
                                        target_size=(IMAGE_SIZE,IMAGE_SIZE),
                                        batch_size=validation_batch_size,
                                        class_mode='categorical')

# Note: shuffle=False causes the test dataset to not be shuffled
test_gen = datagen.flow_from_directory(validation_path,
                                        target_size=(IMAGE_SIZE,IMAGE_SIZE),
                                        batch_size=1,
                                        class_mode='categorical',
                                        shuffle=False)
   
vgg19_model = tf.keras.applications.vgg19.VGG19(include_top=False,
                                                input_shape=(96, 96, 3),
                                                # weights='../input/VGG16weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
                                                weights='imagenet')

model = tf.keras.models.Sequential()
model.add(vgg19_model)
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(2048, activation="relu"))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(1024, activation="relu"))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(512, activation="relu"))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(2, activation="softmax"))
model.summary()

model.compile(tf.keras.optimizers.Adam(lr=0.00001), loss='binary_crossentropy', metrics=['accuracy'])

print(val_gen.class_indices)

filepath = "/home/htrived2/model/vgg/model_with_2048_1024_512_2_00001_02.h5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='validation_accuracy', verbose=1, 
                             save_best_only=True, mode='max')

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='validation_accuracy', factor=0.5, patience=2, 
                                   verbose=1, mode='max', min_lr=0.00001)
                              
                              
callbacks_list = [checkpoint, reduce_lr]
history = model.fit_generator(train_gen, steps_per_epoch=training_steps, 
                    validation_data=val_gen,
                    validation_steps=validation_steps,
                    epochs=30, verbose=1,
                   callbacks=callbacks_list)

model.metrics_names
model.load_weights('/home/htrived2/model/vgg/model_with_2048_1024_512_2_00001_02.h5')
validation_loss, validation_accuracy = model.evaluate_generator(test_gen, steps=len(df_val))

print('validation_loss:', validation_loss)
print('validation_accuracy:', validation_accuracy)
