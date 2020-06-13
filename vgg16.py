import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import shutil
from tqdm import tqdm
from PIL import Image
from PIL import ImageDraw
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

tf.keras.applications.vgg16.VGG16
tf.keras.applications.vgg16.preprocess_input
tf.keras.models.Sequential
tf.keras.layers.Dense
tf.keras.layers.Flatten
tf.keras.layers.Dropout
tf.keras.preprocessing.image.ImageDataGenerator
tf.keras.optimizers.Adam
tf.keras.optimizers.SGD
tf.keras.callbacks.EarlyStopping
tf.keras.callbacks.ReduceLROnPlateau
tf.keras.callbacks.ModelCheckpoint


df = pd.read_csv("/home/dmalaviy/train_labels.csv")
dataframe_train, dataframe_validation = train_test_split(df, test_size=0.2, stratify=df['label'])

# Check balancing
print("True positive in train data: " + str(len(dataframe_train[dataframe_train["label"] == 1])))
print("True negative in train data: " + str(len(dataframe_train[dataframe_train["label"] == 0])))
print("True positive in validation data: " + str(len(dataframe_validation[dataframe_validation["label"] == 1])))
print("True negative in validation data: " + str(len(dataframe_validation[dataframe_validation["label"] == 0])))

# Import VGG16 model, with weights pre-trained on ImageNet.
# VGG model without the last classifier layers (include_top = False)
vgg16_model = tf.keras.applications.vgg16.VGG16(include_top=False,
                                                input_shape=(96, 96, 3),
                                                weights='imagenet')

# Freeze the layers
for layer in vgg16_model.layers[:-12]:
    layer.trainable = False

# Check the trainable status of the individual layers
for layer in vgg16_model.layers:
    print(layer, layer.trainable)

model = tf.keras.models.Sequential()
model.add(vgg16_model)
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(4096, activation="relu"))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(4096, activation="relu"))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(1024, activation="relu"))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(512, activation="relu"))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(2, activation="softmax"))


num_train_samples = len(dataframe_train)
num_val_samples = len(dataframe_validation)
training_batch_size = 32
validation_batch_size = 32

training_steps = np.ceil(num_train_samples / training_batch_size)
validation_steps = np.ceil(num_val_samples / validation_batch_size)

print(training_steps)
print(validation_steps)

# Augmentation
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    vertical_flip=True,
    horizontal_flip=True,
    rotation_range=90,
    shear_range=0.05)


# Augmentation configuration for validatiopn and testing: only rescaling!
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

# Generator that will read pictures found in subfolers of 'main/train', and indefinitely generate batches of augmented image data
train_generator = train_datagen.flow_from_directory('/home/dmalaviy/data/main/train',
                                                    target_size=(96, 96),
                                                    batch_size=training_batch_size,
                                                    class_mode='categorical')

val_generator = test_datagen.flow_from_directory('/home/dmalaviy/data/main/val',
                                                 target_size=(96, 96),
                                                 batch_size=validation_batch_size,
                                                 class_mode='categorical')

# !!! batch_size=1 & shuffle=False !!!!
test_generator = test_datagen.flow_from_directory('/home/dmalaviy/data/main/val',
                                                  target_size=(96, 96),
                                                  batch_size=32,
                                                  class_mode='categorical',
                                                  shuffle=False)

model.summary()
# Get the labels that are associated with each index
print(val_generator.class_indices)

#model.compile(tf.keras.optimizers.Adam(lr=0.00001), loss='binary_crossentropy', metrics=['acc'])
model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.SGD(lr=0.00001, momentum=0.95),
              metrics=['accuracy'])

# updated code
filepath = "/home/dmalaviy/data/model/vgg16model1.h5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

history = model.fit_generator(
    train_generator,
    steps_per_epoch=training_steps,
    validation_data=val_generator,
    validation_steps=validation_steps,
    epochs=30,callbacks=callbacks_list)

model.load_weights('/home/dmalaviy/data/model/vgg16model1.h5')
print("Validation Accuracy: " + str(history.history['val_acc'][-1:]))

# Prediction on validation data sets
val_predict = model.predict_generator(test_generator, steps=len(dataframe_validation), verbose=1)
