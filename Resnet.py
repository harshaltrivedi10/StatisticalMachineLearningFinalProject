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

tf.keras.applications.resnet50.ResNet50
tf.keras.applications.resnet50.preprocess_input
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


# Save train labels to dataframe
df = pd.read_csv("/home/dmalaviy/train_labels.csv")
dataframe_train, dataframe_validation = train_test_split(df, test_size=0.2, stratify=df['label'])

# Check balancing
print("True positive in train data: " + str(len(dataframe_train[dataframe_train["label"] == 1])))
print("True negative in train data: " + str(len(dataframe_train[dataframe_train["label"] == 0])))
print("True positive in validation data: " + str(len(dataframe_validation[dataframe_validation["label"] == 1])))
print("True negative in validation data: " + str(len(dataframe_validation[dataframe_validation["label"] == 0])))

# Import Resnet model, with weights pre-trained on ImageNet.
# Resnet model without the last classifier layers (include_top = False)
resnet_model = tf.keras.applications.resnet50.ResNet50(include_top=False,input_shape=(197, 197, 3),weights='imagenet')

# Freeze the layers
for layer in vgg16_model.layers[:-14]:
    layer.trainable = False

# Check the trainable status of the individual layers
for layer in resnet_model.layers:
    print(layer, layer.trainable)

model = resnet_model.output
model = tf.keras.layers.GlobalAveragePooling2D()(model)
model = tf.keras.layers.Dropout(0.5)(model)
predictions = tf.keras.layers.Dense(4096, activation= 'relu')(model)
predictions = tf.keras.layers.Dropout(0.5)(predictions)
predictions = tf.keras.layers.Dense(2048, activation= 'relu')(predictions)
predictions = tf.keras.layers.Dropout(0.5)(predictions)
predictions = tf.keras.layers.Dense(1024, activation= 'relu')(predictions)
predictions = tf.keras.layers.Dropout(0.5)(predictions)
predictions = tf.keras.layers.Dense(512, activation= 'relu')(predictions)
predictions = tf.keras.layers.Dropout(0.5)(predictions)
predictions = tf.keras.layers.Dense(2, activation= 'softmax')(predictions)
model = tf.keras.models.Model(inputs = resnet_model.input, outputs = predictions)

# Generate batches of tensor image data with real-time data augmentation.
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
                                                    target_size=(197, 197),
                                                    batch_size=training_batch_size,
                                                    class_mode='categorical')

val_generator = test_datagen.flow_from_directory('/home/dmalaviy/data/main/val',
                                                 target_size=(197, 197),
                                                 batch_size=validation_batch_size,
                                                 class_mode='categorical')

# !!! batch_size=1 & shuffle=False !!!!
test_generator = test_datagen.flow_from_directory('/home/dmalaviy/data/main/val',
                                                  target_size=(197, 197),
                                                  batch_size=32,
                                                  class_mode='categorical',
                                                  shuffle=False)

model.summary()

# Get the labels that are associated with each index
print(val_generator.class_indices)
model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.SGD(lr=0.00001, momentum=0.95),
              metrics=['accuracy'])

# updated code
filepath = "/home/dmalaviy/data/model/resnetmodel1.h5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1,
                             save_best_only=True, mode='max')
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=2,
                                                 verbose=1, mode='max', min_lr=0.00001)
callbacks_list = [checkpoint, reduce_lr]

# Due to Disk limits the saving of best weights isn't possible during the training process
history = model.fit_generator(
    train_generator,
    steps_per_epoch=training_steps,
    validation_data=val_generator,
    validation_steps=validation_steps,
    epochs=30,callbacks=callbacks_list)

model.load_weights('/home/dmalaviy/data/model/resnetmodel1.h5')
print("Validation Accuracy: " + str(history.history['val_acc'][-1:]))

# Prediction on validation data sets
val_predict = model.predict_generator(test_generator, steps=len(dataframe_validation), verbose=1)
