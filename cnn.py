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

import os
import cv2

from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import itertools
import shutil

IMAGE_SIZE = 96
IMAGE_CHANNELS = 3

SAMPLE_SIZE = 80000 # the number of ima

os.listdir('/home/jvaghasi/data/')

print(len(os.listdir('/home/jvaghasi/data/train')))
print(len(os.listdir('/home/jvaghasi/data/test')))

df_data = pd.read_csv('/home/jvaghasi/data/train_labels.csv')

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

training_path = '/home/jvaghasi/data/base_dir/train_dir'
validation_path = '/home/jvaghasi/data/base_dir/val_dir'

num_train_samples = len(df_train)
num_val_samples = len(df_val)
training_batch_size = 16
validation_batch_size = 16


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
   
kernel_size = (3,3)
pool_size= (2,2)
first_filters = 32
second_filters = 64
third_filters = 128

dropout_conv = 0.2
dropout_dense = 0.2


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(first_filters, kernel_size, activation = 'relu', input_shape = (96, 96, 3)))
model.add(tf.keras.layers.Conv2D(first_filters, kernel_size, activation = 'relu'))
model.add(tf.keras.layers.Conv2D(first_filters, kernel_size, activation = 'relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size = pool_size)) 
model.add(tf.keras.layers.Dropout(dropout_conv))

model.add(tf.keras.layers.Conv2D(second_filters, kernel_size, activation ='relu'))
model.add(tf.keras.layers.Conv2D(second_filters, kernel_size, activation ='relu'))
model.add(tf.keras.layers.Conv2D(second_filters, kernel_size, activation ='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size = pool_size))
model.add(tf.keras.layers.Dropout(dropout_conv))

model.add(tf.keras.layers.Conv2D(third_filters, kernel_size, activation ='relu'))
model.add(tf.keras.layers.Conv2D(third_filters, kernel_size, activation ='relu'))
model.add(tf.keras.layers.Conv2D(third_filters, kernel_size, activation ='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size = pool_size))
model.add(tf.keras.layers.Dropout(dropout_conv))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(2048, activation = "relu"))
model.add(tf.keras.layers.Dropout(dropout_dense))
model.add(tf.keras.layers.Dense(1024, activation = "relu"))
model.add(tf.keras.layers.Dropout(dropout_dense))
model.add(tf.keras.layers.Dense(512, activation = "relu"))
model.add(tf.keras.layers.Dropout(dropout_dense))
model.add(tf.keras.layers.Dense(2, activation = "softmax"))

model.summary()

model.compile(tf.keras.optimizers.Adam(lr=0.00001), loss='binary_crossentropy', metrics=['accuracy'])

print(val_gen.class_indices)

filepath = "/home/jvaghasi/cnn/model10.h5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, 
                             save_best_only=True, mode='max')

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=2, 
                                   verbose=1, mode='max', min_lr=0.00001)
                              
                              
callbacks_list = [checkpoint, reduce_lr]



history = model.fit_generator(train_gen, steps_per_epoch=training_steps, 
                    validation_data=val_gen,
                    validation_steps=validation_steps,
                    epochs=30, verbose=1,
                   callbacks=callbacks_list)

model.metrics_names

model.load_weights(filepath)

val_loss, val_acc = model.evaluate_generator(test_gen, steps=len(df_val))

print('val_loss:', val_loss)
print('val_acc:', val_acc)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fileacc = "/home/jvaghasi/cnn/accuracy10"
fileloss = "/home/jvaghasi/cnn/loss10"
fileconf = "/home/jvaghasi/cnn/confusion10"

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig(fileloss)


plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig(fileacc)

predictions = model.predict_generator(test_gen, steps=len(df_val), verbose=1)
predictions.shape

test_gen.class_indices

df_preds = pd.DataFrame(predictions, columns=['no_tumor_tissue', 'has_tumor_tissue'])

df_preds.head()

y_true = test_gen.classes

# Get the predicted labels as probabilities
y_pred = df_preds['has_tumor_tissue']

from sklearn.metrics import roc_auc_score

roc_auc_score(y_true, y_pred)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(fileconf)

test_labels = test_gen.classes

test_labels.shape

cm = confusion_matrix(test_labels, predictions.argmax(axis=1))

test_gen.class_indices

cm_plot_labels = ['no_tumor_tissue', 'has_tumor_tissue']

plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')

from sklearn.metrics import classification_report

# Generate a classification report

# For this to work we need y_pred as binary labels not as probabilities
y_pred_binary = predictions.argmax(axis=1)

report = classification_report(y_true, y_pred_binary, target_names=cm_plot_labels)

print(report)
