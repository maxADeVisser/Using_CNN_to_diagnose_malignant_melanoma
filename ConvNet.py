import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import math
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
import itertools
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers
#from keras.applications.inception_v3 import InceptionV3



#Next two lines are used to check if the CNN is running on the GPU
#physical_devices = tf.config.list_physical_devices("GPU")
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

#Define how many images we have for each class
train_examples = 14793
validation_examples = 1915
test_examples = 1910
img_height = img_width = 224  # Det er en standard når man bruger keras, at sætte billed width og height til 256, men det virker ikke. Så vi satte den til 224 i stedet lmao
batch_size = 64
class_weight = {0: 1,
                1: 13.82}

#Define model and extra layers
hub_url = "https://tfhub.dev/google/imagenet/inception_v3/feature_vector/5"
model = keras.Sequential([
    hub.KerasLayer(hub_url, trainable=True),
    layers.Dense(16, kernel_regularizer=regularizers.l2(l2=0.005), activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid"),
])

#Datagenerators

#Data augmentation for training data
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=15,  # kan evt. være op til 90
    zoom_range=(0.95, 0.95),
    horizontal_flip=True,
    vertical_flip=True,
    data_format="channels_last",
    dtype=tf.float32,
    brightness_range=[0.2, 1.0]
)

#The next two lines should be identical to make sure train and val has the same augmentation
validation_datagen = ImageDataGenerator(rescale=1.0 / 255, dtype=tf.float32)
test_datagen = ImageDataGenerator(rescale=1.0 / 255, dtype=tf.float32)

#Load training data
train_gen = train_datagen.flow_from_directory(
    "data_lidtOversampled/train/",
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode="rgb",
    class_mode="binary",
    shuffle=True,
    seed=123,
)

#Load validation data
validation_gen = validation_datagen.flow_from_directory(
    "data_lidtOversampled/validation/",
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode="rgb",
    class_mode="binary",
    shuffle=True,
    seed=123,
)

#Load test data
test_gen = test_datagen.flow_from_directory(
    "data_lidtOversampled/test/",
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode="rgb",
    class_mode="binary",
    shuffle=True,
    seed=123,
)

METRICS = [
    keras.metrics.TruePositives(name='tp'),
    keras.metrics.FalsePositives(name='fp'),
    keras.metrics.TrueNegatives(name='tn'),
    keras.metrics.FalseNegatives(name='fn'),
    keras.metrics.BinaryAccuracy(name="accuracy"),
    keras.metrics.Precision(name="precision"),#False positives (lægen siger man har kræft, men det har man ikke)
    keras.metrics.Recall(name="recall"),#False negatives (lægen siger man ikke har kræft, men det har man). Recall er klart mest vigtig for os.
    keras.metrics.AUC(name="auc"), #betyder Area under curve. Curve refererer til en graf lavet af TPR og FPR. AUC er den mest udbredte måde at sammenligne svar på.
]

model.compile(
    optimizer=keras.optimizers.Adam(lr=3e-4),
    loss=[keras.losses.BinaryCrossentropy()],
    metrics=METRICS,
)

model.fit(
    train_gen,
    epochs=10,
    verbose=1,
    steps_per_epoch=train_examples // batch_size,
    validation_data=validation_gen,
    validation_steps=validation_examples // batch_size,
    class_weight=class_weight
    #callbacks=[tensorboard_callback]
    # callbacks=[keras.callbacks.ModelCheckpoint("isic_model")], # How to load a saved model
)


# Laver en model der viser false positive og true positive
def plot_roc(labels, data):
    predictions = model.predict(data)
    fp, tp, _ = roc_curve(labels, predictions) #the actual labels and the predictions for them
    plt.plot(100 * fp, 100 * tp)
    plt.xlabel("False positives [%]")
    plt.ylabel("True positives [%]")
    plt.show()


test_labels = np.array([])
num_batches = 0

# Itererer gennem test generatoren
for _, y in test_gen:
    test_labels = np.append(test_labels, y)
    num_batches += 1
    if num_batches == math.ceil(test_examples / batch_size):
        break


model.evaluate(validation_gen, verbose=1)
model.evaluate(test_gen, verbose=1)
plot_roc(test_labels, test_gen)
model.save("second_project/Savedmodel_InceptionFV5_24.1")
