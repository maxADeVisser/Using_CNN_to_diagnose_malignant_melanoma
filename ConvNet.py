import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" #Removes messages from TensorFlow saying exactly what the software does

import tensorflow as tf
import math
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import roc_curve
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers

#Next two lines are used to check if the CNN is running on the GPU
#physical_devices = tf.config.list_physical_devices("GPU")
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

#Define how many images we have in each folder created by process_data.py
train_examples = 14793
validation_examples = 1915
test_examples = 1910
img_height = img_width = 224 #InceptionV3 feature vector 5 takes images in this format as input. This might change depending on what transfer learning model is used.
batch_size = 64 #How many images are in each batch? Increasing
class_weight = {0: 1, 1: 13.82} #Define the weight of our classes. 0 = benign, 1 = malignant

#Define model and extra layers
hub_url = "https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4" #The link to the trasfer learning model we use
model = keras.Sequential([ #Set model to sequencial so we can define layers one by one.
    hub.KerasLayer(hub_url, trainable=True), #Set trainable to true, so the transfer learning model trains the entire network on our data.

    #Adding our own layers onto the transfer learning model
    layers.Dense(16, kernel_regularizer=regularizers.l2(l2=0.005), activation='relu'),

    layers.Dropout(0.5),

    layers.Dense(1, activation="sigmoid"),
])

#Datagenerators for train, validation, and test data

#Data augmentation for training data
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255, #Turns the images' pixel values into numbers between 0 and 1.
    rotation_range=15,  #Rotate training images by 0-15%.
    zoom_range=(0.95, 0.95), #Zoom in/out on training images by 0-5%.
    horizontal_flip=True, #Randomly flip the training images horizontally.
    vertical_flip=True, #Randomly flip the training images vertically.
    brightness_range=[0.2, 1.0], #Darkens the image randomly by a % value between the defined numbers.
    data_format="channels_last", #channels_last for 2D data: (rows, cols, channels) - we use channels_last because we use keras. If we used Theano, which is another API supported by TensorFlow, it would be channels_first: for 2D data: (channels, rows, cols).
    dtype=tf.float32 #Define data type
)

#The next two lines should be identical to make sure train and val has the same augmentation
validation_datagen = ImageDataGenerator(rescale=1.0 / 255, dtype=tf.float32)
test_datagen = ImageDataGenerator(rescale=1.0 / 255, dtype=tf.float32)

#Load training data
train_gen = train_datagen.flow_from_directory(
    "data_lidtOversampled/train/",
    target_size=(img_height, img_width), #If the images are not 224 height and width, they are rescaled to that size.
    batch_size=batch_size, #Batch size for training data is equal to the batch size we defined in the beginning.
    color_mode="rgb", #This is the default setting, but we defined it to make it clear.
    class_mode="binary", #Define that we only have 2 classes
    shuffle=True, #mangler kommentar
    seed=123, #mangler kommentar
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

#Load metrics
METRICS = [
    keras.metrics.TruePositives(name='tp'),
    keras.metrics.FalsePositives(name='fp'),
    keras.metrics.TrueNegatives(name='tn'),
    keras.metrics.FalseNegatives(name='fn'),
    keras.metrics.BinaryAccuracy(name="accuracy"), #True positive + true negative / total samples
    keras.metrics.Precision(name="precision"),#True positives / true positives + false positives
    keras.metrics.Recall(name="recall"),#True positives / true positives + false negatives
    keras.metrics.AUC(name="auc"), #AUC means area under curve. The curve refers to an ROC curve made by the true positive rate and the false positive rate.
]

#Compile the model
model.compile(
    optimizer=keras.optimizers.Adam(lr=3e-4), #Learning rate optimizer
    loss=[keras.losses.BinaryCrossentropy()], #Loss fucntion
    metrics=METRICS,
)

#Start the training and define how it runs
model.fit(
    train_gen, #The model trains on the training data
    epochs=10, #The number of epochs the model trains for
    verbose=1, #How is the information about training in each epoch presented? Setting verbose=2 doesn't show a progress bar for the epoch like verbose=1 does.
    steps_per_epoch=train_examples // batch_size, #Defining how many steps per epoch using floor division between number of training images and the batch size
    validation_data=validation_gen,
    validation_steps=validation_examples // batch_size,
    class_weight=class_weight
    #callbacks=[keras.callbacks.ModelCheckpoint("name_of_your_model")], #This is how to load a saved model
)


#Making a method that prints an ROC curve
def plot_roc(labels, data):
    predictions = model.predict(data) #Predictions is equal to what the model predicts on the input data
    fp, tp, _ = roc_curve(labels, predictions) #the actual labels and the predictions for them
    plt.plot(100 * fp, 100 * tp) #Values for the x and y axis
    plt.xlabel("False positives [%]") #Label for the x axis
    plt.ylabel("True positives [%]") #Label for the y axis
    plt.show() #Print the model


#Create array to store test labels for ROC curve in.
test_labels = np.array([])
num_batches = 0

#For loop that appends the labels of the images to the array we just created until there are no images left
for _, y in test_gen:
    test_labels = np.append(test_labels, y)
    num_batches += 1
    if num_batches == math.ceil(test_examples / batch_size):
        break


#Evaluate returns the loss value and metrics for the model in validation/test mode.
model.evaluate(validation_gen, verbose=1)
model.evaluate(test_gen, verbose=1)

#Call the roc function
plot_roc(test_labels, test_gen)

#Save the model (you have to close the popup with the ROC curve before the code moves on to this)
model.save("second_project/Savedmodel_InceptionFV5_24.1")
