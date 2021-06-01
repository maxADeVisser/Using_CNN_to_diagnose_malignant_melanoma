# Convolutional Neural Network

#### *How can a convolutional neural network be used to diagnose malignant melanoma?*

##### **by Simon Hindsgaul :medal_military: Max de Visser :medal_military: William Dyrnesli :medal_military: Sebastian Rohr** :medal_military:

# Setup
To run a training on our convolutional neural network you will need to download a number of things. 
The most crucial thing required is a computer with a GPU.
We will also preface by saying that this is a rather tedious process that requires multiple software installations.

We are using python 3.8. The following downloads and versions of the required programs are listed in accordance with this.
If you are using a newer or older version of python you can use this link: https://www.tensorflow.org/install/source#gpu to check the compatibility of your version of python.

NOTE: make sure your graphics cards' drivers are up to date. 
The programs you will need to download to enable TensorFlow 2.4.1 to run on your GPU are:
1. CUDA toolkit 11.0 - https://developer.nvidia.com/cuda-toolkit-archive
1. CuDNN version 8.0 - https://developer.nvidia.com/cudnn (getting CuDNN requires you to become a member of the NVIDIA developer program) 

The next thing you will need is the command prompt known as Anaconda prompt.
- Anaconda prompt: https://docs.anaconda.com/anaconda/install/

After you have finished installing the anaconda prompt, you simply need to go to the requirements.txt in our repository and follow the instructions there. Once you are done following those instructions and you have your virtual environment with TensorFlow installed, you have to download the data for the ConvNet.

The data for the ConvNet can be found through this link: https://www.kaggle.com/c/siim-isic-melanoma-classification/data?select=jpeg
The data is from a kaggle competition called "SIIM-ISIC Melanoma Classification". You will only need to download the file with jpeg images, as that is the only data our ConvNet takes as input. Once the images are downloaded you are ready to run the "process_data.py" file, which separates the images into training, validatio, and test folders. Congratulations! You are ready to train your first ConvNet on your computer using the "train_data.py" file. You are welcome to modify and redistribute the program :) 


