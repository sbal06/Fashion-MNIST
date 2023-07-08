# Description
Welcome! This is where you will find different Python scripts (code) for the Fashion MNIST dataset. The Fashion-MNIST dataset is a collection of images depicting different types of clothing items. You can learn more about the fashion MNIST dataset [here](https://www.tensorflow.org/datasets/catalog/fashion_mnist). The Fashion-MNIST dataset was introduced to serve as a replacement for the original [MNIST](https://www.kaggle.com/datasets/avnishnish/mnist-original) dataset, which was released in 1999. The originial MNIST dataset became overused due to many models achieving 99% accuracy with a simple acrhitecture and lack of noise in the training/testing samples. The Fashion-MNIST dataset was designed to provide a more challenging task for machine learning models to perform well on the testing data.

# Dataset overview
The Fashion MNIST contains 70000 images, which are divided into the training and testing sets. The training set contains 60000 images and the testing set contains 10000 images. Each example is a 28x28 grayscale image associated from one of the ten classes below:


|      Clothing     |
|------------------  |
| 0 T-shirt/top <br>  |
| 1 Trouser <br>    |
| 2 Pullover <br>   |
| 3 Dress <br>      |
| 4 Coat <br>       |
| 5 Sandal <br>     |
| 6 Shirt <br>      |
| 7 Sneaker <br>    |
| 8 Bag <br>        |
| 9 Ankle boot <br>  |


The goal is to identify the correct clothing in the image.

![Plot-of-a-Subset-of-Images-from-the-Fashion-MNIST-Dataset-1024x768](https://github.com/sbal06/Fashion-MNIST/assets/101956177/019424d0-c197-4c04-9ed8-446ad9acf09e) <br>


# Notebooks

### Other Classification Algorithms ( KNN, Decision Trees)
K-Nearest Neighbors, Decision Trees on the fashion MNIST images. <br>
- [Other Classification Algorithms](https://github.com/sbal06/Fashion-MNIST/blob/main/ClassificationAlgorithms.ipynb) <br>

### Convolutional Neural Network
A convolutional neural network on the fashion MNIST images.
- [Tensorflow Convolutional Neural Network](https://github.com/sbal06/Fashion-MNIST/blob/main/FashionMNISTDataSet.ipynb) <br>

# Results
### CNN
-----------------------------------------------------------------------------------------------------------------------------------------
For the Convolutional Neural Network, I used 3 Convolutional 2D layers, followed by Batch Normalization after the first and third layers to normalize the inputs to the MaxPooling layer. For the neural network, I used two Dense layers followed by Batch Normalization.

As shown in the training and testing results of the model during each epoch, the model is able to reach an accuracy of 90% after 4 epochs, but staggers around 91-92% for the remainder of testing. 

If we take a look at the confusion matrix, ![download](https://github.com/sbal06/Fashion-MNIST/assets/101956177/c84adbd5-254b-4216-b7cb-07f3eb9dc022)


----------------------------------------------------------------------------------------------------------------------------------------
# Dependencies
### Python distribution
- [Anaconda](https://www.anaconda.com/blog/upcoming-releases-anaconda-distribution-2023-03-and-beyond)
### Jupyter Notebook or Google Coloboratory
- [numpy](https://numpy.org/): `conda install numpy` <br>
-  [sklearn](https://scikit-learn.org/stable/): `conda install scikit-learn` <br>
- [matplotlib](https://matplotlib.org/): `conda install matplotlib` <br>
- [seaborn](https://seaborn.pydata.org/): `conda install seaborn` <br>
#### Make sure the Python and Numpy versions are compatible in Jupyter notebook.

### For deep learning models
For more information, [watch](https://www.youtube.com/watch?v=CrEl8QL8hsM) <br>
- [Tensorflow](https://www.tensorflow.org/) `conda install tensorflow -gpu` <br>
- [Keras](https://keras.io/)  <br>


# Contributions
Contributions to this repository are welcome! If you have any improvements, bug fixes, or additional scripts, feel free to submit a pull request.

# License









