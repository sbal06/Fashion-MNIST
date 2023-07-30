# Description of Fashion-MNIST
This is where you will find different Python scripts (code) for the Fashion MNIST dataset. The Fashion-MNIST dataset is a collection of images depicting different types of clothing items. You can learn more about the fashion MNIST dataset [here](https://www.tensorflow.org/datasets/catalog/fashion_mnist). The Fashion-MNIST dataset was introduced to serve as a replacement for the original [MNIST](https://www.kaggle.com/datasets/avnishnish/mnist-original) dataset, which was released in 1999. The originial MNIST dataset became overused due to many models achieving 99% accuracy with a simple acrhitecture and lack of noise in the training/testing samples. The Fashion-MNIST dataset was designed to provide a more challenging task for machine learning models to perform well on the testing data.

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
![250211420-019424d0-c197-4c04-9ed8-446ad9acf09e](https://github.com/sbal06/Fashion-MNIST/assets/101956177/276bbd93-02ae-4491-98d5-b11e3462238d)
<br>

# Notebooks

### Other Classification Algorithms (ClassificationAlgorithms.ipynb)
K-Nearest Neighbors, Decision Trees on the fashion MNIST images. <br>
- [Other Classification Algorithms](https://github.com/sbal06/Fashion-MNIST/blob/main/ClassificationAlgorithms.ipynb) <br>

### Convolutional Neural Network (FashionMNISTDataSet.ipynb)
A convolutional neural network on the fashion MNIST images.
- [Tensorflow Convolutional Neural Network](https://github.com/sbal06/Fashion-MNIST/blob/main/FashionMNISTDataSet.ipynb) <br>

### Fashion-MNIST webapp image input (InputImage.ipynb)
This is the code I used to take an image, and convert it to grayscale for the Fashion-MNIST Web App I created (different repository).
-[ImageInput](https://github.com/sbal06/Fashion-MNIST/blob/main/InputImage.ipynb) <br>

# Results
### K-Nearest Neighbors, Decision Trees, Basic Neural Network 
For finding the optimal value of the number of nearest neighbors in the KNN algorithm, I implemented a GridSearchCV to tune the KNN's hyperparameters in a certain range. <br>

The neural network had four Dense layers, with 1024, 512,  512, and 10 neurons, and each followed by BatchNormalization to normalize the outputs to have a mean of zero and variance of one. 

Test accuracy for KNN: 83.95% <br>
Test accuracy for Decision Trees: 78.39% <br>
Test accuracy for Neural Network: 86% <br>

### CNN
-----------------------------------------------------------------------------------------------------------------------------------------
For the Convolutional Neural Network, I used 3 Convolutional 2D layers, followed by Batch Normalization after the first and third layers to normalize the inputs to the MaxPooling layer. For the neural network, I used two Dense layers followed by Batch Normalization.

As shown in the training and testing results of the model during each epoch, the model is able to reach an accuracy of 90% after 4 epochs, but staggers around 91-92% for the remainder of testing. 

If we take a look at the confusion matrix, 
<p align = "center" >
<image src = "https://github.com/sbal06/Fashion-MNIST/assets/101956177/cd98406a-68a1-48ba-a622-325ed480d60e" width = "400" height = "400">
</p>
we can see that the model has diffculty in distinguishing between 'T-shirt/Top' and a 'Shirt.'


Score for training accuracy: 90-91% <br>
Scores for testing accuracy: 91.5% <br>

----------------------------------------------------------------------------------------------------------------------------------------
### Validation Accuracies for Basic Neural Network and CNN

Here are the training and testing accuracies for both the basic neural network and the CNN, respectively:
<p align = "center" >
 <image src = "https://github.com/sbal06/Fashion-MNIST/assets/101956177/997e764b-73fc-4ede-9298-46db2c96833c" width = "400" height = "400">
 <image src = "https://github.com/sbal06/Fashion-MNIST/assets/101956177/86d1ed74-87ce-403b-9a92-7eb2a513b653" width = "400" height = "400">
</p>

 For the basic neural network (image on the left), the training and testing accuracies are relatively similar and increase (more or less) when the number of epochs reaches 20. 
 
 However, even though the CNN (image on the right) produces a higher accuracy, the testing accuracy is always below the training accuracy, and changing the number of epochs in the CNN alters the testing accuracy.



# Installation/How to run the project
Follow these steps to set up the project and install the required dependencies:

1. **Open the CMD (Command Prompt) on your computer**
2.  **Clone the repository**: `git clone https://github.com/sbal06/Fashion-MNIST.git`
3. **Change into the project directory**:  `cd Fashion-MNIST`
4. **Create a virtual environment** (optional): `python -m venv "name"`
5. **Activate the virtual environment**:
    - On macOS and Linux
      `source "Name you choose"/bin/activate`
    - On Windows
      `.\"Name you choose"/bin/activate`
      
6. **Install the project Dependencies located below.**

Remember to replace the "name" with your preferred name for the virtual environment. Learn more about virtual environments [here](https://www.dataquest.io/blog/a-complete-guide-to-python-virtual-environments/).

## Dependencies
### Python distribution
- [Anaconda](https://www.anaconda.com/blog/upcoming-releases-anaconda-distribution-2023-03-and-beyond)
### Google Colaboratory
Google Colab is cloud-based, and you get automatic-version control; there is no need to download or install any packages.
### Jupyter Notebook
- [numpy](https://numpy.org/): `conda install numpy` <br>
-  [sklearn](https://scikit-learn.org/stable/): `conda install scikit-learn` <br>
- [matplotlib](https://matplotlib.org/): `conda install matplotlib` <br>
- [seaborn](https://seaborn.pydata.org/): `conda install seaborn` <br>
#### Make sure the Python and Numpy versions are compatible in Jupyter notebook.

### For deep learning models
It is probable to run into errors installing. If so, please watch [here](https://www.youtube.com/watch?v=CrEl8QL8hsM) <br>
- [Tensorflow](https://www.tensorflow.org/) `conda install tensorflow` <br>
- [Keras](https://keras.io/)  <br>






# Contributions
Contributions to this repository are welcome! If you have any improvements, bug fixes, or additional scripts, feel free to submit a pull request.











