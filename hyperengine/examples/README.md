# Examples

All examples are self-contained to ease understanding.

#### 1. Getting Started
 - [**Hello World**](1_1_hello_word_with_mnist.py):
 a very simple example to run your existing model in Hyper-Engine environment.
 - [**Getting started with tuning**](1_2_getting_started_with_tuning.py):
 tuning a single parameter (`learning_rate`) to find the best value for
 a simple CNN.
 - [**Saving best models**](1_3_saving_best_models_mnist.py):
 tuning several hyper-parameters and saving the best CNN models on the disk.
 - [**Fine-tuning the saved model**](1_4_fine_tuning_saved_model.py):
 training the selected model further to squeeze the highest possible accuracy out of it.
 - [**Learning curve prediction**](1_5_learning_curve_prediction.py):
 optimizing the process with the learning curve prediction.

#### 2. Convolutional Neural Networks
 - [**Exploring CNN architectures to beat MNIST record**](2_1_cnn_mnist.py):
 exploring and tuning all possible variations of the CNN design to get
 to the top accuracy for the MNIST dataset.
 - [**Exploring CNN architectures for CIFAR-10 dataset**](2_2_cnn_cifar.py):
 exploring different variations of the CNN design to get good accuracy for the CIFAR-10 dataset
 (state-of-the art accuracy would require a bit more efforts - in the next examples).
 
   **Note**: this example needs `tflearn` library for fetching CIFAR-10.
