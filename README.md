# Deep_Learning_on_ImageNet_dataset
This repository downloads images from ImageNet website and trains a simple feed-forward neural network to classify them. The "hw02_ImageNet_Scrapper.py" gets the requested images from the web and downsamples them to 64 by 64 pixels. This script works based on a parser argument that gets the subclass list, the number of images in each subclass, root to the main directory that contains the main class folders, main class list, and path of the directory that contains the file "imagenet_class_info.json" as inputs. "hw02_imagenet_task2.py" includes the data loader and a simple neural network containing 2 hidden layers for the classification of the downloaded images. This script runs by providing the root to the main directory that contains the main class folders and main class list as its parser arguments, saves all the weights of the trained network as a pickle file, and outputs the loss at each epoch and classification accuracy on the test set. Although the attached codes work for general cases, the obtained results are for the following dataset: <br>
The main classes are cat and dog. <br>
Training set: 200 images of "Persian cat", Burmese cat", "Siamese cat", "hunting dog", "sporting dog", and "shepherd dog" each. <br>
Test set: 100 images of "domestic cat", "alley cat", "police dog", and "working dog". <br>

## Reference:
https://github.com/johancc/ImageNetDownloader 
