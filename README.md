# Classifying butterfly species in MATLAB

## Introduction

*Deep learning* is a branch of machine learning that teaches computers to do what comes naturally to humans: learn from experience. Deep learning uses neural networks to learn useful representations of features directly from data. Neural networks combine multiple nonlinear processing layers, using simple elements operating in parallel and inspired by biological nervous systems.

The purpose of this work is to build an architecture of a neural network that can learn to correctly recognize and classify the family to which a butterfly belongs. To do this, we will train the network using the dataset '*Butterfly & Moths Image Classification 100 species*'. \
You can view and download the dataset at the following: [link](https://www.kaggle.com/datasets/gpiosenka/butterfly-images40-species).

The *Deep Learning Designer* tool, belonging to the *Deep Learning Toolbox™* family of tools available within the *MATLAB* programming language, will be used to build such a network. *Deep Learning Toolbox™* provides simple MATLAB® commands to create and interconnect the layers of a deep neural network.

## Data exploration and preparation

The dataset used at the network implementation consists of 4500 butterfly images, divided into 50 different classes each representing one species. All images are in color, size 224 x 224 x 3, in jpg format. Shown below are 25 of the 50 different families of butterflies. Specifically they are respectively: *African Giant Swallowtail*, *AN 88*, *Arcigera Flower Moth, Atlas Moth*, *Banced Peacock*, *Becker white*, *Black Hairstreak*, *Blue Spotted Crow*, *Brown Argus*, *Cabbage White*, *Chalk Hill Blue*, *Chestnut*, *Clearwing Moth*, *Clodius Parnassia*, *Comet Moth*, *Common Wood-Nymph*, *Crecent*, *Danaid Eggfly*, *Easter Dapple White*, *Elbowed Pierrot*, *Garden Tiger Month*, *Glittering Sapphire*, *Great Eggfly*, *Green Celled Cattleheart* and *Grey Cloak*.

![species](pictures/classes.png?msec=1672791245723)

The entire dataset is located in the folder *butterflies*, within it are 50 subfolders, one for each different species, and within these are 90 images of butterflies all belonging to the same family.

First, a variable *path_to_images* was created in which a string representing the pathway to the *butterflies* folder having the images was allocated.

```matlab
path_to_images = "dataset/butterflies"
```

From this, what is referred to in the MATLAB language as the *ImageDatastore* was generated. To do this we made use of the *imageDatastore()* function, specifying both the path from where to get the pictures and to assign to each picture the label corresponding to the name of the subfolder to which it belongs.

```matlab
image_datastore = imageDatastore(path_to_images, "IncludeSubfolders",true,"LabelSource","foldernames")
```

![imds](pictures/imds.png?msec=1672791245703)

At this point, the train, validation and test datasets were generated. Use was made of the *splitEachLabel()* function to split each class following a percentage given as input; in addition, it was specified that images be taken randomly in the generation of the three sets.

```matlab
[train, validation, test] = splitEachLabel(image_datastore,0.7, 0.15, 0.15, 'randomized')
```

For each folder, 70 percent of the images were chosen to pull the network, 15 percent to validate it, and the remaining 15 percent to test it. The idea is to initially use the validation set to find the best configuration of the hyperparameters (tuning phase) to increase the performance of the network (avoid underfitting and overfitting, have optimal accuracy curve growth and optimal loss curve descrescence, have good metrics, etc.), and then to use the test set to give an overall evaluation of the optimized network.

![train](pictures/train.png?msec=1672791245703)
![validation](pictures/validation.png?msec=1672791245703)
![test](pictures/test.png?msec=1672791245704)

This just described is one of the most famous and most widely used validation methods. Another possible approach in Deep Learning Toolbox is to initially split the dataset into train and test sets and then, during the training phase, select a percentage of the train images to validate the model. However, given the fact that we have '*randomized*' the splitting of the imageDatastore, the two approaches are very similar to each other.

## Problem Designing and Network Architecture

As mentioned earlier, we want to construct a network that can recognize and classify the species of each butterfly from an image of it. The technique that will be used for the construction of such a network is that of *Transfer Learning*. In simple words, this technique consists of 'reusing' an already pre-trained network. The basic idea of transfer learning is to, as the name suggests, transfer knowledge gained from a network from one dataset to a new target dataset. There are three possible approaches in using transfer learning:

- Inductive: in which an attempt is made to adapt a previously trained model with a supervised approach to a new annotated dataset.
- Transudditional: in which an attempt is made to fit a model previously trained with a supervised approach to a new unannotated dataset.
- Non-supervised: in which we try to fit a previously trained model with non-supervised approach to a new non-annotated dataset.

In our case we use an inductive approach, in that we will pass annotated images to the network with their corresponding classes. From a more technical point of view, once the network to be reused is chosen, it will be re-trained by '*freezing*' the first layers and replacing the last layers (often fully-connected layers) with new layers adapted to the new target. This procedure is also known in the literature as *fine-tuning*. The big difference between 'classical' training and fine-tuning lies in the setting of the initial weights which, while in the classical case they are chosen randomly and then updated during back propagation, in this procedure they are not chosen randomly but taken from the pre-trained neural network.

The network that was chosen for this task is GoogLeNet. GoogLeNet is a 22-layer deep convolutional neural network that is a variant of Inception Network, a deep convolutional neural network developed by Google researchers.

![architecture](pictures/googlenet.png?msec=1672791245706)

The main feature, and at the same time what immediately jumps out at you, is the repetition of three sequences of inception layers, named with the numbers '3', '4' and '5'. The central sequence '4' is the one with the most body in terms of inception layers, in fact it is composed of 5 of them; The other two 'outermost' sequences, '3' and '5', are composed of the only two inception layers. The structure of the network, from a visual point of view, thus appears almost symmetrical, with the central body formed by the just-mentioned sequences and with the two 'tails' having the input layer and the output layer as extremes.

![architecture](pictures/architecture.png?msec=1672791245704)

Let us now analyze in detail the various individual components of the network.

![matgoogle](pictures/matlab_googlenet.png?msec=1672791245704)

As in all concolutional networks, the first layer is the input layer. The input layer of the GoogLeNet architecture accommodates an RGB image of size 224x224x3 and then scales it by centering its average in zero. Next there are two convolutional layers: the first uses a 7x7 filter with stride 2, while the second uses a 3x3 filter with the same stride. The main purpose of this layer is to reduce the input image, but trying not to lose information by using such a large filter compared to the others in the network. The input image size is reduced by a factor of four at the second conv layer and a factor of eight before reaching the first inception module, but a larger number of feature maps are generated. The second convolutional layer has a depth of two and leverages the 1x1 conv block, which as the effect of dimensionality reduction, which allows the decrease of computational load by lessening the layers' number of operations.

As mentioned earlier, the main body of googlenet consists of nine incipit modules divided into three sequences. Bounding each sequence from the next are the MaxPooling layers, whose job is to resample the input as it passes through the network. This is done by reducing the height and width of the input data. In addition, this size reduction is another effective way to reduce the computational load on the network.

We focus attention on the inception modules. It is precisely the particular use of these modules that are the distinguishing feature of GoogLeNet. In fact, the inception modules present within the structure of the network preexist small differences from their classical version.

![inception](pictures/inception_differences.png?msec=1672791245704)

As is easy to see, the main difference lies in the addition of 1x1 convolutional layersbefore the canonical 3x3 and 5x5 convolutional layers and after the 3x3 max-pooling layer. The task of these layers is to reduce the computational cost by a certain amount. The task of these layers is to reduce the computational cost by a large amount. In fact, if we calculate the computational cost of the inception module in its native version, we realize how the use of such a module is very expensive, with a number of parameters close to 120 million. The addition of this 1x1 convolutional layer, reducing the number of input tensor channels, trastically lowers the number of total paramenters to about 12 million.

In the final 'tail' of GoogLeNet, there are sequentially: an average pooling layer, a dropout layer, a fully connected layer, a softmax layer, and finally a classification layer. Both the fully-connected layer and the classification layer are the ones that will be replaced in the fine-tuning phase to adapt the network to the new dataset.

![finetuning](pictures/fine_tuning.png?msec=1672791245706)

Initially, the average pooling layer takes an average of all feature maps produced by the last inception module and reduces the height and width of the input to 1x1. Next, a 40% dropout layer is applied to avoid a possible overfitting problem by randomly reducing the number of interconnected neurons within a neural network. At this point, we arrive at the fully-connected layer, which initially had an output of 1,000 classes, the same as the dataset with which the network was trained. The replacement layer for this was sethaving an output equal to 50, such as the different families of butterflies in the imageDatastore. The same fate befell the classification layer, which was replaced with one having the same number of input classes as 50.

Information about the new network architecture, including both the layers and the links between them, was allocated in the variable *lgraph*. The matlab file containing this implementation is available at: [link](network_architecture.m)

## Training phase

This is the stage where the model is trained and evaluated, looking for the best hyperparameter configuration. First of all, it should be specified that in order to train the network, it was not necessary to shrink the size of the training and validation images, as they were already in the required format: 224x224x3. Moreover, due to the large amount and variety of images available for each butterfly class/species, it was not necessary to apply any DataAugmentation techniques; in fact, we will see in a moment that, without having applied such techniques, good results were obtained immediately.

After several attempts to change the configuration of all hyperparameters to maximize accuracy and minimize error, again avoiding overfitting problems, the network was trained by setting:

```matlab
opts = trainingOptions("sgdm",...
    "ExecutionEnvironment","auto",...
    "InitialLearnRate",0.01,...
    "MaxEpochs",20,...
    "MiniBatchSize",64,...
    "Shuffle","every-epoch",...
    "ValidationFrequency",70,...
    "Plots","training-progress",...
    "ValidationData",augimdsValidation)
```

Once we have set the hyperparameters for training, we make use of the *trainNetwork()* function to tow the network, specifying the datastore containing the images with which the network will be trained, the architecture of the network (saved in the *lgraph* variable), and the hyperparameters just chosen.

```matlab
[net, traininfo] = trainNetwork(augimdsTrain,lgraph,opts)
```

Two new variables will be generated from the execution of the previous code:

- *net*, a variable of type DAG-Natwork (Directed Acyclic Graph), in which the entire neural network already trained will be allocated.
- *traininfo*, a variable in which all the information and statistics inherent in the training phase of the network are allocated.

The following are the final results of the training phase.

![curves](pictures/learning_curves.png?msec=1672791245706)

![table](pictures/table.png?msec=1672791245707)

It can be seen that the accuracy and loss curves remained almost constant during the first ten epochs, meaning that our model failed to capture the 'distinctive details' of each butterfly species until the tenth epoch. From this onwards we see how the two curves propagate through the iterations drawing an optimal curve, i.e. the model has learnt to recognise these distinctive features. Furthermore, we can see that no overfitting problem occurred; this fact is also confirmed by observation of the table, for there was no persistent period of iterations in which while the accuracy curve calculated on the train data continued to increase, that calculated on the validation data steadily decreased.

We can therefore conclude that the training phase of the model was successful and yielded good results:

| \   | accuracy | loss |
| --- | --- | --- |
| **Training** | 96.88% | 0.0488% |
| **Validation** | 93.85% | 0.2745% |

## Testing phase and results

So we have now reached the phase where, after training the model, we need to test it. To do this, we make use of the *classify()* function, specifying the trained network and the image datastore that the network has never seen.

```matlab
pred_test_labels = classify(net, test)
```

At this point in the we evaluate the accuracy of our prediction.

```matlab
true_test_labels = test.Labels
accuracy_test = mean(true_test_labels == pred_test_labels)
```

![acctest](pictures/accuracy_test.png?msec=1672791245704)

Running the code we receive an accuracy of **92%** in output.

The following graph is called the 'confusion chart', it is the graphical representation of the confusion matrix, i.e. that matrix which counts the number of instances correctly classified by the model; The confusion chart creates a confusion matrix graph from the trueLabels and predictedLabels and returns a ConfusionMatrixChart object. The rows of the confusion matrix correspond to the true class and the columns to the predicted class. Diagonal and non-diagonal cells correspond to correctly and incorrectly classified observations, respectively. The numbers on the rows and columns in the graph represent the ordinal number as a factor of the categorical variable.

```matlab
C = confusionmat(true_test_labels, pred_test_labels);
confusionchart(C)
```

![conf](pictures/confchart.png?msec=1672791245704)

From the analysis of the confusion chart, we see that the model succeeds more or less evenly in correctly classifying the species of each butterfly. However, when looking at line number 9, corresponding to the species 'Brown Argus', the model found it difficult to distinguish this species; in fact, out of thirteen instances, the model only managed to classify seven correctly, exchanging it three times, i.e. half the times it got it wrong, with variable 11, corresponding to the species 'Chalk Hill Blue'.

We are, however, obliged to note the following: the model did, on three occasions, confuse a butterfly of the species 'Brow Argus' with one of the type 'Chalk Hill Blue', but it is also true that it almost always recognised (twelve times out of thirteen) butterflies of the latter species.

| Brown Argus | Chalk Hill Blue |
| --- | --- |
| ![alt](pictures/001.jpg?msec=1672791245705) | ![alt](pictures/1.jpg?msec=1672791245705) |

Seeing the two images, we might assume that the model was confused by the shape and style of the wings of the two species, thus focusing on the first two features rather than others, such as colour.

We can now conclude our work by again analysing the results obtained in both the training and testing phases, and can state that our model, especially thanks to the distinctive features (inception modules and 1x1 convolutions above all) of the googlenet network, has achieved its goal of classifying the species of the butterfly classes in the dataset with high precision.
