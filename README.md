# Activity-1-Prediction-with-Back-Propagation-and-Linear-Regression
Neural and Evolutionary Computation (NEC)

Part 2: Implementation of BP
The most important part of this activity is the programming of a neural network with back propagation from scratch. Here are some instructions on how to code the project.
•
You should create a project in a github account where you are going to develop the activity. You should do regular commits to this project so we can observe the evolution of the project from the initial file to the final delivery. We will apply a penalization to the grade if there is one single commit in the project.
•
We recommend the implementation of the project using Python (version >= 3.6). We are providing a base code that can be used to start the project (MyNeuralNetwork.py).
•
In this part you must implement all the methods necessary for the network to learn, you cannot use external libraries that already implement BP or Neural Networks.
•
The implementation must be based on the algorithm and equations in document [G] of Unit 3 at Moodle.
•
The implementation must use the following variables to hold all the information about the structure of the multilayer neural network and the BP:
o
L: number of layers
o
n: an array with the number of units in each layer (including the input and output layers)
o
h: an array of arrays for the fields (h)
o
xi: an array of arrays for the activations (ξ)
o
w: an array of matrices for the weights (w)
o
theta: an array of arrays for the thresholds (θ)
o
delta: an array of arrays for the propagation of errors (Δ)
o
d_w: an array of matrices for the changes of the weights (δw)
o
d_theta: an array of arrays for the changes of the weights (δθ)
o
d_w_prev: an array of matrices for the previous changes of the weights, used for the momentum term (δw(prev))
o
d_theta_prev: an array of arrays for the previous changes of the thresholds, used for the momentum term (δθ(prev))
o
fact: the name of the activation function that it will be used. It can be one of these four: sigmoid, relu, linear, tanh.
•
For example, the weight 𝑤𝑖𝑗(𝐿) between unit j in layer L-1 and unit i in layer L is accessed as w[L][i,j]
•
The idea behind this structure is that the code must be able to deal with arbitrary multilayer networks. For example, a network with architecture 3:9:5:1 (4 layers, 3 input units, 1 output unit, and two hidden layers with 9 and 5 units, respectively), would have n=[3; 9; 5; 1], and xi would be an array of length 4 (one component per layer), with xi[1] and array of real numbers of length 3, xi[2] and array of real numbers of length 9, xi[3] and array of real numbers of length 5, and xi[4] and array of real numbers of length 1. Similarly, w[2] would be an array 9x3, w[3] an array 5x9, and w[4] and array 1x5; w[1] is not used.
•
Additionally, the use of this structure, name conventions and array dimensions make it easy to convert the equations into code.
•
The code will receive one input dataset, and using the percentage of data that is passed as a parameter in the class constructor, should divide this dataset into training and validation. If the percentage is 0, then we consider that there is no validation, and all the input data is used for training.
•
The class MyNeuralNetwork will receive all these parameters in the class constructor:
o
Number of layers
o
Number of units in each layer
o
Number of epochs
o
Learning rate and momentum
o
The selected activation function (sigmoid, relu, linear, tanh)
o
The percentage of data that should be used as the validation set
•
The class MyNeuralNetwork will provide three public functions that can be called externally:
o
A function fit(X, y) that has 2 parameters: an array X of size (n_samples, n_features), which holds the training samples represented as floating point feature vectors; and a vector y of size (n_samples), which holds the target values (class labels) for the training samples. This method allows us to train the network with this data.
o
A function predict(X) that has 1 parameter, an array X of size (n_samples, n_features) that contains the samples. This method returns a vector with the predicted values for all the input samples.
o
A function loss_epochs() that returns 2 arrays of size (n_epochs, 2) that contain the evolution of the training error and the validation error for each of the epochs of the system, so this information can be plotted.
