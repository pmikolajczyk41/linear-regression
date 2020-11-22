## Linear regression

### Implementation
The project was implemented as a modular framework for linear regression. It provides basic tools (such as regularization or standardization) in the form of flexible abstractions, thanks to which it is very easy to combine individual modules and extend the existing ones. The implementation does not use any external libraries, it is a pure Python (version 3.6) code.

#### Normalization
Three classical variants of normalization are provided:
 - MIN_MAX_1: `(x - min) / (max - min)`
 - MIN_MAX_2: `(x - mean) / (max - min)`
 - STANDARD: `(x - mean) / stdev`

#### Basis functions
By default two sets of basis functions are available:
 - identity ( &phi;<sub>i</sub>(x) = x<sub>i</sub> )
 - monomials of degree 2 or less ( x<sub>i</sub>, x<sub>i</sub><sup>2</sup>, x<sub>i</sub>x<sub>j</sub> )
 
#### Loss functions
Two common loss funcions are provided:
 - square loss
 - absolute loss
 
 
#### Gradient descent
##### Initial hypothesis
Initial hypothesis is generated from the normal distribution with mean 0 and standard deviation being one of the hyperarameters.

##### Stop conditions
The algorithm checks three conditions on an ongoing basis (provided before the training process):
 - number of iterations
 - difference of consecutive errors
 - difference of consecutive gradients
 
##### Step
One of the hyperparameters.
