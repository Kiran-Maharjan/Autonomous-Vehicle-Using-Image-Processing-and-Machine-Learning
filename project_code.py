import numpy as np
import scipy as sp
from scipy import io
from scipy.optimize import minimize
from functions_used import *

# initializing layers of the network
input_layer_size = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25  # 25 hidden units
num_labels = 10  # 10 labels, from 1 to 10. Remember that 0 is mapped to 10

# loading weights

print("Loading neural network parameters \n")

params = sp.io.loadmat('ex4weights')
# print(len(params))
# print(type(params))
# print(params.keys())
th1 = params.get('Theta1')  # (25,401)
th2 = params.get("Theta2")  # (10,26)
# print("the dimension of theta1 is : ", th1.shape, "and type is : ", type(th1))
# print("dimension of first element of theta1: ", len(th1[0]))
# print("dimension of theta2: ", th2.shape)
# print("dimension of first element of theta2: ", len(th2[0]))

# Unrolling parameters
un_th1 = th1.reshape(10025, 1)
# print('dimension of un_th1: ',un_th1.shape)
un_th2 = th2.reshape(260, 1)
un_params = np.append(un_th1, un_th2)  # don't use np.array([un_th1, un_th2]} for some reasons you can't reshape as
# (10285,1) later even though it worked on smaller numpy arrays in numpy_test_file in lines 279 to 287.
# print(type(un_params))
# print(un_params.shape)
# print(un_params[10266])
'''
print([0].shape)
print(type([0]))
print([1].shape)
print()
'''

# loading data

print("Loading training data.....")

loaded_data = sp.io.loadmat('ex4data1')
# print(type(loaded_data))
# print(loaded_data.keys())

z = loaded_data['y']
y = loaded_data['y'].flatten()  # use flatten to get only one dimension
# y has shape (50000,)
y = (y - 1) % 10  # hack way of fixing conversion MATLAB 1-indexing

# print('shape of z: ', z.shape)
# print('Shape of y is: ', y.shape)
# print('Data-type of y is: ', type(y))
# print('Printing y[4]: ', y[4])
# print('Printing z[4]: ', z[4])
# print('Hence it is better to use y with flatten().\n\n')

X = loaded_data['X']  # (5000,400)

# print('Data-type of X is: ', type(X))
# print('Shape of X is: ', X.shape)
# print('Printing X[53][8]: ', X[53][8]) # Seen data from matlab X(54,9)
# print('Printing X[3669][6]: ', X[3669][6])
# print(X.dtype)
# print('hello')

'''
m = len(y)
ones = np.ones((m,1)) 
d = np.hstack((ones,X))
a1 = d[0][:,None]

print(a1.dtype)
'''
'''
params = sp.io.loadmat('ex4weights')
th1 = params.get('Theta1')
th2 = params.get("Theta2")
print(th2.shape)
print(th2[:,0].shape)
'''

# Compute Regularized Cost
print("Checking cost function with regularization...")
reg_param = 1.0  # This is represented by Lambda in the nnCostFunction
reg_cost, g = nnCostFunction(un_params, input_layer_size, hidden_layer_size, num_labels,
                             X, y, reg_param)
np.testing.assert_almost_equal(reg_cost, 0.383770, decimal=6,
                               err_msg="Regularized Cost incorrect.")
print("The calculated cost is: ", reg_cost)

# Checking sigmoid gradient
print("Checking sigmoid gradient...")
vals = np.array([1, -0.5, 0, 0.5, 1])
g = sigmoid_gradient(vals)
np.testing.assert_almost_equal(0.25, g[2], decimal=2, err_msg="Sigmoid function incorrect")

# Initialize neural network parameters
print("Initializing neural network parameters...")
initial_theta1 = randInitializeWeights(input_layer_size + 1, hidden_layer_size)
initial_theta2 = randInitializeWeights(hidden_layer_size + 1, num_labels)

# Unroll parameters
initial_nn_params = np.append(initial_theta1,initial_theta2).reshape(-1)

reg_param = 0.0
initial_cost, g = nnCostFunction(initial_nn_params,input_layer_size,
                                 hidden_layer_size,num_labels,X,y,reg_param)

print("The initial cost after random initialization: ", initial_cost)

# Train NN Parameters
reg_param = 3.0
def reduced_cost_func(p):
    """ Cheaply decorated nnCostFunction """
    return nnCostFunction(p,input_layer_size,hidden_layer_size,num_labels,
                          X,y,reg_param)

print("Now training the netowrk......")
results = minimize(reduced_cost_func,
                   initial_nn_params,
                   method="CG",
                   jac=True,
                   options={'maxiter':50, "disp":True})

fitted_params = results.x
# Reshape fitted_params back into neural network
theta1 = fitted_params[:(hidden_layer_size *
             (input_layer_size + 1))].reshape((hidden_layer_size,
                                       input_layer_size + 1))

theta2 = fitted_params[-((hidden_layer_size + 1) *
                      num_labels):].reshape((num_labels,
                                   hidden_layer_size + 1))

predictions = predict(theta1, theta2, X)
accuracy = np.mean(y == predictions) * 100
print("Training Accuracy with neural network: ", accuracy, "%")
