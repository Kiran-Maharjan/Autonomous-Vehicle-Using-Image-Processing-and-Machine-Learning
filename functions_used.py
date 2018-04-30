import numpy as np
from scipy.optimize import minimize


def sigmoid(z):
    g = 1.0 / (1 + np.e**(-z))
    return g


def sigmoid_gradient(z):
    return sigmoid(z)*(1-sigmoid(z))


def randInitializeWeights(ip_nodes, op_nodes):
    # here ip_nodes and op_nodes are no. of nodes at the input side and receiving(or output) side of the weights of connections.
    epsilon_init = 0.12
    rand_weights = np.random.uniform(-epsilon_init,epsilon_init,(ip_nodes,op_nodes))
    # or you could have used: rand_weights = (np.random.random(L_out, 1 + L_in)) * 2 * epsilon_init -  epsilon_init
    return rand_weights


def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda):
    # Note that 'lambda' is used in python to define anonymous functions. So use 'Lambda'
    no_of_weights1 = (input_layer_size+1)*hidden_layer_size
    theta1 = nn_params[:no_of_weights1].reshape((hidden_layer_size,input_layer_size+1))

    theta2 = nn_params[no_of_weights1:].reshape((num_labels,hidden_layer_size+1)) # this could also be done with negative index

    # to get no. of training examples
    m = len(y) # also equivalent to m = len(X[0])

    # Add column of ones to X
    ones = np.ones((m,1))
    newX = np.hstack((ones,X))  # Add a column of ones
    tempj = 0
    # c = np.zeros(num_labels)
    init_y = np.zeros((m,num_labels))
    # for i in range(m):
    #    init_y[i][y[i]] = 1

    DEL1 = np.zeros(theta1.shape) # or you can use: DEL1 = np.zeros_like(theta1)
    DEL2 = np.zeros_like(theta2)

    for i in range(m):
        a1 = newX[i][:,None] # used None to make shape from (401,) into (401,1)
        # using None above added one more dimension to the array. So now it will display vertically 401 elements.
        # Now due to none one more index for one more dimension is needed to specify particular element.
        # i.e. a1[0][0] is an element and a1[0] is an array of 1 element only.

        z2 = np.dot(theta1,a1)  # (25,401) and (401,1) give (25,1)
        a2 = sigmoid(z2)
        newa2 = np.vstack((np.ones(1),a2)) # to make (26,1) for extra bias unit like we did for X to get newX
        # in newX we had to add bias for all of the 'm' training examples. but not here.
        # so np.ones(1) is used which an array of single element 1 (i.e. [1]) with shape (1,)
        # here vstack used to add a row of [1]

        z3 = np.dot(theta2,newa2) # (10,26) and (26,1) give (10,1)
        h = sigmoid(z3)
        # thus feed forward has been completed

        # now we need to calculate cost.
        # note that in ex4data y is set to 10 to represent 0. So, if y[i] = 10 it is actually 0.
        # c[0] = 0
        # for j in range(1,10):
        #    c[j] = j

        # if (y[i] == 10):
        #    y[i] = 0

        # newy = (c == y[i])
        #cost[i] = (np.sum((-init_y[i][:,None])*(np.log(h)) -
        #         (1-init_y[i][:,None])*(np.log(1-h))))/m

        init_y[i][y[i]] = 1
        
        jpart1 = (-init_y[i][:,None])*(np.log(h)) - (1-init_y[i][:,None])*(np.log(1-h))
        # if you don't use None to change dimenstion result will be a (10,10) matrix, not a vector. That's how it is in numpy.
        tempj = tempj + np.sum(jpart1)

        # Now calculating gradients for back propagation.
        del3 = h - init_y[i][:,None] # (10,1). Again if you didn't use None to add dimension you would end up with a matrix of (10,10)
        del2 = (np.dot(theta2.T,del3)) * (np.vstack((np.zeros(1),sigmoid_gradient(z2)))) # (26,1)
        #  Or if you don't want to add gradient for bias which would have zero derivative do following:
        # d2 = np.dot(theta2.T,d3)[1:]*(sigmoidGradient(z2))
        del2 = del2[1:]  # (25,1). Removing the gradient of bias term which we will not use
        #  you wouldn't have to remove if you calculated del2 like d2. and also there is no bias for del3

        # Now accumulating errors from gradients
        DEL2 = DEL2 + np.dot(del3, newa2.T) # use newa2 as we also need gradient for the bias branch from layer 2 to 3
        # here branch from previous layer's bias goes to every node in present layer so we need newa2 which adds bias's output
        # but no any input goes to any bias node so we don't need gradient for any bias node and we removed it from del2 above
        # DEL2 has order (10,26) just like theta2
        DEL1 = DEL1 + np.dot(del2, a1.T) #a1 already has bias in it and also it is a column vector so we need transpose
        # DEL1 has order (25,401) just like theta1


    # Now all 'm' training samples have been used.
    # Now performing regularization.

    reg_theta1 = theta1[:,1:]
    reg_theta2 = theta2[:,1:]
    sum1 = np.sum(reg_theta1**2)
    sum2 = np.sum(reg_theta2**2)
    reg = (Lambda/(2*m))*( sum1 + sum2 )

    # the final cost is given by J:
    J = (tempj/m) + reg

    # Calculating final gradient with regularization.
    theta1_grad = DEL1 / m + (Lambda/m) * theta1
    theta2_grad = DEL2 / m + (Lambda/m) * theta2
    # Now removing regularization for weights from bias unit
    theta1_grad[:,0] = theta1_grad[:,0] - (Lambda/m) * theta1[:,0]
    theta2_grad[:,0] = theta2_grad[:,0] - (Lambda/m) * theta2[:,0]
    # grad1[0] = grad1[0] - (Lambda / m) * theta1[0]
    # theta1_grad[0] -= (Lambda / m) * theta1[0]
    # theta2_grad[0] -= (Lambda / m) * theta2[0]

    # Now unrolling gradients
    grad = np.append(theta1_grad,theta2_grad).reshape(-1)


    return (J, grad)


'''
def nn_CostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels,
                   X, y, reg_param):
    """
    Computes loss using sum of square errors for a neural network
    using theta as the parameter vector for linear regression to fit
    the data points in X and y with penalty reg_param.
    """

    # Initialize some useful values
    m = len(y)  # number of training examples

    # Reshape nn_params back into neural network
    theta1 = nn_params[:(hidden_layer_size *
                     (input_layer_size + 1))].reshape((hidden_layer_size,
                                                       input_layer_size + 1))

    theta2 = nn_params[-((hidden_layer_size + 1) *
                     num_labels):].reshape((num_labels,
                                            hidden_layer_size + 1))

    # Turn scalar y values into a matrix of binary outcomes
    init_y = np.zeros((m, num_labels))  # 5000 x 10

    for i in range(m):
        init_y[i][y[i]] = 1

    # Add column of ones to X
    ones = np.ones((m, 1))
    d = np.hstack((ones, X))  # add column of ones

    # Compute cost by doing feedforward propogation with theta1 and theta2
    cost = [0] * m  # Creating a list with m elements.
    # Initalize gradient vector
    D1 = np.zeros_like(theta1)
    D2 = np.zeros_like(theta2)
    for i in range(m):
        # Feed Forward Propogation
        a1 = d[i][:, None]  # 401 x 1
        z2 = np.dot(theta1, a1)  # 25 x 1
        a2 = sigmoid(z2)  # 25 x 1
        a2 = np.vstack((np.ones(1), a2))  # 26 x 1
        z3 = np.dot(theta2, a2)  # 10 x 1
        h = sigmoid(z3)  # 10 x 1
        a3 = h  # 10 x 1

        # Calculate cost
        cost[i] = (np.sum((-init_y[i][:, None]) * (np.log(h)) -
                      (1 - init_y[i][:, None]) * (np.log(1 - h)))) / m

        # Calculate Gradient
        d3 = a3 - init_y[i][:, None]
        d2 = np.dot(theta2.T, d3)[1:] * (sigmoid_gradient(z2))

        # Accumulate 'errors' for gradient calculation
        D1 = D1 + np.dot(d2, a1.T)  # 25 x 401 (matches theta0)
        D2 = D2 + np.dot(d3, a2.T)  # 10 x 26 (matches theta1)

    # Add regularization
    reg = (reg_param / (2 * m)) * ((np.sum(theta1[:, 1:] ** 2)) +
                               (np.sum(theta2[:, 1:] ** 2)))

    # Compute final gradient with regularization
    grad1 = (1.0 / m) * D1 + (reg_param / m) * theta1
    grad1[0] = grad1[0] - (reg_param / m) * theta1[0]

    grad2 = (1.0 / m) * D2 + (reg_param / m) * theta2
    grad2[0] = grad2[0] - (reg_param / m) * theta2[0]

    # Append and unroll gradient
    grad = np.append(grad1, grad2).reshape(-1)
    final_cost = sum(cost) + reg

    return (final_cost, grad)
'''

def predict(theta1, theta2, X):
    m = len(X)  # number of samples

    if np.ndim(X) == 1:
        X = X.reshape((-1, 1))

    D1 = np.hstack((np.ones((m, 1)), X))  # add column of ones

    # Calculate hidden layer from theta1 parameters
    hidden_pred = np.dot(D1, theta1.T)  # (5000 x 401) x (401 x 25) = 5000 x 25

    # Add column of ones to new design matrix
    ones = np.ones((len(hidden_pred), 1))  # 5000 x 1
    hidden_pred = sigmoid(hidden_pred)
    hidden_pred = np.hstack((ones, hidden_pred))  # 5000 x 26

    # Calculate output layer from new design matrix
    output_pred = np.dot(hidden_pred, theta2.T)  # (5000 x 26) x (26 x 10)
    output_pred = sigmoid(output_pred)
    # Get predictions
    p = np.argmax(output_pred, axis=1) # to return index of maximum value of each row of output_pred

    return p