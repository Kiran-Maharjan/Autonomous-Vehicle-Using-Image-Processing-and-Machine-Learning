import numpy as np
import scipy as sp
from scipy import io
from scipy.optimize import minimize


def randInitializeWeights(ip_nodes, op_nodes):
    # here ip_nodes and op_nodes are no. of nodes at the input side and receiving(or output) side of the weights of connections.
    epsilon_init = 0.12
    rand_weights = np.random.uniform(-epsilon_init,epsilon_init,(ip_nodes,op_nodes))
    # or you could have used: rand_weights = (np.random.random(L_out, 1 + L_in)) * 2 * epsilon_init -  epsilon_init
    return rand_weights

def sigmoid(z):
    g = 1.0 / (1 + np.e**(-z))
    return g


def sigmoid_gradient(z):
    return sigmoid(z)*(1-sigmoid(z))


# initializing layers of the network
input_layer_size = 400;  # 20x20 Input Images of Digits
hidden_layer_size = 25;  # 25 hidden units
num_labels = 10;  # 10 labels, from 1 to 10. Remember that 0 is mapped to 10


# loading data

print("Loading training data.....")

loaded_data = sp.io.loadmat('ex4data1')
z = loaded_data['y']
y = loaded_data['y'].flatten()
# y has shape (50000,)
X = loaded_data['X']  # (5000,400)
# Initialize neural network parameters
print("Initializing neural network parameters...")
initial_theta1 = randInitializeWeights(input_layer_size + 1, hidden_layer_size)
initial_theta2 = randInitializeWeights(hidden_layer_size + 1, num_labels)

# Unroll parameters
nn_params = np.append(initial_theta1,initial_theta2).reshape(-1)
Lambda = 3.0



# Now following is the cost function part that needs to be checked
no_of_weights1 = (input_layer_size+1)*hidden_layer_size
theta1 = nn_params[:no_of_weights1].reshape((hidden_layer_size,input_layer_size+1))
theta2 = nn_params[no_of_weights1:].reshape((num_labels,hidden_layer_size+1))


ut_theta1 = nn_params[:(hidden_layer_size * 
			           (input_layer_size + 1))].reshape((hidden_layer_size, 
							                             input_layer_size + 1))
  
ut_theta2 = nn_params[-((hidden_layer_size + 1) * 
                          num_labels):].reshape((num_labels,
					                             hidden_layer_size + 1))
print('theta1 shape: ',theta1.shape)
print('theta2 shape: ',theta2.shape)
print(theta1[4][100], " = ", ut_theta1[4][100])
print(theta2[5][10], " = ", ut_theta2[5][10])
m = len(y)
print("m = ", m)

# from utils start
# Turn scalar y values into a matrix of binary outcomes
init_y = np.zeros((m,num_labels)) # 5000 x 10

for i in range(m):
	init_y[i][y[i]] = 1

#from utils end


# Add column of ones to X
ones = np.ones((m,1))
newX = np.hstack((ones,X))  # Add a column of ones
print(newX.shape)
tempj = 0
c = np.zeros(num_labels)
print("shape of c : ", c.shape)
DEL1 = np.zeros(theta1.shape) # or you can use: DEL1 = np.zeros_like(theta1)
DEL2 = np.zeros_like(theta2)
print(DEL1.shape, "\n", DEL2.shape)


#############################################################
####THIS IS WHERE FOR LOOP IN COST FUNCTION WOULD START######
#############################################################
i = 4700
a1 = newX[i][:,None]
print(a1.shape)
z2 = np.dot(theta1,a1)  # (25,401) and (401,1) give (25,1)
a2 = sigmoid(z2)
newa2 = np.vstack((np.ones(1),a2))
print(a2.shape, newa2.shape)
z3 = np.dot(theta2,newa2) # (10,26) and (26,1) give (10,1)
h = sigmoid(z3)
print(h.shape)

c[0] = 0
for j in range(1,10):
	c[j] = j

print(c)
if (y[i] == 10):
	y[i] = 0

print(y[i])

print('y is : ', y[i])
newy = (c == y[i])
print('newy is : ', newy)
print(newy.dtype)
newy = newy.astype(float)
print(newy)
print(newy.dtype)
jpart1 = (-newy[:,None])*(np.log(h)) - (1-newy[:,None])*(np.log(1-h))
print(jpart1)
tempj = tempj + np.sum(jpart1)
print("tempj is : ",tempj)

del3 = h - newy[:,None]
del2 = (np.dot(theta2.T,del3)) * (np.vstack((np.zeros(1),sigmoid_gradient(z2))))

print(del3)
print(h)
print(del2.shape)
del2 = del2[1:]
print(del2.shape)
DEL2 = DEL2 + np.dot(del3, newa2.T)
DEL1 = DEL1 + np.dot(del2, a1.T)
print(DEL1.shape, DEL2.shape)


#exiting out of loop here

reg_theta1 = theta1[:,1:]
reg_theta2 = theta2[:,1:]
print(reg_theta1.shape, reg_theta2.shape)

sum1 = np.sum(reg_theta1**2)
sum2 = np.sum(reg_theta2**2)
reg = (Lambda/(2*m))*( sum1 + sum2 )
print(reg_theta1[2][2])
print((reg_theta1**2)[2][2])
print(reg)


# the final cost is given by J:
J = (tempj/m) + reg
print(tempj/m)
print(J)
