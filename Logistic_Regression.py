import numpy as np
import matplotlib.pyplot as plt
import scipy
import h5py
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

#obtain training dataset and test dataset
x_training_orig, y_training, x_test_orig, y_test,classes = load_dataset()

m_train = x_training_orig.shape[0]
m_test = x_test_orig.shape[0]
num_px = x_training_orig.shape[1]
### END CODE HERE ###

print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))
print ("Height/Width of each image: num_px = " + str(num_px))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("x_training_orig shape: " + str(x_training_orig.shape))
print ("y_training shape: " + str(y_training.shape))
print ("x_test shape: " + str(x_test_orig.shape))
print ("y_test shape: " + str(y_test.shape))


x_train_flatten = x_training_orig.reshape(x_training_orig.shape[0],-1).T
x_test_flatten = x_test_orig.reshape(x_test_orig.shape[0],-1).T

print("x_train_flatten: {}".format(x_train_flatten.shape))
print("x_test_flatten: {}".format(x_test_flatten.shape))

x_training = x_train_flatten / 255.
x_test  = x_test_flatten / 255.

def sigmoid(z):
    return 1.0/(1+np.exp(-z))

def initialize_with_zeros(dim):
    # This function create a vector of zeros of shape (dim,1) and initialize b to 0
    w = np.zeros((dim,1))
    b =0

    assert (w.shape == (dim,1))
    assert (isinstance(b,float) or isinstance(b,int))
    return w,b

def propagate(w,b,X,Y):
    """
    implement cost function and its gradient for the propagation
    Arguments:
        w--weights,shape:(num_px*num_px*3,1)
        b--bias,a scalar
        X--data,shape(num_px*num_px*3,number of examples)
        Y--label vector(0 if not cat 1 if cat),shape(1,number of examples)
    Return:
        cost -- negative log-likelihood cost for logistic regression
        dw -- gradient of the loss with respect to w, thus same shape as w
        db -- gradient of the loss with respect to b, thus same shape as b
    """
    m = X.shape[1]
    A = sigmoid(np.dot(w.T,X) + b)
    cost = -np.sum(Y*np.log(A)+(1-Y)*np.log(1-A))/m

    #backpropagation
    dw = np.dot(X,(A-Y).T) / m
    db = np.sum(A-Y) / m

    assert (dw.shape == w.shape)
    assert (db.dtype == float)

    cost = np.squeeze(cost)
    assert (cost.shape == ())

    grad = {"dw":dw,
            "db":db}
    return grad,cost

def optimize(w,b,X,Y,num_iteration,learning_rate,print_cost =False):
    """

    :param w:   weights,shape(num_px*num_px*3,1)
    :param b:   bias,a scalar
    :param X:   data,shape(num_px*num_px*3,numbers of examples)
    :param Y:   true labels,shape(1,numbers of examples)
    :param num_iteration:   numbers of the optimization loop
    :param learning_rate:   learning rate of the gradient descent update rule
    :param print_cost:      print the cost every 100 steps
    :return:
            params -- dictionary containing w and b
            grads -- dictionary containing gradients of w and b with respect to the cost function
            costs -- list of all the costs,then to plot the learning curve
    """
    costs = []

    for i in range(num_iteration):
        grads,cost = propagate(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]

        #update rule
        w = w - learning_rate*dw
        b = b - learning_rate*db

        if i % 100 == 0:
            costs.append(cost)

        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" %(i,cost))

    params = {"w":w,
              "b":b}
    grads = {"dw":dw,
             "db":db}
    return params,grads,costs


def predict(w,b,X):
    '''

    :param w: weights, shape(num_px*num_px*3,1)
    :param b: bias, a scalar
    :param X: data, shape(num_px*num_px*3,numbers of examples)
    :return:
        Y_prediction -- a vector containing  all predictions (0/1) for the examples X
    '''

    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0],1)

    A = sigmoid(np.dot(w.T,X) + b)

    for i in range(A.shape[1]):
        Y_prediction[0,i] = (A[0,i]>0.5)

    assert(Y_prediction.shape == (1,m))

    return Y_prediction


def model(X_train,Y_train,X_test,Y_test,num_iterations = 2000 ,learning_rate = 0.5,print_costs = False):
    '''

    :param X_train: training set,shape(num_px*num_px*3,m_train)
    :param Y_train: training labels,shape(1,m_train)
    :param X_test:  test set ,shape(num_px*num_px*3,m_test)
    :param Y_test:  test labels,shape(1,m_test)
    :param num_iterations: the numbers of the iterations  to optimize the parameters
    :param learning_rate: used in the update rule of gradient descent
    :param print_costs:  print the cost every 100 iterations
    :return:
        d -- information of the model
    '''
    w, b = initialize_with_zeros(X_train.shape[0])

    parameters, grads, costs = optimize(w,b,X_train,Y_train,num_iterations,learning_rate,print_costs)

    w = parameters["w"]
    b = parameters["b"]
    Y_prediction_test = predict(w,b,X_test)
    Y_prediction_train = predict(w,b,X_train)
    print("train accuracy: {}".format(100-np.mean(np.abs(Y_prediction_train - Y_train))*100))
    print("test accuracy: {}%".format(100-np.mean(np.abs(Y_prediction_test - Y_test))*100))

    d ={"costs":costs,
        "Y_prediction_test":Y_prediction_test,
        "Y_prediction_train":Y_prediction_train,
        "w":w,
        "b":b,
        "learning_rate":learning_rate,
        "num_iterations":num_iterations}

    return d

#d = model(x_training,y_training,x_test,y_test,num_iterations=4000,learning_rate=0.008,print_costs=True)



#Plot learning curve

#the learning cruve performance of diffrernt learnning rate used in model
learning_rate =[0.01,0.001,0.0001]
models = {}
for i in learning_rate:
    print("learning rate is : " + str(i))
    models[str(i)] = model(x_training,y_training,x_test,y_test,num_iterations=1500,learning_rate= i,print_costs=False)
    print('\n'+ "-"*30 + '\n')
for i in learning_rate:
    plt.plot(np.squeeze(models[str(i)]["costs"]),label = models[str(i)]["learning_rate"])
plt.ylabel("cost")
plt.xlabel("iterations")

legend = plt.legend(loc= 'upper center',shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()

