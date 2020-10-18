import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse

def augment_feature_vector(X):
    """
    Adds the x[i][0] = 1 feature for each data point x[i].

    Args:
        X - a NumPy matrix of n data points, each with d - 1 features

    Returns: X_augment, an (n, d) NumPy array with the added feature for each datapoint
    """
    column_of_ones = np.ones([len(X), 1])
    return np.hstack((column_of_ones, X))

def compute_probabilities(X, theta, temp_parameter):
    """
    Computes, for each datapoint X[i], the probability that X[i] is labeled as j
    for j = 0, 1, ..., k-1

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        theta - (k, d) NumPy array, where row j represents the parameters of our model for label j
        temp_parameter - the temperature parameter of softmax function (scalar)
    Returns:
        H - (k, n) NumPy array, where each entry H[j][i] is the probability that X[i] is labeled as j
    """
    H = np.empty((theta.shape[0], X.shape[0]))
    temp = np.empty((theta.shape[0],))
    for i in range(X.shape[0]):
        temp[:] = (theta @ X[i,:].reshape(X.shape[1],1))[:,0]
        np.divide(temp, temp_parameter, out=temp)
        np.subtract(temp, np.max(temp), out=temp)
        np.exp(temp, out=temp)
        H[:,i] = np.multiply(temp, 1/temp.sum(), out=temp)
    return H

def compute_cost_function(X, Y, theta, lambda_factor, temp_parameter):
    """
    Computes the total cost over every datapoint.

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns
        c - the cost value (scalar)
    """
    temp = np.empty((theta.shape[0], X.shape[0]))
    np.dot(theta, X.T, out=temp)
    np.divide(temp, temp_parameter, out=temp)
    np.subtract(temp, np.max(temp, axis=0), out=temp)
    np.exp(temp, out=temp)
    np.divide(temp, temp.sum(axis=0), out=temp)
    J = np.arange(0, theta.shape[0], 1, dtype=int).reshape((theta.shape[0], 1))
    J = (J==Y)
    np.log(temp, out=temp)
    np.multiply(J, temp, out=temp)
    return -np.sum(temp)/X.shape[0] + (lambda_factor/2)*np.sum(theta**2)

def run_gradient_descent_iteration(X, Y, theta, alpha, lambda_factor, temp_parameter):
    """
    Runs one step of batch gradient descent

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        alpha - the learning rate (scalar)
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        theta - (k, d) NumPy array that is the final value of parameters theta
    """
    grad = np.empty(theta.shape)
    p = compute_probabilities(X, theta, temp_parameter)
    J = np.arange(0, theta.shape[0], 1).reshape(theta.shape[0],1)
    J = (J == Y).astype(float)
    np.subtract(J, p, out=J)
    np.dot(J, X, out=grad)
    np.divide(grad, lambda_factor*temp_parameter*X.shape[0], out=grad)
    np.subtract(theta, grad, out=grad)
    np.multiply(grad, alpha*lambda_factor, out=grad)
    np.subtract(theta, grad, out=grad)
    return grad

def update_y(*ys):
    """
    Changes the old digit labels for the training and test set for the new (mod 3)
    labels.

    Args:
        train_y - (n, ) NumPy array containing the labels (a number between 0-9)
                 for each datapoint in the training set
        test_y - (n, ) NumPy array containing the labels (a number between 0-9)
                for each datapoint in the test set

    Returns:
        train_y_mod3 - (n, ) NumPy array containing the new labels (a number between 0-2)
                     for each datapoint in the training set
        test_y_mod3 - (n, ) NumPy array containing the new labels (a number between 0-2)
                    for each datapoint in the test set
    """
    out = []
    for y in ys:
        out.append(np.mod(y,3))
    return tuple(out)

def compute_test_error_mod3(X, Y, theta, temp_parameter):
    """
    Returns the error of these new labels when the classifier predicts the digit. (mod 3)

    Args:
        X - (n, d - 1) NumPy array (n datapoints each with d - 1 features)
        Y - (n, ) NumPy array containing the labels (a number from 0-2) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        test_error - the error rate of the classifier (scalar)
    """
    return 1 - (np.mod(get_classification(X, theta, temp_parameter),3) == Y).mean()

def softmax_regression(X, Y, temp_parameter, alpha, lambda_factor, k, num_iterations):
    """
    Runs batch gradient descent for a specified number of iterations on a dataset
    with theta initialized to the all-zeros array. Here, theta is a k by d NumPy array
    where row j represents the parameters of our model for label j for
    j = 0, 1, ..., k-1

    Args:
        X - (n, d - 1) NumPy array (n data points, each with d-1 features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        temp_parameter - the temperature parameter of softmax function (scalar)
        alpha - the learning rate (scalar)
        lambda_factor - the regularization constant (scalar)
        k - the number of labels (scalar)
        num_iterations - the number of iterations to run gradient descent (scalar)

    Returns:
        theta - (k, d) NumPy array that is the final value of parameters theta
        cost_function_progression - a Python list containing the cost calculated at each step of gradient descent
    """
    X = augment_feature_vector(X)
    theta = np.zeros([k, X.shape[1]])
    cost_function_progression = []
    for _ in range(num_iterations):
        cost_function_progression.append(compute_cost_function(X, Y, theta, lambda_factor, temp_parameter))
        theta = run_gradient_descent_iteration(X, Y, theta, alpha, lambda_factor, temp_parameter)
    return theta, cost_function_progression

def get_classification(X, theta, temp_parameter):
    """
    Makes predictions by classifying a given dataset

    Args:
        X - (n, d - 1) NumPy array (n data points, each with d - 1 features)
        theta - (k, d) NumPy array where row j represents the parameters of our model for
                label j
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        Y - (n, ) NumPy array, containing the predicted label (a number between 0-9) for
            each data point
    """
    X = augment_feature_vector(X)
    probabilities = compute_probabilities(X, theta, temp_parameter)
    return np.argmax(probabilities, axis = 0)

def compute_test_error(X, Y, theta, temp_parameter):
    assigned_labels = get_classification(X, theta, temp_parameter)
    return 1 - np.mean(assigned_labels == Y)

def compute_kernel_probabilities(theta, K, temp_parameter):
    """
    Computes, for each datapoint X[i], the probability that X[i] is labeled as j
    for j = 0, 1, ..., k-1

    Args:
        theta - (k, n) NumPy array, where row j represents the parameters of our model for label j
        K - (n, n) Numpy array (Kernel Matrix)
        temp_parameter - the temperature parameter of softmax function (scalar)
    Returns:
        H - (k, n) NumPy array, where each entry H[j][i] is the probability that X[i] is labeled as j
    """
    H = np.dot(theta, K)
    np.divide(H, temp_parameter, out=H)
    np.subtract(H, np.max(H, axis=0), out=H)
    np.exp(H, out=H)
    np.divide(H, H.sum(axis=0), out=H)
    return H
  
def run_kernel_gradient_descent_iteration(theta, K, JY, alpha, lambda_factor, temp_parameter, BUFF1, BUFF2):
    """
    Runs one step of batch gradient descent

    Args:
        theta - (k, n) Mutable NumPy array, where row j represents the parameters of our model for label j
        K - (n, n) Numpy array (Kernel Matrix)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        alpha - the learning rate (scalar)
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)
        JY - (k, n) computed (J == Y)
        BUFF1 (k, n) buffer
        BUFF2 (k, n) buffer

    Returns:
        cost - escalar that is the cost before currer iteration
    """
    np.dot(theta, K, out=BUFF1)
    cost2 = (lambda_factor/2)*np.sum(BUFF1)
    np.divide(BUFF1, temp_parameter, out=BUFF1)
    np.subtract(BUFF1, np.max(BUFF1, axis=0), out=BUFF1)
    np.exp(BUFF1, out=BUFF1)
    np.divide(BUFF1, BUFF1.sum(axis=0), out=BUFF1)
    np.log(BUFF1, out=BUFF2)
    np.multiply(JY, BUFF2, out=BUFF2)
    cost = -np.sum(BUFF2)/K.shape[0] + cost2
    np.subtract(JY, BUFF1, out=BUFF1)
    np.multiply(BUFF1, alpha/(temp_parameter*K.shape[0]), out=BUFF1)
    np.multiply(theta, (1-alpha*lambda_factor), out=theta)
    np.add(theta, BUFF1, out=theta)
    return cost

def softmax_kernel_regression(X, Y, kernel, temp_parameter, alpha, lambda_factor, k, num_iterations):
    """
    Runs batch gradient descent for a specified number of iterations on a dataset
    with theta initialized to the all-zeros array. Here, theta is a k by n NumPy array

    Args:
        X - (n, d) NumPy array (n data points, each with d - 1 features)
        kernel - Function Kernel
        temp_parameter - the temperature parameter of softmax function (scalar)
        alpha - the learning rate (scalar)
        lambda_factor - the regularization constant (scalar)
        k - the number of labels (scalar)
        num_iterations - the number of iterations to run gradient descent (scalar)
        
    Returns:
        theta - (k, d) NumPy array that is the final value of parameters theta
        cost - a Python list containing the cost calculated at each step of gradient descent
    """
    X = augment_feature_vector(X)
    K = kernel(X,X)
    theta = np.zeros((k, X.shape[0]))
    BUFF1 = np.empty((k, X.shape[0]))
    BUFF2 = np.empty((k, X.shape[0]))
    J = np.arange(0, theta.shape[0], 1).reshape(theta.shape[0],1)
    JY = (J == Y).astype(float)  
    cost = np.empty((num_iterations,))
    for i in range(num_iterations):
        cost[i] = run_kernel_gradient_descent_iteration(theta, K, JY, alpha, lambda_factor, temp_parameter, BUFF1, BUFF2)
    return theta, cost

def get_kernel_classification(X, kernel, theta, Xt, temp_parameter):
    """
    Makes predictions by classifying a given dataset

    Args:
        X - (n, d - 1) NumPy array (n data points, each with d - 1 features)
        kernel - Function Kernel
        theta - (k, n) NumPy array where row j represents the parameters of our model for
                label j
        Xt - (m, d - 1) NumPy array (m data points, each with d - 1 features)
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        Y - (n, ) NumPy array, containing the predicted label (a number between 0-9) for
            each data point
    """
    X = augment_feature_vector(X)
    Xt = augment_feature_vector(Xt)
    K = kernel(Xt, X)
    probabilities = compute_kernel_probabilities(theta, K, temp_parameter)
    return np.argmax(probabilities, axis = 0)

def compute_kernel_test_error(X, Y, kernel, theta, Xt, temp_parameter):
    assigned_labels = get_kernel_classification(X, kernel, theta, Xt, temp_parameter)
    return 1 - np.mean(assigned_labels == Y)

def plot_cost_function_over_time(cost_function_history, title=None):
    plt.figure()
    plt.title(title)
    plt.plot(range(len(cost_function_history)), cost_function_history)
    plt.ylabel('Cost Function')
    plt.xlabel('Iteration number')
    plt.show()
