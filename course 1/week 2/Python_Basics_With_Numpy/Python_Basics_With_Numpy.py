import math
import numpy as np
import time


def sigmoid(x):
    """
    Compute the sigmoid of x

    Arguments:
    x -- A scalar or numpy array of any size

    Return:
    s -- sigmoid(x)
    """
    s = 1 / (1 + np.exp(-x))
    return s


def basic_sigmoid(x):
    """
    Compute sigmoid of x.

    Arguments:
    x -- A scalar

    Return:
    s -- sigmoid(x)
    """
    s = 1 / (1 + math.exp(-x))
    return s


def sigmoid_derivative(x):
    """
    Compute the gradient (also called the slope or derivative) of the sigmoid function with respect to its input x.
    You can store the output of the sigmoid function into variables and then use it to calculate the gradient.

    Arguments:
    x -- A scalar or numpy array

    Return:
    ds -- Your computed gradient.
    """
    s = sigmoid(x)
    ds = s * (1 - s)
    return ds


def image2vector(image):
    """
    Argument:
    image -- a numpy array of shape (length, height, depth)

    Returns:
    v -- a vector of shape (length*height*depth, 1)
    """
    # Using -1 for good idea!!!!
    v = image.reshape(-1, 1)
    # Or other:
    #     v = image.reshape(image.shape[0]*image.shape[1]*image.shape[2], 1)
    return v


def normalizeRows(x):
    """
    Implement a function that normalizes each row of the matrix x (to have unit length).

    Argument:
    x -- A numpy matrix of shape (n, m)

    Returns:
    x -- The normalized (by row) numpy matrix. You are allowed to modify x.
    """
    # Compute x_norm as the norm 2 of x. Use np.linalg.norm(..., ord = 2, axis = ..., keepdims = True)
    x_norm = np.linalg.norm(x, ord=2, axis=1, keepdims=True)
    x = x / x_norm
    return x


def softmax(x):
    """Calculates the softmax for each row of the input x.

    Your code should work for a row vector and also for matrices of shape (m,n).

    Argument:
    x -- A numpy matrix of shape (m,n)

    Returns:
    s -- A numpy matrix equal to the softmax of x, of shape (m,n)
    """
    # Apply exp() element-wise to x. Use np.exp(...).
    x_exp = np.exp(x)
    print('x_exp - shape:', x_exp.shape)
    # Create a vector x_sum that sums each row of x_exp. Use np.sum(..., axis = 1, keepdims = True).
    x_sum = np.sum(x_exp, axis=1, keepdims=True)
    print('x_sum - shape', x_sum.shape)
    # Compute softmax(x) by dividing x_exp by x_sum. It should automatically use numpy broadcasting.
    s = x_exp / x_sum
    return s


def L1(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)

    Returns:
    loss -- the value of the L1 loss function defined above
    """
    loss = np.sum(np.abs(yhat - y))
    return loss


def L2(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)

    Returns:
    loss -- the value of the L2 loss function defined above
    """
    loss = np.dot(yhat - y, yhat - y)
    return loss


def main():
    x = 3
    print(basic_sigmoid(x))
    # example of np.exp
    x = np.array([1, 2, 3])
    print(np.exp(x))
    x = np.array([1, 2, 3])
    print(x + 3)
    x = np.array([1, 2, 3])
    print(sigmoid(x))
    x = np.array([1, 2, 3])
    print("sigmoid_derivative(x) = " + str(sigmoid_derivative(x)))
    image = np.array([[[0.67826139, 0.29380381],
                       [0.90714982, 0.52835647],
                       [0.4215251, 0.45017551]],

                      [[0.92814219, 0.96677647],
                       [0.85304703, 0.52351845],
                       [0.19981397, 0.27417313]],

                      [[0.60659855, 0.00533165],
                       [0.10820313, 0.49978937],
                       [0.34144279, 0.94630077]]])
    print("image2vector(image) = " + str(image2vector(image)))
    x = np.array([
        [0, 3, 4],
        [1, 6, 4]])
    print("normalizeRows(x) = " + str(normalizeRows(x)))
    x = np.array([
        [9, 2, 5, 0, 0],
        [7, 5, 0, 0, 0]])
    print("softmax(x) = " + str(softmax(x)))

    y = np.array([[9],
                  [5]])
    print(y.shape)
    print('x / y: ', x / y)

    x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
    x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]

    ### VECTORIZED DOT PRODUCT OF VECTORS ###
    tic = time.process_time()
    dot = np.dot(x1, x2)
    toc = time.process_time()
    print("dot = " + str(dot) + "\n ----- Computation time = " + str(1000 * (toc - tic)) + "ms")

    ### VECTORIZED OUTER PRODUCT ###
    tic = time.process_time()
    outer = np.outer(x1, x2)
    toc = time.process_time()
    print("outer = " + str(outer) + "\n ----- Computation time = " + str(1000 * (toc - tic)) + "ms")

    ### VECTORIZED ELEMENTWISE MULTIPLICATION ###
    tic = time.process_time()
    mul = np.multiply(x1, x2)
    toc = time.process_time()
    print("elementwise multiplication = " + str(mul) + "\n ----- Computation time = " + str(1000 * (toc - tic)) + "ms")

    ### VECTORIZED GENERAL DOT PRODUCT ###
    tic = time.process_time()
    W = np.random.rand(3, len(x1))  # Random 3*len(x1) numpy array
    dot = np.dot(W, x1)
    toc = time.process_time()
    print("gdot = " + str(dot) + "\n ----- Computation time = " + str(1000 * (toc - tic)) + "ms")

    yhat = np.array([.9, 0.2, 0.1, .4, .9])
    y = np.array([1, 0, 0, 1, 1])
    print("L1 = " + str(L1(yhat, y)))
    yhat = np.array([.9, 0.2, 0.1, .4, .9])
    y = np.array([1, 0, 0, 1, 1])
    print("L2 = " + str(L2(yhat, y)))


if __name__ == "__main__":
    main()
