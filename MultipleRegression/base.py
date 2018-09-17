import numpy as np
import math
import time

def compute_mse_vectorized(w,X,Y):
    res = Y - np.dot(X,w)
    totalError = np.dot(res.T,res)
    return totalError / float(len(Y))

def step_gradient_vectorized(w_current,X,Y,learningRate):
    res = Y - np.dot(X,w_current)
    b_gradient = np.sum(res)
    X = X[:,1][:,np.newaxis]
    m_gradient = np.sum(np.multiply(res,X))
    new_w = np.array([(w_current[0] + (2 * learningRate * b_gradient)),
             (w_current[1] + (2 * learningRate * m_gradient))])
    return [new_w,b_gradient,m_gradient]

def gradient_descent_runner_vectorized(starting_w, X,Y, learning_rate, epsilon):
    w = starting_w
    grad = np.array([np.inf,np.inf])
    i = 0
    while (np.linalg.norm(grad)>=epsilon):
        w,b_gradient,m_gradient = step_gradient_vectorized(w, X, Y, learning_rate)
        grad = np.array([b_gradient,m_gradient])
        #print(np.linalg.norm(grad))
        if i % 1000 == 0:
            print("MSE na iteração {0} é de {1}".format(i,compute_mse_vectorized(w, X, Y)))
        i+= 1
    return w

points = np.genfromtxt("income.csv", delimiter=",")
points = np.c_[np.ones(len(points)),points]
X = points[:,[0,1]]
Y = points[:,2][:,np.newaxis]
init_w = np.zeros((2,1))
learning_rate = 0.0001
#num_iterations = 10000
epsilon = 0.5
print("Starting gradient descent at w0 = {0}, w1 = {1}, error = {2}".format(init_w[0], init_w[1], compute_mse_vectorized(init_w, X,Y)))
print("Running...")
tic = time.time()
w = gradient_descent_runner_vectorized(init_w, X,Y, learning_rate, epsilon)
toc = time.time()
print("Gradiente descendente convergiu com w0 = {0}, w1 = {1}, error = {2}".format(w[0], w[1], compute_mse_vectorized(w,X,Y)))
print("Versão vetorizada rodou em: " + str(1000*(toc-tic)) + " ms")
