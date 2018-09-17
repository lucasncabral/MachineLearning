import numpy as np
import time
from sklearn.linear_model import LinearRegression

def compute_mse_vectorized(w,X,Y):
    res = Y - np.dot(X,w)
    totalError = np.dot(res.T,res)
    return totalError / float(len(Y))

def step_gradient_vectorized(w_current,X,Y,learningRate):
    res = Y - np.dot(X,w_current)
    gradient = np.multiply(res,X)
    gradient = np.sum(gradient,axis=0)
    gradient = gradient[:,np.newaxis]
    new_w = w_current + 2 * learningRate * gradient
    return [new_w,gradient]

def gradient_descent_runner_vectorized(starting_w, X,Y, learning_rate, epsilon):
    w = starting_w
    grad = np.array([np.inf,np.inf])
    while (np.linalg.norm(grad)>=epsilon):
        w,grad = step_gradient_vectorized(w, X, Y, learning_rate)
    return w

points = np.genfromtxt("sample_treino.csv", delimiter=",")
points = np.c_[np.ones(len(points)),points]
X = points[:,0:-1]
Y = points[:,-1][:,np.newaxis]
init_w = np.zeros((len(X[0]),1))
learning_rate = 0.0001
epsilon = 0.5
print("Starting gradient descent at w0 = {0}, w1 = {1}, error = {2}".format(init_w[0], init_w[1], compute_mse_vectorized(init_w, X,Y)))
print("Running...")
tic = time.time()
w = gradient_descent_runner_vectorized(init_w, X,Y, learning_rate, epsilon)
toc = time.time()
print("Gradiente descendente convergiu com w0 = {0}, w1 = {1}, w2 = {2}, w3 = {3}, w4 = {4}, w5 = {5}, error = {6}".format(w[0], w[1], w[2], w[3], w[4], w[5], compute_mse_vectorized(w,X,Y)))
print("Versão vetorizada rodou em: " + str(1000*(toc-tic)) + " ms")

reg = LinearRegression()
reg.fit(X,Y)
print(reg.coef_)