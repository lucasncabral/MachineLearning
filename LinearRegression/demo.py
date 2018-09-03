# -*- coding: utf-8 -*-

"""
Created on Thu Aug 30 10:08:52 2018
@author: Lucas Cabral
"""

from numpy import *
import time
import matplotlib.pyplot as plt

def compute_error_for_line_given_points(b, m, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m * x + b)) ** 2
    return totalError / float(len(points))

def step_gradient(current_b, current_m, points, learning_rate):
    gradient_b = 0
    gradient_m = 0
    N = float(len(points))
    
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        
        gradient_b += -(2/N) * (y - ((current_m * x) + current_b))
        gradient_m += -(2/N) * x * (y - ((current_m * x) + current_b))    
    new_b = current_b - (learning_rate * gradient_b)
    new_m = current_m - (learning_rate * gradient_m)
    
    return [new_b, new_m, gradient_b, gradient_m]

def gradient_size(b,m):
    return math.sqrt(b ** 2 + m ** 2)

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    
    iteration = 0
    converged = False
    error = 0.04
    while(not converged):
        if iteration >= num_iterations:
            return [b, m, iteration]
        
        b, m, gradient_b, gradient_m = step_gradient(b, m, array(points), learning_rate)
        plot_values_y.append(gradient_size(gradient_b, gradient_m))  
        iteration += 1
        if(gradient_size(gradient_b, gradient_m) < error):
            converged = True
    return [b, m, iteration]


def closed_form(points):
    sum_x = 0
    sum_y = 0
    N = float(len(points))
    for i in range(0, len(points)):
        sum_x += points[i, 0]
        sum_y += points[i, 1]
    mean_x = sum_x / N
    mean_y = sum_y / N
    
    erro_x = 0
    superior = 0
    for i in range(0, len(points)):
        x = points[i,0]
        y = points[i,1]
        superior += (x - mean_x) * (y - mean_y)
        erro_x += (x - mean_x) ** 2
    
    m = superior / erro_x
    b = mean_y - m * mean_x
    
    return [b,m]

def run():
    # QUESTÃO 1
    # points = genfromtxt("data.csv", delimiter=",")
    points = genfromtxt("income.csv", delimiter=",")
    learning_rate = 0.003
    
    # y = mx + b
    initial_b = 0 
    initial_m = 0
    num_iterations = 16000
    print ("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points)))
    
    localtime_before = time.time()
    [b, m, iteration] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    num_iterations = iteration
    localtime_gd = time.time() - localtime_before
    print ("Tempo execução: {0}".format(localtime_gd))
    
    print ("After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, compute_error_for_line_given_points(b, m, points)))
    
    
    localtime_before = time.time()
    [b, m] = closed_form(array(points))
    localtime_cf = time.time() - localtime_before
    print ("Tempo execução: {0}".format(localtime_cf))
    
    print ("By closed form b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, compute_error_for_line_given_points(b, m, points)))
    
    
if __name__ == '__main__':
    run()
    
    
# QUESTÃO 3
# RSS diminui a cada iteração, considerando apenas as 1000
    
# QUESTÃO 4
# original: learning_rate = 0.0001 and num_interations = 1000
# original: learning_rate = 0.003 and num_interations = 16000
# W0 = b = -39.0339106358241
# W1 = m = 5.574936756431266

# QUESTÃO 5
# Feito

# QUESTÃO 6
# limiar = 0.04
# W0 = b = -39.026550442698195
# W1 = m = 5.574498619168578
    
# QUESTÃO 7
# FORMA FECHADA: 0.0
# GRADIENTE DESCENDENTE: ~0.72 