import math
import numpy as np
from scipy.optimize import fmin

def mean(numbers):
	return sum(numbers)/float(len(numbers))
 
def stdev(numbers):
	avg = mean(numbers)
	variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
	return math.sqrt(variance)
	
def getProbability(x,mu,si):
	return (np.exp(-np.power(x - mu, 2.) / (2 * np.power(si, 2.))))*(1/(si*math.sqrt(2*math.pi)))
	
def sigmoid(X):
    """ Compute the sigmoid function """
         
    den = 1.0 + np.exp(-1.0 * X)
 
    d = 1.0 / den
 
    return d

def threshold_probs(probs):
    """ Converts probabilities to hard classification """
    classes = np.ones(len(probs),)
    classes[probs < 0.5] = 0
    return classes
            
def fmin_simple(grad, loss, initparams):
    step =1 
    alpha = 0.000000001
    w_old = initparams
    diff = 1
    while(step<=10000 and diff >= 1e-10):
        w_new = w_old - alpha*grad(w_old)
        diff = w_new - w_old
        diff = diff.dot(diff)        
        if(loss(w_new) > loss(w_old)):
            alpha = alpha/2
        if(loss(w_new) <= loss(w_old)):
            w_old = w_new
            step = step + 1
    return w_new
	
	