import numpy as np
import utilities as utils

class Classifier:
    """
    Generic classifier interface; returns random classification
    """
    
    def __init__( self ):
        """ Params can contain any useful parameters for the algorithm """
        
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        
    def predict(self, Xtest):
        probs = np.random.rand(Xtest.shape[0])
        ytest = utils.threshold_probs(probs)
        return ytest
            
class NaiveBayes(Classifier):

    def __init__( self ):
        self.parameters = {}
        
    # Uncomment and define, as currently simply does parent    
    def learn(self, Xtrain, ytrain):
		rows = len(Xtrain)
		col = len(Xtrain[0])
		x = np.zeros((rows,col))
		y = np.zeros((rows,col))
		k = 0
		l = 0
		cp1 = np.zeros((col,2))
		cp0 = np.zeros((col,2))
		for i in range(Xtrain.shape[0]):
			if (ytrain[i] == 1.0):
				x[k] = Xtrain[i]
				k+=1
			else:
				y[l] = Xtrain[i]
				l+=1
		for i in range(Xtrain.shape[1]):
			cp1[i] = (utils.mean(x[:k,i]),utils.stdev(x[:k,i]))
			cp0[i] = (utils.mean(y[:l,i]),utils.stdev(y[:l,i]))
		self.parameters = {0:cp0,1:cp1}
		
    def predict(self, Xtest):
		ytest = []
		for i in range(len(Xtest)):
			class_ = self.predict_each_row(Xtest[i])
			ytest.append(class_)
		return ytest
	
    def predict_each_row(self,Row):
		probabilities = self.getProbabilities(Row)
		class_, class_p = None, -1
		for index, p in probabilities.iteritems():
			if class_ is None or p > class_p:
				class_p = p
				class_ = index
		return class_
    def getProbabilities(self, Xtest):
		probabilities = {}
		for index, parameter in self.parameters.iteritems():
			probabilities[index] = 1
			for i in range(len(parameter)):
				mu, si = parameter[i]
				x = Xtest[i]
				probabilities[index] *= utils.getProbability(x, mu, si)
		return probabilities

class LogitReg(Classifier):

    def __init__( self ):
        self.weights = None
        
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        self.weights = np.zeros(Xtrain.shape[1],)
        #print ytrain
        #lossfnc = lambda l: self.lfunc(l, Xtrain, ytrain)
        lossfnc = lambda l: self.lfunc(l,Xtrain,ytrain)
        grad = lambda w: self.gradient_descent(w, Xtrain,ytrain)
        self.weights = utils.fmin_simple(grad, lossfnc, self.weights)
        
    def predict(self, Xtest):
        probs = utils.sigmoid(np.dot(Xtest, self.weights))
        ytest = utils.threshold_probs(probs)  
        return ytest
    def lfunc(self, theta,X,y): 
	    tt = X.shape[0] # number of training examples
	    theta = np.reshape(theta,(len(theta),1))
	    J = (1./tt) * (-np.transpose(y).dot(np.log(utils.sigmoid(X.dot(theta)))) - np.transpose(1-y).dot(np.log(1-utils.sigmoid(X.dot(theta)))))
	    return J[0]
    def gradient_descent(self, theta,X,y): 
    	tt = X.shape[0] # number of training examples
    	theta = np.reshape(theta,(len(theta),1))
        y = np.reshape(y,(len(y),1))
    	delta = np.transpose(np.transpose(-y + utils.sigmoid(X.dot(theta))).dot(X))        
    	# When you write your own minimizers, you will also return a gradient here
        return delta[0]