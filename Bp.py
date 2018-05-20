import random
import numpy as np

class Network(object):

	def __init__(self, size):
		'''
		param size: type:list store the number of every layer's neuron

		'''
		self.num_layers = len(size)
		self.size = size

		#Random seed
		np.random.seed(1)

		#bias initialization
		self.bias = [np.random.randn(y,1) for y in size[1:]]

		#weights initialization
		self.weights = [np.random.randn(y,x) for x,y in zip(size[:-1],size[1:])]

	


		



	def forward(self,a):
		'''
		forward propagation
		:param a: input value
		:return: output of neuronn
		'''
		for b, w in zip(self.bias, self.weights):
			a = sigmoid(np.dot(w,a)+b)
		return a

	def backward(self,x,y):
		'''
		'''
		new_b = [np.zeros(b.shape) for b in self.bias]
		new_w = [np.zeros(w.shape) for w in self.weights]
		activation	= x
		#matrix that store the value of every neuron in every layer
		activations = [x]
		#store value of uncalculated neuron
		zs = []
		for b, w in zip(self.bias, self.weights):
			z = np.dot(w, activation) + b
			zs.append(z)
			activation = sigmoid(z)
			activations.append(activation)

		# calculate delta
		delta = self.cost_derivative(activations[-1], y) * \
			deri_sigmoid(zs[-1])
		new_b[-1] = delta
		# multiply previous layer's output
		new_w[-1] = np.dot(delta, activations[-2].transpose())
		for l in xrange(2, self.num_layers):
			# update from the last l layer
			# use delta of l+1 to calculate delta of l
			z = zs[-l]
			sp = deri_sigmoid(z)
			delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
			new_b[-l] = delta
			new_w[-l] = np.dot(delta, activations[-l-1].transpose())
		return(new_b, new_w)


	def evaluate(self, test_data):

		test_results = [(np.argmax(self.forward(x)),y)
						for (x,y) in test_data]
							
		return sum(int(x == y) for (x,y) in test_results)


	def cost_derivative(self, output_activations, y):
		return (output_activations-y)

	def SGD(self, training_data, epochs, mini_batch_size, eta,test_data=None):
		'''

		:param training_data:
		:param epochs:
		:param mini_batch_size:
		:param eta:
		:param test_data:
		:return:
		'''
		if test_data.any: n_test = len(test_data)
		n = len(training_data)

		for j in xrange(epochs):
			random.shuffle(training_data)

			self.update(training_data, eta)
		if test_data:
			print "Epoch {0}:{1}/{2}".format(j,self.evaluate(test_data),n_test)
		else:
			print "Epoch {0} complete".format(j)

	def update(self,set,eta):
		'''

		:param self:
		:param set:
		:param eta:
		:return:
		'''
		new_b = [np.zeros(b.shape) for b in self.bias]
		new_w = [np.zeros(w.shape) for w in self.weights]

		for x,y in set:
			delta_new_b, delta_new_w = self.backward(x,y)
			new_b = [nb+dnb for nb, dnb in zip(new_b, delta_new_b)]
			new_w = [nw+dnw for nw, dnw in zip(new_w, delta_new_w)]

		self.weights = [w - (eta/len(set)) * nw for w, nw in zip(self.weights,new_w)]
		self.bias = [b - (eta/len(set)) * nb for b, nb in zip(self.bias, new_b)]














# Transfer function
def sigmoid(a):
	return 1.0 / (1.0 + np.exp(-a))


def tanh(a):
	return (np.exp(a) - np.exp(-a)) / (np.exp(a) + np.exp(-a))


# Derivative of transfer function
def deri_sigmoid(a):
	return sigmoid(a) * (1 - sigmoid(a))


def deri_tanh(a):
	return 1 - (tanh(a) * tanh(a))
