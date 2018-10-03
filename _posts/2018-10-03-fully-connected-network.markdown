---
layout: post
title:  "Fully Connected Neural Networks From Scratch"
permalink: /fully-connected/
excerpt: "Demystify backprop."
mathjax: true
date:   2018-10-02 11:01:00
mathjax: true

---

Neural networks are often explained in the most complicated ways possible, but we'll show just how simple they can be.

Suppose we wish to implement a fully-connected feedforward neural network with 1 input layer, 1 hidden layer, and 1 output layer. We call such a network to be a two-layer neural network (ignoring the input layer as it is trivially present).

In the feedforward step, we feed an input \\(x \in R^{1 \times d}\\) through the network.

$$
z^{(1)} = W^{(1)}x + b^{(1)}
$$

$$
a^{(1)} = \sigma(z^{(1)})
$$

$$
z^{(2)} = W^{(2)}a^{(1)} + b^{(2)} 
$$

$$
\hat{y} = a^{(2)} = \mbox{ softmax }(z^{(2)} )
$$

```python
def python_feedforward(X, cache):
	assert(X.shape[0] == cache['W1'].shape[1])
	cache['X'] = X

	pdb.set_trace()
	z1 = cache['W1'].dot(X) + np.matlib.repmat(cache['b1'], cache['N'], 1).T
	cache['z1'] = z1

	a1 = 1./ (1. + np.exp(-z1))
	cache['a1'] = a1

	assert(a1.shape[0] == cache['W2'].shape[1])
	z2 = cache['W2'].dot(a1) + np.matlib.repmat(cache['b2'], cache['N'], 1).T
	cache['z2'] = z2

	#logits = np.exp(z2) / np.matlib.repmat( np.sum(np.exp(z2.T),axis=1).T, cache['D_out'], 1)
	logits = np.exp(z2) / np.matlib.repmat( np.sum(np.exp(z2),axis=0), cache['D_out'], 1)
	cache['logits'] = logits # also called cache.yc, a2
	return logits
```

We'll add L2 regularization on the weights of the network:

```python
def norms(cache):
	all_sum =  scipy.linalg.norm(cache['W1']) **2
	all_sum += scipy.linalg.norm(cache['W2']) **2
	return all_sum
```

We use a cross-entropy loss:

$$
CE(y, \hat{y}) = - \sum\limits_{i=0}^{C-1} y_i \mbox{ log }(\hat{y}_i)
$$


And we also add the penalization term for the magnitude of \\(W\\):

$$
J(W,b;x,y) = \frac{1}{N} \sum\limits_{i=1}^N CE^{(i)}(y, \hat{y}) + 0.5 \lambda \| p \|^2
$$

```python
def python_loss(cache, y):
	"""
	a = array([[ 0,  1,  2],
		       [ 3,  4,  5],
		       [ 6,  7,  8],
		       [ 9, 10, 11]])
	logits[np.array([3,2,1]),np.arange(3)]
	
	returns array([9, 7, 5])
	"""
	yc_scores = cache['logits'][y,np.arange(cache['N'])]
	ce_sum = -np.sum(np.log(yc_scores))

	data_loss = ce_sum / cache['N'] # divide by batch size
	reg_loss = 0.5 * cache['reg'] * norms(cache)
	loss = data_loss + reg_loss
	print "Loss: ", loss
	return loss
```

Now we use the chain rule to backpropagate gradients:

$$
\frac{\partial CE(y, \hat{y}) }{ \partial z^{(2)}} = \hat{y} − y
$$


Recall that \\(z^{(2)} = W^{(2)}a^{(1)} + b^{(2)} \\). Therefore,

$$
\frac{ \partial CE(y, \hat{y} )}{\partial W^{(2)}} =
\frac{ \partial CE(y, \hat{y} )}{\partial z^{(2)}} \frac{\partial z^{(2)}}{\partial W^{(2)}}
$$

$$
\frac{ \partial CE(y, \hat{y} )}{ \partial W^{(2)} }= ( \hat{y} − y) a^{(1)^T}
$$

Similarly,

$$
\frac{ \partial CE(y, yˆ) }{ \partial b^{(2)} }= \hat{y} − y
$$

Going across L2:

$$
\frac{ \partial z^{(2)} }{\partial a^{(1)} }= W^{(2)^T}
$$

$$
\frac{ \partial CE(y, yˆ) }{\partial a^{(1)} } = \frac{\partial CE(y, yˆ)}{\partial z^{(2)} } \frac{ \partial z^{(2)} }{\partial a^{(1)} } = W^{(2)^T} ( \hat{y} − y)
$$

Going across the non-linearity of L1:

$$
\frac{\partial CE(y, yˆ) }{\partial z^{(1)} }= \frac{ \partial CE(y, yˆ)}{ \partial a^{(1)}} \frac{ \partial σ(z)}{ \partial z^{(1)} }
$$

$$
= \frac{\partial CE(y, yˆ)}{\partial a^{(1)} } ◦ σ(z^{(1)}) ◦ (1 − \sigma(z^{(1)}))
$$

Note that we have assumed that $$\sigma(·)$$ works on matrices by applying an element-wise sigmoid, and ◦ is the
element-wise (Hadamard) product.
That brings us to our final gradients:
\partial CE(y, yˆ)
\partial W(1) =
\partial CE(y, yˆ)
\partial z(1)
\partial z(1)
\partial W(1)
\partial CE(y, yˆ)
\partial W(1) =
∂CE(y, yˆ)
∂z(1) 
x
T
(6)
Similarly,
\partial CE(y, yˆ)
\partial b(1) =
\partial CE(y, yˆ)



```python
def python_backprop(cache, grads, y):
	# divide by batch size
	one_hot_y = np.zeros((cache['D_out'],cache['N']))
	one_hot_y[y,np.arange(cache['N'])] = np.ones(cache['N'])
	diff = (1.0 / cache['N']) * (cache['logits'] - one_hot_y)
	grads['dW2'] = mm(diff, cache['a1'].T) + cache['reg'] * cache['W2'] # 0.5 cancels
	#grads['db2'] = np.sum(diff, axis=1)
	grads['db2'] = np.sum(diff.T, axis=0).T

	da1 = mm( cache['W2'].T, diff )
	# elementwise
	dz1 = np.multiply(da1, np.multiply( cache['a1'], (1 - cache['a1']) ) )

	grads['dW1'] = mm(dz1, cache['X'].T ) + cache['reg'] * cache['W1']  # 0.5 cancels
	#grads['db1'] = np.sum(dz1, axis=1)
	grads['db1'] = np.sum(dz1.T, axis=0).T
```













## Implementing the same network in Pytorch


```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb
from torch.autograd import Variable
import numpy.matlib
import scipy.linalg
from numpy import matmul as mm


class PytorchFCNet(nn.Module):
	def __init__(self, cache):
		super(PytorchFCNet, self).__init__()
		self.cache = cache
		self.fc1 = nn.Linear( cache['D_in'], cache['H'])
		self.fc2 = nn.Linear( cache['H'], cache['D_out'])

		self.weights_init()

	def weights_init(self):
		for idx, m in enumerate( self.modules() ):
			#classname = m.__class__.__name__
			if isinstance(m, nn.Linear):
				print 'idx = %d' % idx
				if idx == 1:
					#m.weight.data = torch.from_numpy(self.cache['W1'])
					#m.bias.data = torch.from_numpy(self.cache['b1'])
					torch.nn.init.constant(m.weight[:,0], 1)
					torch.nn.init.constant(m.weight[:,1], 2)
					torch.nn.init.constant(m.weight[:,2], 3)
					torch.nn.init.constant(m.bias, 1)
				elif idx == 2:
					#m.weight.data = torch.from_numpy(self.cache['W2'])
					#m.bias.data = torch.from_numpy(self.cache['b2'])
					torch.nn.init.constant(m.weight, 2)
					torch.nn.init.constant(m.bias, 2)

		# # Create random Tensors for weights; setting requires_grad=True means that we
		# # want to compute gradients for these Tensors during the backward pass.
		# W1 = torch.fromnumpy(cache['W1'] , requires_grad=True)
		# b1 = torch.fromnumpy(cache['b1'] device=device, requires_grad=True)

		# W2 = torch.fromnumpy(cache['W2'] device=device, requires_grad=True)
		# b2 = torch.fromnumpy(cache['b2'] device=device, requires_grad=True)

		# W1 = torch.as_tensor(W1, device=device)
```

```python
	def forward(self, x):
		pdb.set_trace()
		x = self.fc1(x)
		x = F.sigmoid(x)
		x = self.fc2(x)
		return x
```

```python
def run_pytorch_fc_net(X, y, cache):

	model = PytorchFCNet(cache)
	#device = torch.device('cpu')

	# Create random Tensors to hold input and outputs
	X = torch.from_numpy(X.T) #, device=device)
	y = torch.from_numpy(y) #, device=device)

	X = Variable(X.type(torch.FloatTensor), volatile=False)
	y = Variable(y.type(torch.LongTensor), volatile=False)

	learning_rate = 5.0

	#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=cache['reg'])



	logits = model(X)
	logits = F.log_softmax(logits) #, dim=1)
	loss = F.nll_loss(logits, y, size_average=True)
	
	# exclude bias weights
	norms =  torch.norm( list(model.modules())[1].weight) **2
	norms += torch.norm( list(model.modules())[2].weight) **2

	loss += 0.5 * cache['reg'] * norms


	print loss # sum up batch loss

	# Use autograd to compute the backward pass. This call will compute the
	# gradient of loss with respect to all Tensors with requires_grad=True.
	# After this call w1.grad and w2.grad will be Tensors holding the gradient
	# of the loss with respect to w1 and w2 respectively.
	#optimizer.zero_grad()


	loss.backward()
	#optimizer.step()


	# optimizer weight_decay=0

	print 'W1 grad:'
	print list(model.modules())[1].weight.grad.data

	print 'b1 grad:'
	print list(model.modules())[1].bias.grad.data

	print 'W2 grad:'
	print list(model.modules())[2].weight.grad.data

	print 'b2 grad:'
	print list(model.modules())[2].bias.grad.data


	# print 'W1 weight:'
	# print list(model.modules())[1].weight.data

	# print 'b1 weight:'
	# print list(model.modules())[1].bias.data

	# print 'W2 weight:'
	# print list(model.modules())[2].weight.data

	# print 'b2 weight:'
	# print list(model.modules())[2].bias.data

	# Update weights using gradient descent. For this step we just want to mutate
	# the values of w1 and w2 in-place; we don't want to build up a computational
	# graph for the update steps, so we use the torch.no_grad() context manager
	# to prevent PyTorch from building a computational graph for the updates
	# with torch.no_grad():
	# list(model.modules())[1].weight.data -= learning_rate * list(model.modules())[1].weight.grad.data
	# list(model.modules())[2].weight.data -= learning_rate * list(model.modules())[2].weight.grad.data
```



```python
def run_python_fc_net(X,y, cache):
	learning_rate = 5.0

	grads = {}

	python_feedforward(X, cache)
	print 'logits:'
	print cache['logits']
	python_loss(cache, y)
	python_backprop(cache, grads, y)

	print 'W1 grad'
	print grads['dW1']

	print 'b1 grad'
	print grads['db1']

	print 'W2 grad'
	print grads['dW2']

	print 'b2 grad'
	print grads['db2']

	# # Update weights using gradient descent
	# cache['W1'] -= learning_rate * grads['dW1']
	# cache['b1'] -= learning_rate * grads['db1']

	# cache['W2'] -= learning_rate * grads['dW2']
	# cache['b2'] -= learning_rate * grads['db2']


	# print 'weight for W1', cache['W1']
	# print 'weight for b1', cache['b1']
	# print 'weight for W2', cache['W2']
	# print 'weight for b2', cache['b2']



def main():
	# N is batch size; D_in is input dimension;
	# H is hidden dimension; D_out is output dimension.
	N = 2
	D_in = 3
	H = 4
	D_out = 5

	cache = {}
	cache['reg'] = 0.5

	cache['N'] = N
	cache['D_in'] = D_in
	cache['H'] = H
	cache['D_out'] = D_out

	# Create random input and output data
	X = np.array(range(D_in*N))
	X = np.reshape(X, (D_in, N)) 
	y = np.array([3,4]) 
				#	[1,0],
				#	[0,1],
				#	[0,0]]	)


		#np.random.randn(D_out,N)

	# Randomly initialize weights
	cache['W1'] = np.ones((H, D_in))
	cache['W1'][:,1] *= 2.
	cache['W1'][:,2] *= 3.
	cache['W2'] = np.ones((D_out, H)) * 2.

	cache['b1'] = np.ones(cache['H'])
	cache['b2'] = np.ones(cache['D_out']) * 2.
	run_pytorch_fc_net(X, y, cache)
	run_python_fc_net(X, y, cache)
```


References:

(1) Eric Darve, Stanford University.




