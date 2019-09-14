---
layout: post
title:  "PyTorch Tutorial"
permalink: /pytorch-tutorial/
excerpt: "PyTorch Tutorial"
mathjax: true
date:   2019-09-13 11:00:00
mathjax: true

---
This tutorial was contributed by John Lambert.

This tutorial will serve as a crash course for those of you not familiar with PyTorch. It is written in the spirit of [this Python/Numpy tutorial](http://cs231n.github.io/python-numpy-tutorial/).

We will be focusing on CPU functionality in PyTorch, not GPU functionality, in this tutorial. We'll be working with PyTorch 1.1.0, in these examples.

Table of Contents:
- [PyTorch Tensors](#tensors)
	- [Creating a tensor](#create-tensor)
	- [Data types in Pytorch and Casting](#data-types)
- [Operations on Tensors](#tensor-ops)
	- [Tensor Indexing ](#indexing)
	- [Reshaping tensors](#reshaping)
	- [Tensor Arithmetic](#arithmetic)
	- [Matrix Multiplication vs. Elementwise Multiplication](#mat-mul)
	- [Other helpful transcendental functions ](#transcendental)
	- [Combining Tensors](#combining-tensors)
	- [Logical Operations](#logical-ops)
	- [Sorting Operations](#sorting-ops)
- [Conv Layers](#conv-layers)
	- [Weights for Convolutional layers](#conv-weights)
	- [Groups in Conv Layers](#conv-groups)
	- [Bias in Convolutional Layers](#conv-bias)
- [Max-Pooling layers](#maxpool)
- [Creating a Model](#creating-model)
	- [Creating a Pytorch Module, Weight Initialization](#custommodule)
	- [Executing a forward pass through the model](#forward)
	- [Instantiate Models and iterating over their modules](#iterate-modules)
	- [Sequential Networks](#sequential)


<a name='tensors'></a>
## PyTorch Tensors
PyTorch's fundamental data structure is the `torch.Tensor`, an n-dimensional array. You may be more familiar with matrices, which are 2-dimensional tensors, or vectors, which are 1-dimensional tensors.

<a name='create-tensor'></a>
### Creating a tensor
```python
import numpy as np
import torch

x = torch.Tensor([1., 2., 3.])
print(x) # Prints "tensor([1., 2., 3.])"
print(x.shape) # Prints "torch.Size([3])"
print(torch.ones(2,1)) # Prints "tensor([[1.],[1.]])"
print(torch.zeros_like(x)) # Prints "tensor([0., 0., 0.])"

# Alternatively, create a tensor by bringing it in from Numpy
y = np.array([0,1,2,3])
print(y) # Prints "[0 1 2 3]"
x = torch.from_numpy(y)
print(x) # Prints "tensor([0, 1, 2, 3])"
```

<a name='data-types'></a>
### Data types in Pytorch and Casting
You'll have a wide range of data types at your disposal, including:

| Data Type Name           | Code keywords (all equivalent)                 |
|--------------------------|------------------------------------------------|
|32-bit floating point     |torch.float32, torch.float, torch.FloatTensor   |
|64-bit floating point     |torch.float64, torch.double, torch.DoubleTensor |
|8-bit integer (unsigned)  | torch.uint8, torch.ByteTensor                  |
|8-bit integer (signed)    |torch.int8, torch.CharTensor                    |
|32-bit integer (signed)   | torch.int32, torch.int, torch.IntTensor        |
|64-bit integer (signed)   |torch.int64, torch.long, torch.LongTensor       |
|Boolean                   | torch.bool, torch.BoolTensor                   |
    
A tensor can be cast to any data type, with possible loss of precision:
```python
print(x.dtype) # Prints "torch.int64", currently 64-bit integer type
x = x.type(torch.FloatTensor)
print(x.dtype) # Prints "torch.float32", now 32-bit float
print(x.float()) # Still "torch.float32"
print(x.type(torch.DoubleTensor)) # Prints "tensor([0., 1., 2., 3.], dtype=torch.float64)"
print(x.type(torch.LongTensor)) # Cast back to int-64, prints "tensor([0, 1, 2, 3])"
```

<a name='tensor-ops'></a>
<a name='indexing'></a>
### Tensor Indexing 
Tensors can be indexed using MATLAB/Numpy-style n-dimensional array indexing. An RGB image is a 3-dimensional array. For a 2 pixel by 2 pixel RGB image, in CHW order, the image tensor would have dimensions (3,2,2). In HWC order, the image tensor would have dimensions (2,2,3). In NCHW order, the image tensor would have shape (1,3,2,2). *N* represents the batch dimension (number of images present), *C* represents the number of channels, and *H,W* represent height and width.
```python
x = torch.arange(4*3) # Create an array of numbers [0,1,2,...,11]
print(x) # Prints "tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])"
x = x.reshape(3,2,2) # Reshape tensor from (12,) to (3,2,2)
print(x[0,:,:]) # Prints 0th channel image, "tensor([[0, 1], [2, 3]])"
print(x[1,:,:]) # Prints 1st channel image, "tensor([[4, 5], [6, 7]])"
print(x[2,:,:]) # Prints 2nd channel image, "tensor([[8, 9],[10, 11]])"

# Index instead to get ALL channels at (0,0) pixel for 0th row, 0th col.
print(x[:,0,0]) # Prints "tensor([0, 4, 8])"
```

<a name='reshaping'></a>
### Reshaping tensors
 Above, we used `reshape()` to modify the shape of a tensor. Note that a reshape is valid only if we do not change the total number of elements in the tensor. For example, a (12,1)-shaped tensor can be reshaped to (3,2,2) since $$12*1=3*2*2$$. Here are a few other useful tensor-shaping operations:
```python
print(x.squeeze().shape) # Prints "torch.Size([3, 2, 2])"
print(x.unsqueeze(0).shape) # Add batch dimension for NCHW, prints "torch.Size([1, 3, 2, 2])"
print(x.view(6,2).shape) # Prints "torch.Size([6, 2])"
print(x.reshape(6,2).shape) # Prints "torch.Size([6, 2])"
print(x.flatten().shape) # Reshape back to flat vector, prints "torch.Size([12])"
```
<a name='arithmetic'></a>
### Tensor Arithmetic
Typical Python or Numpy operators such as *+,-* can be used for arithmetic, or explicit PyTorch operators:
```python
x = torch.tensor([1,2,3])
y = torch.tensor([2,2,2])
print(torch.add(x,y)) # Prints "tensor([3, 4, 5])"
print(x + y) #  Above is identical to using "+" op, prints "tensor([3, 4, 5])"
print(x.add(y)) # Prints "tensor([3, 4, 5])"
print(x.sub(y)) # Prints "tensor([-1,  0,  1])"
print(x - y) # Prints "tensor([-1,  0,  1])"
```

<a name='mat-mul'></a>
## Matrix Multiplication vs. Elementwise Multiplication
Note that the operator $$*$$ will not perform matrix multiplication -- rather, it will perform elementwise multiplication, such as in Numpy:
```python
x = torch.tensor([[1,2,3]])
y = torch.tensor([[2,2,2]])
print(x.shape) # Prints "torch.Size([1, 3])"
print(x.t().shape) # Take matrix transpose, prints torch.Size([3, 1])
print(y.shape) # Prints "torch.Size([1, 3])"
print(torch.mul(x,y)) # Elementwise multiplication of arrays, prints "tensor([[2, 4, 6]])"
print(x.t().mm(y).shape) # Outer product of (3,1) and (1,3), prints "torch.Size([3, 3])"
print(x.mm(y.t())) # Dot product of (1,3) and (3,1), prints "tensor([[12]])"
print(torch.mm(x,y.t())) # Same dot product/inner product, prints "tensor([[12]])""
print(torch.matmul(x,y.t())) # Identical to above, prints "tensor([[12]])"
```

<a name='transcendental'></a>
### Other helpful transcendental functions: 
PyTorch supports cosine, sine, and exponential operations with `cos()`, `sin()`, `exp()`, just like Numpy. The function input must be a PyTorch tensor:
```python
print(np.pi) # Prints float as "3.141592653589793"
# print(torch.cos(np.pi)) # Will crash with TypeError, since np.pi is a float, not tensor
print(torch.cos(torch.tensor(np.pi))) # Prints "tensor(-1.)"
# print(torch.cos(torch.tensor(0))) # Will crash, Prints "RuntimeError: cos_vml_cpu not implemented for 'Long'""
print(torch.sin(torch.tensor(np.pi/2))) # Must use float as argument, prints "tensor(1.)"
print(torch.exp(torch.tensor(1.))) # Prints Euler's number e as "tensor(2.7183)"
```

<a name='combining-tensors'></a>
### Combining Tensors
Tensors can be combined along any dimension, as long as the dimensions align properly. *Concatenating* (`torch.cat()`) or *stacking* (`torch.stack()`) tensors are considered different operations in PyTorch. `torch.stack()` will combine a sequence of tensors along a new dimension, whereas `torch.cat()` will concatenates tensors along a default dimension *dim=0*:
```python
x = torch.tensor([[1,2,3]])
y = torch.tensor([[2,2,2]])
print(x.shape) # Prints torch.Size([1, 3]), x is a row vector
print(y.shape) # Prints torch.Size([1, 3]), y is a row vector
print(torch.stack([x,y]).shape) # prints torch.Size([2, 1, 3])
print(torch.cat([x,y]).shape) #  prints "torch.Size([2, 3])"
```

<a name='logical-ops'></a>
### Logical Operations
Logical operations like AND, OR, etc. can be computed on PyTorch tensors:
```python
x = torch.tensor([True, False, False]) # Create uint8/Byte tensor
y = torch.tensor([True, True, True]) # Create uint8/Byte tensor
print(x.dtype, y.dtype) # Prints "torch.uint8 torch.uint8"
print(x & y) # Logical and op., prints "tensor([1, 0, 0], dtype=torch.uint8)"
print(x | y) # Logical or op., prints "tensor([1, 1, 1], dtype=torch.uint8)""

x = torch.tensor([0,1,2])
y = torch.tensor([2,3,4])
cond = (x > 1) & (y > 1) # Create a condition with logical `AND`
print(cond) # Condition valid only at last index, prints "tensor([0, 0, 1], dtype=torch.uint8)"
```

### More logical operations. 
`torch.where(condition, input, other)` looks at each index $$i$$ of 3 input tensors. It will return $$input_i$$ if $$condition_i$$ true, else will return $$other_i$$:
```python
x = torch.tensor([0,1,2]) # Only at 1st index is element > 0 and < 2
y = torch.tensor([6,5,4])
print(torch.where((x > 0) & (x < 2), x, y)) # Prints "tensor([6, 1, 4])"
```

Logical operations can be combined with `tensor.nonzero()` to retrieve relevant
array indices. `.nonzero()` will return the indices of all non-zero elements of the input. It can be used much like `np.argwhere()`, in the following manner:
```python
x = torch.tensor([0,1,3,3]) # At index i=2 and i=3, x[i]=3
print((x == 3).nonzero()) # Prints "tensor([[2],[3]])"
```

<a name='sorting-ops'></a>
### Sorting Operations
Indices to sort an array can be computed:
```python
x = torch.tensor([3,0,1,2])
print(torch.argsort(x)) # Prints "tensor([1, 2, 3, 0])", representing indices to sort x
```

<a name='conv-layers'></a>
<a name='conv-weights'></a>
### Weights for Convolutional layers
PyTorch convolutional layers require 4-dimensional inputs, in *NCHW* order. As mentioned above, *N* represents the batch dimension, *C* represents the channel dimension, *H* represents the image height (number of rows), and *W* represents the image width (number of columns).
```python
x = torch.ones(12).reshape(1,3,2,2) # Represents 3-channel image, each image has dims (2,2)
x = x.float() # Convert to float32 since Conv2d cannot accept `Long` type
print(x[:,0,:,:]) # Prints "tensor([[[0, 1],[2, 3]]])" as channel 0
print(x[:,1,:,:]) # Prints "tensor([[[4, 5],[6, 7]]])" as channel 1
print(x[:,2,:,:]) # Prints "tensor([[[8, 9],[10, 11]]])"" as channel 2

conv_1group = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, groups=1, bias=False)
print(conv_1group.weight.shape) # 3 filters, each of size (3 x 1 x 1), prints "torch.Size([3, 3, 1, 1])"
print(conv_1group.bias) # Prints "None", since no bias here

w = torch.tensor([1.,1.,1.,2.,2.,2.,3.,3.,3.]).reshape(3,3,1,1)
print(w[0,:,0,0]) # 0th filter, prints "tensor([1., 1., 1.])"
print(w[1,:,0,0]) # 1st filter, prints "tensor([2., 2., 2.])"
print(w[2,:,0,0]) # 2nd filter, prints "tensor([3., 3., 3.])"
conv_1group.weight = torch.nn.Parameter(w) # Initialize the layer weight

# Perform 1x1 convolution, i.e. the dot product
#  of [1,1,1] w/ [1,1,1] = 3, and [1,1,1] w/ [2,2,2] = 6, etc.
print(conv_1group(x)) # Prints "tensor([[[[3.,3.],[3.,3.]],[[6.,6.],[6.,6.]],[[9.,9.],[9.,9.]]]],grad_fn=<MkldnnConvolutionBackward>)"
```

<a name='conv-groups'></a>
### Groups in Conv Layers
Convolutional filters can be applied along a single channel, instead of over all channels, when groups is set to the number of channels.

The [official documentation states](https://pytorch.org/docs/stable/nn.html#torch.nn.Conv2d) that when *groups= in_channels, each input channel is convolved with its own set of filters*.

```python
conv_3groups = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, groups=3, bias=False)
print(conv_3groups.weight.shape) # 3 filters, each of size (1 x 1 x 1) now, prints "torch.Size([3, 1, 1, 1])"

w = torch.tensor([1.,2.,3.]).float().reshape(3,1,1,1) # Create new weight, scalar 1 is for channel 0, etc.
conv_3groups.weight = torch.nn.Parameter(w) # Initialize CONV layer weight w/ Parameter
print(conv_3groups.weight) # Prints "Parameter containing: tensor([[[[1.]]],[[[2.]]],[[[3.]]]], requires_grad=True)"

# Input split into 3 groups, and conv layer split into 3 groups (along channel dim.)
# Each pixel in channel 0 multiplied with 1, each pixel in channel 1 multiplied with 2, etc.
print(x) # prints "tensor([[[[1.,1.],[1.,1.]],[[1.,1.],[1.,1.]],[[1.,1.],[1.,1.]]]])"
print(x.shape) # Prints "torch.Size([1, 3, 2, 2])"
print(conv_3groups(x)) # Prints "tensor([[[[1.,1.],[1.,1.]],[[2.,2.],[2.,2.]],[[3.,3.],[3.,3.]]]], grad_fn=<MkldnnConvolutionBackward>)
```

<a name='conv-bias'></a>
### Bias in Convolutional Layers
Above, we set the bias to false in our conv layers. A bias will add an offset to the dot product result -- below is an example when *bias=5* for each of 3 filters. Bias is initialized randomly when the layer is constructed with `bias=True`.
```python
conv_3groups = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, groups=3, bias=True)
print(conv_3groups.bias) # Prints "Parameter containing:tensor([0.0452, 0.9189, 0.7354], requires_grad=True)"
print(conv_3groups.bias.shape) # One bias term to add for each of 3 filters, prints "torch.Size([3])"
w = torch.tensor([1.,2.,3.]).float().reshape(3,1,1,1) # Conv weight w as above
conv_3groups.weight = torch.nn.Parameter(w)
b = torch.ones(3) * 5 # Fill b w/ 5's -- will add 5 to result of each 1x1 convolution
conv_3groups.bias = torch.nn.Parameter(b)
print(conv_3groups(x)) # Prints "tensor([[[[6.,6.],[6.,6.]],[[7.,7.],[7.,7.]],[[8.,8.],[8.,8.]]]],grad_fn=<MkldnnConvolutionBackward>)"
```

<a name='maxpool'></a>
### Max-Pooling layers
A 2d max-pooling layer will slide a small window over the 2d feature map slices (each channel viewed independently) and will output the largest value in the window:

For example, a max-pooling layer with kernel_size=2 will slide a 2x2 window over the 2d feature maps. With stride=2, this window will be shifted over by 2 pixels along any axis before the subsequent computation. With stride=1, this window will be placed at every possible position (shifted over by 1 pixel at a time). With kernel size $$>1$$, in order to preserve the input size, one must pad the input using the `padding` argument to the convolution op.

For a 4x4 image, a max-pooling kernel of $$2x2$$ and stride 2 will output the max in each 2x2 quadrant of the image:
```python
x = torch.tensor([[1.,2.,3.,4.],[1.,2.,3.,4.],[1.,2.,3.,4.],[1.,2.,3.,4.]]).reshape(1,1,4,4) # 1-channel, 4x4 image
maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2) # Slide `kernel` every 2 pixels
print(maxpool(x)) #  prints "tensor([[[[2., 4.],[2., 4.]]]])"

maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=1) # Step one pixel at a time, taking max in 2x2 cell
print(maxpool(x).shape) # Get back 3x3 image since no padding, prints "torch.Size([1, 1, 3, 3])"
print(maxpool(x)) # Now prints "tensor([[[[2., 3., 4.],[2., 3., 4.],[2., 3., 4.]]]])"

maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=1, padding=1) # Try padding
# Even kernel w/ padding kernel_size//2 will make size increase by 1,
print(maxpool(x).shape) # Prints "torch.Size([1, 1, 5, 5])"
```

<a name='creating-model'></a>
<a name='custommodule'></a>
### Creating a Pytorch Module, Weight Initialization
To define a custom layer, you'll define a class that inherits from `torch.nn.Module`. The class will require a constructor, which should be implemented with `__init__()` in Python.

Consider a simple layer that applies a single convolutional filter to a 3-channel input. For *kernel_size=2*, a filter (a cube of shape 3x2x2) will be slided over the input at default *stride=1*.

Conv layer weights are randomly initialized by default, but can be explicitly specified in a number of ways. In order to initialize all weight values to a constant value, or to draw them from a specific type of distribution, `torch.nn.init()` may be used.

To initialize weight values to a **specific** tensor, the tensor must be wrapped inside a PyTorch `Parameter`, meaning *a kind of Tensor that is to be considered a module parameter* (a special subclass of Tensor that will make the tensor appear in the module's `.parameters()`).


```python
import torch.nn as nn

class MyNewModule(nn.Module):
	def __init__(self):
		super(MyNewModule, self).__init__()
		self.conv = torch.nn.Conv2d(in_channels=3, out_channels=1, kernel_size=2, bias=False)
		self._initialize_weights()

	def _initialize_weights(self):
		# Starts with random weights
		print(self.conv.weight.shape) #  prints "torch.Size([1, 3, 2, 2])"
		print(self.conv.weight) # Prints tensor([[[[ 0.2,0.2],[ 0.0,0.1]],...,[[0.0,-0.1],[-0.0,0.2]]]], requires_grad=True)
		nn.init.constant_(self.conv.weight, 1.) # Fill weight with all ones
		print(self.conv.weight) # Prints "Parameter containing:tensor([[[[1.,1.],[1.,1.]],...[[1.,1.],[1.,1.]]]], requires_grad=True)"
		w = torch.arange(12).float().reshape(1,3,2,2)
		self.conv.weight = torch.nn.Parameter(w) # Insert entirely new parameter
		print(self.conv.weight) # Prints "Parameter containing:tensor([[[[ 0.,1.],[ 2.,3.]],...[10.,11.]]]], requires_grad=True)"

	def forward(self, x): # Define the behavior for the "forward" pass
		return self.conv(x)
```

<a name='forward'></a>
### Executing a forward pass through the model
The `forward()` function of a model can be executed on `x` as follows. With an input of all ones with shape 1x3x2x2, and a filter representing a 3x2x2 cube with numbers [0,1,2,3,...,10,11], the filter can only be applied in a single location, computing a single dot product of 
$$[1,1,1,1,...,1,1,1] \cdot [0,1,2,3,...,9,10,11] = 66$$
```python
model = MyNewModule() 
x = torch.ones(1,3,2,2) # Fill input with all ones
print(model(x)) # Prints tensor([[[[66.]]]], grad_fn=<MkldnnConvolutionBackward>)
```

<a name='iterate-modules'></a>
### Instantiate Models and iterating over their modules
The modules and parameters of a model can be inspected by iterating over the relevant iterators, which may be useful for debugging:
```python
for m in model.modules():
	print(m) # Prints "MyNewModule( (conv): Conv2d(3, 1, kernel_size=(2, 2), stride=(1, 1), bias=False)), ..."

for name, param in model.named_parameters(): # Iterate over parameters
	print(name, param) # Prints "conv.weight Parameter containing: tensor([[[[ 0.,  1.],[ 2.,  3.]],...[10., 11.]]]], requires_grad=True)"
```

<a name='sequential'></a>
### Sequential Networks
A number of different operations can be stacked into a single, sequential network with `nn.Sequential()`. In `nn.Sequential`, the `nn.Module`'s stored inside are connected in a cascaded way. For example, to define a network that applied a convolution and then a max-pooling operation, we could pass these layers to `nn.Sequential()`. 

For a single-channel input, with 1x1 convolution with filter weights all equal to 2, the operation will double every pixel's value. In this case, the dot product is over a 1-dimensional input, so the dot product involves only multiplication, not sum. After subsequent max-pooling of *kernel_size* 2x2 at *stride=2*, a 1x1x2x2 tensor will be reduced to a single number, 1x1x1x1, as follows:
```python
conv = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, bias=False)
nn.init.constant_(conv.weight, 2.)
maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
net = nn.Sequential(conv,maxpool)

x = torch.tensor([[1.,2.],[3.,1.]]).reshape(1,1,2,2)
print(net(x)) # Prints "tensor([[[[6.]]]], grad_fn=<MaxPool2DWithIndicesBackward>)"
```

