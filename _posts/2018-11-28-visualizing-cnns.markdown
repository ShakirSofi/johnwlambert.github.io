





Question:
How is 1x1 convolution idential to a fully-connected layer


5x5 kernel, with 5x5 input. Elementwise multiply. Same as matrix multiplication because...


## Visualizing CNNs via deconvolution

It concerns slide 4 in this presentation.
http://places.csail.mit.edu/slide_iclr2015.pdf

I'm having a hard time understanding the deconvolution part of the slide.

Great question. You can see a presentation from Matt Zeiler on the method here: https://www.youtube.com/watch?v=ghEmQSxT6tw. He summarizes his method from about minutes 8:40-20:00 in the presentation.

 

Suppose there is some layer you want to visualize. Zeiler feeds in 50,000 ImageNet images through a learned convnet, and gets the activation at that layer for all of the images. Then they feed in the highest activations into their deconvolutional network.

 

Their inverse network needs to make max pooling and convolution reversible. So they use unpooling and deconvolution to go backwards. This is how they can visualize individual layers.


## Backprop per-layer equations

I would expect the quiz to include content from the lecture slides and what was discussed in lecture. Professor Hays didn't go into detail how to perform backprop through every single layer, so I wouldn't expect a detailed derivation for each layer. You can find the slides on backprop here.

https://www.cc.gatech.edu/~hays/compvision/lectures/20.pdf
 

If you're interested in digging deeper into the equations, some basic intuition for derivatives can be found [here](http://cs231n.github.io/optimization-2/) and [here](http://cs231n.stanford.edu/slides/2018/cs231n_2018_lecture04.pdf). The chain rules binds together the derivatives of each layer. For two simple examples, the (sub)gradient of a max of two arguments is 1 for the larger argument, 0 for the smaller argument. The derivative for the addition of two arguments is 1 with respect to each argument.




## FC vs. CONV2D MAXPOOL LINEAR

what are the reasons for not using fully connected layers and using conv2d+maxpool+linear layer combinations ? Is it only because the number in fully connected layers are very large for image processing to prohibits from learning the weights fast enough ? #vvm

 John Lambert
 John Lambert 18 hours ago We use convolutions and hierarchies of processing steps since we showed earlier in the course that this is the most effective way to work with image (gridded data).
 

We don't use fully-connected layers at every step because there would be way to many learnable parameters to learn without overfitting, and the memory size would be enormous. The fully-connected layers act as the classifier on top of the automatically-learned features.


From CS 231N at Stanford:
http://cs231n.github.io/convolutional-networks/

Regular Neural Nets. ...Neural Networks receive an input (a single vector), and transform it through a series of hidden layers. Each hidden layer is made up of a set of neurons, where each neuron is fully connected to all neurons in the previous layer, and where neurons in a single layer function completely independently and do not share any connections. The last fully-connected layer is called the “output layer” and in classification settings it represents the class scores.

 

Regular Neural Nets don’t scale well to full images. In CIFAR-10, images are only of size 32x32x3 (32 wide, 32 high, 3 color channels), so a single fully-connected neuron in a first hidden layer of a regular Neural Network would have 32*32*3 = 3072 weights. This amount still seems manageable, but clearly this fully-connected structure does not scale to larger images. For example, an image of more respectable size, e.g. 200x200x3, would lead to neurons that have 200*200*3 = 120,000 weights. Moreover, we would almost certainly want to have several such neurons, so the parameters would add up quickly! Clearly, this full connectivity is wasteful and the huge number of parameters would quickly lead to overfitting.

 

3D volumes of neurons. Convolutional Neural Networks take advantage of the fact that the input consists of images and they constrain the architecture in a more sensible way. In particular, unlike a regular Neural Network, the layers of a ConvNet have neurons arranged in 3 dimensions: width, height, depth. (Note that the word depthhere refers to the third dimension of an activation volume, not to the depth of a full Neural Network, which can refer to the total number of layers in a network.) For example, the input images in CIFAR-10 are an input volume of activations, and the volume has dimensions 32x32x3 (width, height, depth respectively). As we will soon see, the neurons in a layer will only be connected to a small region of the layer before it, instead of all of the neurons in a fully-connected manner. Moreover, the final output layer would for CIFAR-10 have dimensions 1x1x10, because by the end of the ConvNet architecture we will reduce the full image into a single vector of class scores, arranged along the depth dimension. 







## Why does Batch Norm help?
John: (adding this) I wouldn't expect batch norm to help by 15% on the simple network. Maybe by about 4% on the SimpleNet. Are you confounding the influence of normalization, more layers, data augmentation, and dropout with the influence of batch norm?

 

If you train your network to learn a mapping from X->Y, and then the distribution of X changes, then you might need to retrain your network so that it can understand the changed distribution of X. (Suppose you go learned to classify black cats, and then suddenly you need to classify colored cats, example here).
https://www.youtube.com/watch?v=nUUqwaxLnWs
 

Batch Norm speeds up learning. This is because it reduces the amount that the distribution of hidden values moves around. This is often called "reducing internal covariate shift”.  Since all the layers are linked, if the first layer changes, then every other layer was dependent on those values being similar to before (not suddenly huge or small). 

 

General ideas why Batch Norm helps:

Improves gradient flow through the network (want variance=1 in your layers, avoid exponentially vanishing or exploding dynamics in both the forward and the backward pass)
Allows higher learning rates
Reduces the strong dependence on initialization
Acts as a form of regularization because it adds a bit of noise to training (uses statistics from random mini-batches to normalize) and it slightly reduces the need for dropout
 
https://arxiv.org/pdf/1805.11604.pdf
Others say that BatchNorm makes the optimization landscape significantly smoother. They theorize that it reduces the Lipschitz constant of the loss function, meaning the loss changes at a smaller rate and the magnitudes of the gradients are smaller too.
https://en.wikipedia.org/wiki/Lipschitz_continuity
 
https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
In the AlexNet paper, the authors mentioned that local response normalization aids generalization, meaning that the network can accurately understand new examples.

## Where to add Dropout?
Dropout was used in older convolutional network architectures like AlexNet and VGG. Dropout should go in between the fully connected layers (also known as 1x1 convolutions). Dropout doesn't seem to help in the other convolutional layers.  It's not exactly clear why.

 

One hypothesis is that you only need to avoid overfitting in the layers with huge amounts of parameters (generally fully-connected layers), and convolutional layers usually have fewer parameters (just a few shared kernels) so there's less need to avoid overfitting there. Of course, you could have the same number of parameters in both if you had very deep filter banks of kernels, but usually the max filter depth in VGG is only 512 and 384 in AlexNet.

 

ResNet, a more modern convnet architecture, does not use dropout but rather uses BatchNorm.



## Convolutions

Could somebody post answers for Lecture-4 slides 12,13 and Lecture-5 slides 6?


Lecture 4 Slide 12:

(2) this is forward difference derivative approximation.
https://en.wikipedia.org/wiki/Finite_difference#Forward,_backward,_and_central_differences
 

Lecture 4 Slide 13

 

a) y-derivatives of image with Sobel operator

b) derivative of Gaussian computed with Sobel operator

c) image shifted with translated identity filter

 

For explanations of Lecture 5 Slide 6,FOURIER MAGNITUDE IMAGES
 believe the answers are the following:

1 and 3 are distinctive because they are not natural images.

1D -- Gaussian stays as a circle in the Fourier space.

3A -- Sobel has two separated ellipses.

 

2,4,5 are all similar because natural images have fairly similar Fourier magnitude images

 

2B -- flower image has an even distribution of frequencies, so we see an even circular distribution in all directions in Fourier space.

4E -- because we have only lines along the x-axis, so we see a line only on the y-axis in the Fourier amplitude image.

5C -- because strong x,y-axis-aligned lines in the natural image related to x,y-axis-aligned lines in the Fourier magnitude images
For Lecture 4 Slide 12, I have:

 

1. 

 

 0 -1  0

-1  4 -1

 0 -1  0

 

2.

 

0 -1 1

 

For Lecture 4 Slide 13, I have:

 

a) G = D * B

b) A = B * C

c) F = D * E

d) I = D * D

 

For Lecture 5 Slide 6, I have:

 

1 - D

2 - B

3 - A

4 - E

5 - C

 

I'm not sure if I can explain why all of those are the way they are, but I hope this helps!



$$ Normalizing SIFT


You're welcome to experiment with the choice of norm.  However, normally when we say that we'll normalize every feature vector on its own, we mean normalizing descriptor $$D$$ such that $$D_{normalized} = \frac{D}{\|D\|_2}$$.



We do this because feature descriptors are points in high-dimensional space, so we want them all to have the same length. That way the distance between them is based only upon the different angles the vectors point in, rather than considering their different length. 

 
## More efficient calculation of sift descriptor

I'm finding it slightly hard to implement an efficient algorithm for calculating the sift descriptor. The way I'm doing it now is basically like this;

- Calculate all the gradients and their angle

- Loop over all interest points

- Loop over the rows of the 4x4 cell 

- Loop over the columns of the 4x4 cell

- For each cell index, compute the histogram and save it in that cell

 

I'm feeling like there might be a nice numpy-way to get around my two inner loops (the ones over the cell). Is this possible? Any ideas or tips?

 

For reference: it can compute the descriptor for around 1000 interest points in one second, don't know if that's sufficiently fast?

 



One good way to reduce your 3 for-loops into 2 for-loops would be to do the following:



Instead of:

for interest point

      for row

            for col



You could do:

for interest point

       for 4x4 patch in 16x16 window



Creating those patches can be done by reshaping and swapping the axes. For example, if you had an array x like

x = np.reshape(np.array(range(16)),(4,4))
It would look like

array([[ 0,  1,  2,  3],

       [ 4,  5,  6,  7],

       [ 8,  9, 10, 11],

       [12, 13, 14, 15]])

You could break it into 2 parts along dimension 0:

 x.reshape(2,-1)
You'd get

array([[ 0,  1,  2,  3,  4,  5,  6,  7],

       [ 8,  9, 10, 11, 12, 13, 14, 15]])

If you kept breaking the innermost array into 2 arrays you'd get

x.reshape(2,2,-1)
array([[[ 0,  1,  2,  3],

        [ 4,  5,  6,  7]],



       [[ 8,  9, 10, 11],

        [12, 13, 14, 15]]])

and then finally you would get

x.reshape(2,2,2,2)
array([[[[ 0,  1],

         [ 2,  3]],

        [[ 4,  5],

         [ 6,  7]]],



       [[[ 8,  9],

         [10, 11]],

        [[12, 13],

         [14, 15]]]])

At this point you have 2 cubes that are 2x2x2. There are 3 ways you could look at this cube: there are 2 planes along the x-direction, or 2 planes along the y-direction, or 2 planes along the z-direction. If you swap the direction from which you look at the cube (swapaxes), you could now have

array([[[[ 0,  1],

         [ 4,  5]],

        [[ 2,  3],

         [ 6,  7]]],



       [[[ 8,  9],

         [12, 13]],

        [[10, 11],

         [14, 15]]]])

which you'll notice is effectively all of the original 2x2 patches with stride 2 from the original 4x4 matrix.



Sad to say it, but the speedup will probably be completely unnoticeable since the for loop is only over 4 iterations, as opposed to something over 40,000 iterations




## Visualizing SIFT

Your get_features() function should return a NumPy array of shape (k, feat_dim) representing k stacked feature vectors (row-vectors), where "feat_dim" is the feature_dimensionality (e.g. 128 for standard SIFT).


Since this is a 2D matrix, we can treat it as a grayscale image. Each row in the image would correspond to the feature vector for one interest point. We would hope that each feature vector would unique, so the image shouldn't be completely uniform in color (all identical features) or completely black (all zero values). That would be a clue that your features are degenerate.



For example, you might see something like this if you were to call:

import matplotlib.pyplot as plt; plt.imshow(image1_features); plt.show()


https://d1b10bmlvqabco.cloudfront.net/attach/jl1qtqdkuye2rp/jl1r1s4npvog2/jm4fywn2xohb/features.png


## Trilinear Interpolation

http://paulbourke.net/miscellaneous/interpolation/

On slide 30 of Lecture 7, Professor Hays was discussing trilinear interpolation. Trilinear interpolation is the name given to the process of linearly interpolating points within a box (3D) given values at the vertices of the box. We can think of our histogram as a 3D spatial histogram with $$N_{\theta} \times N_x \times N_y$$, bins usually $$8 \times 4 \times 4$$.
https://www.cc.gatech.edu/~hays/compvision/lectures/07.pdf


You aren't required to implement the trilinear interpolation for this project, but you may if you wish. I would recommend getting a baseline working first where the x and y derivatives at each pixel $$I_x, I_y$$ form 1 orientation, and that orientation goes into a single bin.



Then you could try the trilinear interpolation afterwards once that is working (without trilinear interpolation, you can still get >>80% accuracy on Notre Dame).



## nearest neighbor distance ratio algorithm 4.18
1) in the formula 4.18, what is meant by target descriptor, here it is Da ?

2) In the formula 4.18 what is meant by Db and Dc as being descriptors?

3)  What makes Da to be target descriptor and Db and Dc nearest neighbors ?

4) In formular 4.18 ||Da - Db|| is norm? or euclidean distance ?

5) how is value of the descriptor such as Da related to the euclidean distance to Db?

6) Is there such a thing as x and y coordinates of the center for a specific descriptor that helps to calculate the distances between the 

descriptors ? if so how to calculate the center of each descriptor ?




1) in the formula 4.18, what is meant by target descriptor, here it is Da ?



$$D_A$$ is a high-dimensional point. In the case of SIFT descriptors, $$D_A$$ would be the SIFT feature vector in $$R^{128}$$



2) In the formula 4.18 what is meant by Db and Dc as being descriptors?



$$D_B, D_C$$ are high-dimensional points. These are the closest points, as measured by the $$\ell_2$$ norm, from $$D_A$$ in this high dimensional space.



3)  What makes Da to be target descriptor and Db and Dc nearest neighbors ?



$$D_A$$ is the feature vector corresponding to an (x,y) location, for which we are trying to find matches in another image. $$D_A$$ could be from Image1, and $$D_B, D_C$$ might be feature vectors corresponding to points in Image2



4) In formula 4.18 ||Da - Db|| is norm? or euclidean distance ?



$$\|D_A-D_B\|$$ is the $$\ell_2$$ norm, which we often call the Euclidean norm.



5) how is value of the descriptor such as Da related to the euclidean distance to Db?



Euclidean distance from $$D_B$$ to $$D_A$$ depends upon knowing the location of $$D_B,D_A$$ in $$R^{128}$$.



6) Is there such a thing as x and y coordinates of the center for a specific descriptor that helps to calculate the distances between the 

descriptors ? if so how to calculate the center of each descriptor ?



Each SIFT descriptor corresponds to an (x,y) point. We form the SIFT descriptor by looking at a 16x16 patch, centered at that (x,y) location. This could be considered a "center" of the descriptor, although I think using that terminology could be confusing since the (x,y)  "center" location in $$R^2$$ is not necessarily related at all to center of the $$R^{128}$$ space.  It may be possible to use the spatial information in $$R^2$$ to verify the matching of points in $$R^{128}$$.


## Arctan vs. Arctan2

I would use arctan2 instead of arctan. Because of the sign ambiguity, a function cannot determine with certainty in which quadrant the angle falls only by its tangent value. 
https://stackoverflow.com/questions/283406/what-is-the-difference-between-atan-and-atan2-in-c


Numpy arctan2 returns an "Array of angles in radians, in the range [-pi, pi]." (see here).
https://docs.scipy.org/doc/numpy/reference/generated/numpy.arctan2.html


As long as you bin all of the gradient orientations consistently, it turns out it doesn't matter if the histograms are created from 8 uniform intervals from [0,360] or 8 uniform intervals from [-180,180]. Histograms are empirical samples of a distribution, and translating the range won't affect that distribution.



More on arctan2 from StackOverflow:


From school mathematics we know that the tangent has the definition

<em><code>tan(α) = sin(α) / cos(α)</code></em>
and we differentiate between four quadrants based on the angle that we supply to the functions. The sign of the sin, cos and tan have the following relationship (where we neglect the exact multiples of π/2):

<em><code>  Quadrant    Angle              sin   cos   tan
-------------------------------------------------
  I           0    < α < π/2      +     +     +
  II          π/2  < α < π        +     -     -
  III         π    < α < 3π/2     -     -     +
  IV          3π/2 < α < 2π       -     +     -</code></em>
Given that the value of tan(α) is positive, we cannot distinguish, whether the angle was from the first or third quadrant and if it is negative, it could come from the second or fourth quadrant. So by convention, atan() returns an angle from the first or fourth quadrant (i.e. -π/2 <= atan() <= π/2), regardless of the original input to the tangent.

In order to get back the full information, we must not use the result of the division sin(α) / cos(α) but we have to look at the values of the sine and cosine separately. And this is what atan2() does. It takes both, the sin(α) and cos(α) and resolves all four quadrants by adding π to the result of atan() whenever the cosine is negative.








