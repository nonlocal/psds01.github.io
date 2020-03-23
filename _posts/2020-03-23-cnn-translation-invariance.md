# Translation Invariance in CNN

Review of the paper [On Translation Invariance in CNNs: Convolutional Layers can Exploit Absolute Spatial Location](https://arxiv.org/abs/2003.07064)


1. TOC
{:toc}

## Hypothesis
~~We have empirically observed that CNNs are translation invariant.~~ It is common assumption that CNNs are translation invariant. But CNNs can and will exploit absolute location by learning filters which respond to image bourndaries. This exploitation can be observed far from the image boundary because of the large filter sizes of modern/large CNNs.

## Background

Adding convolutiion to NN adds a(n) (visual) inductive bias: objects can appear anywhere. 

Convolution is (like) translation (equivariant) and if that is followed by an operation that does not depend on position, such as max and average, we get translation invariance.

??

## Contributions

1. locations specific filters can be/are learned.
2. this effect can be exploited far from the image boundaries.

## Related works

1. Fully connected and fully convolutional networks
    
    a. fully connected : AlexNet, VGG
    
    b. fully convolutional : Network in Network, All Convolutional Net, ResNet, Inception family, DenseNet, ResNext, 

2. Cropping image regions
  
    object detection, high-res image into patches, local image region matching, local CNN patch pooling encoders.
  
    Faster-R-CNN, BagNet
  
    Pooling methods: sum, BoW, VLAD, Fisher vector.
  
3. Image transformation robustness
    
    Accidental camera position should not affect the semantic content of an image/video. These transformations can be learned via data augmentation. Or via geometric adversarial training methods. But these are brute force solutions, not a constraint on the model's learning.

    But there are methods that make the models learn geometric transformations(rotation, scale, ) in an equivariant and invariant way.

4. Boundary effects
    They cause biases.
    
    Image restoration, deconvolution.
    
    This can be handled by learning separate filters at the boundary, trating boundary pixels as missing values, circular convolutions, etc.

5. Location information in CNNs

    Explicit location information helps in patch-matching, generative modelling, semantic segmentation, instance segmentation.


6. Visual inductive priors

    adding inductive priors increase data efficiency by tying parameters, sharing rotation responses, scale-space basis


## Encoding boundary location

1. Padding : for a filter of size 2k+1

    a. Valid : no padding : 
    
    b. Same : padding of size k surrounding the image 
    
    c. Full : padding of size 2k surrounding the image 
    
This is a very interesting(?) table: ![boundary effect illustration](/images/boundary_effect_illustration.png)

