# Densefuse: A Fusion Approach for Infrared and Visible Image Fusion
Infrared and visible image fusion using CNN layers and dense block architecture.

## Abstract
In this paper, we present a novel deep learning architecture for infrared and visible images fusion problem. 

In contrast to conventional convolutional networks, our encoding network is combined by convolutional neural network layer and dense block which the output of each layer is connected to every other layer. We attempt to use this architecture to get more useful features from source images in encoder process. Then appropriate fusion strategy is utilized to fuse these features. Finally, the fused image is reconstructed by decoder. 

Compare with existing fusion methods, the proposed fusion method achieves state-of-the-art performance in objective and subjective assessment.

### The framework of fusion method
![](https://github.com/exceptionLi/imagefusion_densefuse/blob/master/figures/framework.png)

### Training process
![](https://github.com/exceptionLi/imagefusion_densefuse/blob/master/figures/train.png)

### Fusion strategy - addition
![](https://github.com/exceptionLi/imagefusion_densefuse/blob/master/figures/fuse_addition.png)

### Fusion strategy - l1-norm
![](https://github.com/exceptionLi/imagefusion_densefuse/blob/master/figures/fuse_l1norm.png)
