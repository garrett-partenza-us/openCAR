# OpenCAR

### PyTorch implementation of the super resolution paper "Learned Image Downscaling for Upscaling using Content Adaptive Resampler"

[Original Paper](https://arxiv.org/abs/1907.12904)

The original PyTorch implementation, which was opened source by the authors, does not include the necessary code for training your own models. More specifically, the CUDA kernels and custom PyTorch module the authors opened sourced raises a NotImplemented error when attempting to propegate gradients through the graph with loss.backward(). You can find this issue on line 32 of the author's [gridsampler.py](https://github.com/sunwj/CAR/blob/master/adaptive_gridsampler/gridsampler.py).

So far, this repository implements the authors model entirely in the PyTorch machine learning framework. The end goal is to rewrite the downsampler code as CUDA kernels to speed up training. Pure PyTorch implementation takes around one minute for a single optimization step with a 96x96 image patch on a k40.

To do...
1. Add image data augmentation (random flips, etc)
2. Determine max batch size for training
3. Add spatial offset loss (equation 9)
4. Add beta paramters of 0.9 and 0.999 to Adam optimizer
5. Add validation loss in traing loop
6. Change learning rate to 10e-4 and decrease after validation performance stagnates for 100 epochs
7. Understand and verify the gradient formulas in equations 4-6 match with torch gradient graph
8. Correct for ksize and offset unit parameters in 4.1.2
