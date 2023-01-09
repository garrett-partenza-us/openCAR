Open sourcing the training code for content adaptive resampling. 

To do...
1. Convert kernel application script to nn.Module called Downsampler
2. Write training script which trains on randomly cropped 96x96 patches with random flips
3. Add sigmoid to ESPCNN output and remove clamps and round which break gradient
4. Increase batch size to maximum 8GB GPU can handle
5. Add spatial offset loss (equation 9)
6. Add total variation loss (equation 10)
7. Add beta paramters of 0.9 and 0.999 to Adam optimizer
8. Change learning rate to 10e-4 and decrease after validation performance stagnates for 100 epochs
9. Add soft round after downscaling module to simulate real low resolution image input
10. Verify bilinear interpolation module is correct with respect to equation 3
11. Understand and verify the gradient formulas in equations 4-6 match with torch gradient graph
12. Correct for ksize and offset unit parameters in 4.1.2