
Patch-based Deep Neural Network
=====

Summary
-----

For image denoising with non-local methods, using the choice of which non-local patches to use dramatically impacts denoising quality. For the Video Non-Local Bayes (VNLB) image denoisig method, if the patches are carefully chosen (explained below) the method's resulting PSNR is state-of-the-art. This library implements a neural network method to efficiently denoise the patches returned from a non-local search to improve the patch matching quality. 

