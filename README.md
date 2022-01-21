
Patch-based Deep Neural Network
=====

Summary
-----

For image denoising with non-local methods, using the choice of which non-local patches to use dramatically impacts denoising quality. For the Video Non-Local Bayes (VNLB) image denoisig method, if the patches are carefully chosen (explained below) the method's resulting PSNR can approximately match and is sometimes better than state-of-the-art deep learning based methods. This library implements a methods to efficiently denoise a the patches returned from a non-local search to improve the patch matching quality. 

