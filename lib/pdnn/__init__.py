"""

PDNN: Patch-based Denoising Neural Network

"""

from .learn import exec_learn


"""

Coding Wants:

1. apply PDNN to denoise a subset of patches, with shape (B,N,T,C,H,W):
   e.g. denoised = some_function(model,patches)
2. convert a image bursts to training patches
   e.g. inputs,target = some_function(images)
3. we want "hooks" into the patch subsetting problem, asap
   e.g. each iteration we can see how well our subsetting is getting.
4. we want to stratify results across the patch content,
   e.g. "stripy" v.s. "curvy" v.s. "text" types of patches

Choice Justification:

- why use a patch-based image denoiser?
-> standard CNNs are build for higher resolution images,
   and some architectures (e.g. resnet) don't work for smaller patches
-> leveraging non-local information requires creating an entire image
   to denoise the patches, but in our case we only want to patches denoised.

- we have to pick a number of patches to jointly denoise:
-> do we denoise all of them or just one?
   -> all of them since we can average output across "N"
-> how do we motivate a specific number?
   -> too few = not leveraging non-local information,
   -> too many = including too many non-similar patches
   *-> we use a histogram of differences returned using the l2 difference
      and we pick the specific number at an "elbow" in the flow
   -> since the goal is to specify a subset of patches to be used in VNLB
      it seems this choice should somehow related to the "beam search" code...
      -> this is not needed if we have (*) above.

- we have to pick a resolution:
-> 32x32 v.s. 7x7 v.x. 15x15?
-> what is the denoising quality result w.r.t a cropped version of each type?
-> how does this resolution relate to the bias of non-local methods?
-> how does this resolution impact the subseting of the non-local method

- we want to vary the "denoising resolution" and the "sorting resolution"
-> denoise a 15x15 and only use the 7x7 for sorting
-> do we only predict the interior? no. this is not standard so we don't need to.


"""
