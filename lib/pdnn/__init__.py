"""

PDNN: Patch-based Denoising Neural Network

"""

from .learn import exec_learn,load_model,load_sigma_model
from .denoise import denoise_patches
from .utils import config_from_uuid
# from .train_patches import
from .search import exec_patch_search

"""

Coding Wants:

1. convert a image bursts to training patches
   e.g. inputs,target = some_function(images)
   -> complete draft.
   -> we kind of don't use the "sepconv" n x f x ... concept;
      -> we do not add the patch content along "f"
         -> however, we DO use "pt" along "f".. maybe justifiable
      -> alt idea: maybe we should use an "image" of size X x Y and
         then add the patch features as PaCNet does?

2. apply PDNN to denoise a subset of patches, with shape (B,N,T,C,H,W):
   e.g. denoised = some_function(model,patches)

3. we want "hooks" into the patch subsetting problem, asap
   e.g. each iteration we can see how well our subsetting is getting.

4. we want to stratify results across the patch content,
   e.g. "stripy" v.s. "curvy" v.s. "text" types of patches
   -> kind-of check this off; we can do edges

5. inspect results

Choice Justification:

- why use a patch-based image denoiser?
-> standard CNNs are build for higher resolution images,
   and some architectures (e.g. resnet) don't work for smaller patches
-> leveraging non-local information requires creating an entire image
   to denoise the patches, but in our case we only want to patches denoised.
   -> advocating for the "right tool for the right job" perspective

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
      -> or rather (**) below
   -> PaCNet comes shipped using 15 patches... we can just reference their
      results to justify our choice

- we have to pick a resolution:
-> 32x32 v.s. 7x7 v.x. 15x15?
   -> again, since we use SepConv from PaCNet we can merely state 13x13
      because our reference uses a 13x13 patch.
-> what is the denoising quality result w.r.t a cropped version of each type?
-> how does this resolution relate to the bias of non-local methods?
-> how does this resolution impact the subseting of the non-local method

- we want to vary the "denoising resolution" and the "sorting resolution"
-> denoise a 15x15 and only use the 7x7 for sorting
-> do we only predict the interior? no. this is not standard so we don't need to.

"""
