# Data Folder of RealFill
We include the following two datasets used in our evaluation and user study:

1. `RealBench/`: Our proposed dataset for quantatitive evaluation. It consists of 33 scenes (23 outpainting and 10 inpainting), where each scene has a set of reference images $\mathcal{X}_{ref}$, a target image $I_{tgt}$ to fill, a binary mask $M_{tgt}$ indicating the missing region, and the ground-truth result $I_{gt}$. The number of reference images in each scene varies from 1 to 5. The dataset contains diverse, challenging scenarios with significant variations between the reference and target images, such as changes in viewpoint, aperture, lighting, style, and subject pose.
2. `Qualitative/`: 25 additional challenging scenes for qualitative evaluation and user study. They are mainly collected from internet photos, so there're no ground-truths.

Both data folders share the same organization structure. Each subfolder contains one scene,  entitled with an index like `0`. It has the following two folders:

1. `ref/`: a set of reference images.
2. `target/`: 
    - `target.png`: target image to fill. 
    - `mask.png`: binary mask indicating the region to fill. 
    - `gt.png`: ground-truth image, if any.