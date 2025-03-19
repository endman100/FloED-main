<h1 align="center" style="font-weight: bold; font-size: 2.5rem;">
  FloED: <span style="font-size: 1.5rem; font-weight: normal; color: #555;">Coherent Video Inpainting Using Optical Flow-Guided Efficient Diffusion</span>
</h1>


<div align="center">

Bohai Gu &emsp; Hao Luo &emsp; Song Guo &emsp; Peiran Dong  &emsp; Qihua Zhou  

HKUST, DAMO Academy, Alibaba Group

[![arXiv](https://img.shields.io/badge/arXiv-2412.00857-b31b1b)](https://arxiv.org/abs/2412.00857)
[![Project Page](https://img.shields.io/badge/Project-Website-orange)](https://nevsnev.github.io/FloED)
</div>



 <div class="content has-text-justified">
          <p>
      The text-guided video inpainting technique has significantly improved the performance of content generation applications. A recent family for these improvements uses diffusion models, which have become essential for achieving high-quality video inpainting results, yet they still face performance bottlenecks in temporal consistency and computational efficiency. This motivates us to propose a new video inpainting framework using optical Flow-guided Efficient Diffusion (FloED) for higher video coherence. Specifically, FloED employs a dual-branch architecture, where the time-agnostic flow branch restores corrupted flow first, and the multi-scale flow adapters provide motion guidance to the main inpainting branch. Besides, a training-free latent interpolation method is proposed to accelerate the multi-step denoising process using flow warping. With the flow attention cache mechanism, FLoED efficiently reduces the computational cost of incorporating optical flow. Extensive experiments on background restoration and object removal tasks show that FloED outperforms state-of-the-art diffusion-based methods in both quality and efficiency. 
          </p>
        </div>


![Method Overview](Assert/Fig_0.jpg)

**Video Demo**

![video demo](Assert/video.mp4)


Please refer our [project page](https://nevsnev.github.io/FloED) for more details.



## To-Do List

- [ ] Release the inference code and weights  

- [ ] Release the training code  

  


## BibTeX

```bibtex
@article{gu2024advanced,
  title={Advanced Video Inpainting Using Optical Flow-Guided Efficient Diffusion},
  author={Gu, Bohai and Luo, Hao and Guo, Song and Dong, Peiran},
  journal={arXiv preprint arXiv:2412.00857},
  year={2024}
}
