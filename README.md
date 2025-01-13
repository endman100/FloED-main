<h1 align="center" style="font-weight: bold; font-size: 2.5rem;">
  FloED: <span style="font-size: 1.5rem; font-weight: normal; color: #555;">Advanced Video Inpainting Using Optical Flow-Guided Efficient Diffusion</span>
</h1>

<div align="center">

Bohai Gu &emsp; Hao Luo &emsp; Song Guo &emsp; Peiran Dong

HKUST, DAMO Academy, Alibaba Group

[![arXiv](https://img.shields.io/badge/arXiv-2412.00857-b31b1b)](https://arxiv.org/abs/2412.00857)
[![Project Page](https://img.shields.io/badge/Project-Website-orange)](https://nevsnev.github.io/FloED)
</div>



 <div class="content has-text-justified">
          <p>
            Recently, diffusion-based methods have achieved great improvements in the video inpainting task. However, these methods still face many challenges, such as maintaining temporal consistency and the time-consuming issue.
            This paper proposes an advanced video inpainting framework using optical Flow-guided Efficient Diffusion, called FloED. 
            Specifically, FloED employs a dual-branch architecture, where a flow branch first restores corrupted flow and a multi-scale flow adapter provides motion guidance to the main inpainting branch.
            Additionally, a training-free latent interpolation method is proposed to accelerate the multi-step denoising process using flow warping. Further introducing a flow attention cache mechanism, FLoED efficiently reduces the computational cost brought by incorporating optical flow.
          </p>
        </div>

![Method Overview](Assert/Fig_0.jpg)




For more visualization results, please check our [project page](https://nevsnev.github.io/FloED).

## Code coming soon.


## 📚 BibTeX

```bibtex
@article{gu2024advanced,
  title={Advanced Video Inpainting Using Optical Flow-Guided Efficient Diffusion},
  author={Gu, Bohai and Luo, Hao and Guo, Song and Dong, Peiran},
  journal={arXiv preprint arXiv:2412.00857},
  year={2024}
}
