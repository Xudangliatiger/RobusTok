## Image Tokenizer Needs Post-Training

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2509.12474-b31b1b.svg)](https://arxiv.org/abs/2509.12474)
[![🤗 Weights](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-yellow)](https://huggingface.co/your-org/RobusTok) 

</div>

<div align="center">
  <img src="assets/teaser.png" alt="Teaser" width="95%">
</div>

---

## TL;DR

We present RobusTok, a new image tokenizer with a two-stage training scheme:
Main training → constructs a robust latent space
Post-training → aligns the generator’s latent distribution with its image space

## Key highlights of Post-Training

- 🚀 **Better generative quality**: gFID 1.60 → 1.36.
- 🔑 **Generalizability**: applicable to both autoregressive & diffusion models.
- ⚡ **Efficiency**: strong results with only ~400M generative models.


---

## Updates
- (2025.09.16) Paper released in Arxiv. Working on code cleaning, code and checkpoints will be released in these two weeks.

---

## Visualization

<div align="center">
  <img src="assets/ft-diff.png" alt="vis" width="95%">
  <p>
    visualization of 256&times;256 image generation before (top) and after (bottom) post-training. Three improvements are observed: (a) OOD mitigation, (b) Color fidelity, (c) detail refinement.
  </p>
</div>

---

## Citation

If our work assists your research, feel free to give us a star ⭐ or cite us using

```
@misc{qiu2025imagetokenizerneedsposttraining,
      title={Image Tokenizer Needs Post-Training}, 
      author={Kai Qiu and Xiang Li and Hao Chen and Jason Kuen and Xiaohao Xu and Jiuxiang Gu and Yinyi Luo and Bhiksha Raj and Zhe Lin and Marios Savvides},
      year={2025},
      eprint={2509.12474},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2509.12474}, 
}
```