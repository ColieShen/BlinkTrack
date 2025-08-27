<div align="center">
<h1>BlinkTrack & MultiTrack (ICCV 2025)</h1>
<h3>BlinkTrack: Feature Tracking over 80 FPS via Events and Images</h3>

<!-- [Yichen Shen](https://scholar.google.com/citations?view_op=list_works&hl=en&user=LFa-H2cAAAAJ)<sup>1</sup>, [xxx](http)<sup>1</sup>,

<sup>1</sup> Zhejiang University, <sup>2</sup> XXX -->

[![Paper](https://img.shields.io/badge/arXiv-<2409.17981>-A42C25.svg)](https://arxiv.org/abs/2409.17981)

[![Weight](https://img.shields.io/badge/ModelScope-<Weight>-604DF4.svg)](https://modelscope.cn/models/ColieShen/BlinkTrack)
[![Weight](https://img.shields.io/badge/HuggingFace-<Weight>-F8D44E.svg)](https://huggingface.co/ColieShen/BlinkTrack)

[![MultiTrack](https://img.shields.io/badge/ModelScope-<MultiTrack>-604DF4.svg)](https://modelscope.cn/datasets/ColieShen/MultiTrack)
[![MultiTrack](https://img.shields.io/badge/HuggingFace-<MultiTrack>-F8D44E.svg)](https://huggingface.co/datasets/ColieShen/MultiTrack)

[![EC-occ_EDS-occ](https://img.shields.io/badge/ModelScope-<EC--occ_EDS--occ>-604DF4.svg)](https://modelscope.cn/datasets/ColieShen/EC-occ_EDS-occ)
[![EC-occ_EDS-occ](https://img.shields.io/badge/HuggingFace-<EC--occ_EDS--occ>-F8D44E.svg)](https://huggingface.co/datasets/ColieShen/EC-occ_EDS-occ)


</div>



## Abstract



Event cameras, known for their high temporal resolution and ability to capture asynchronous changes, have gained significant attention for their potential in feature tracking, especially in challenging conditions. However, event cameras lack the fine-grained texture information that conventional cameras provide, leading to error accumulation in tracking. To address this, we propose a novel framework, BlinkTrack, which integrates event data with grayscale images for high-frequency feature tracking. Our method extends the traditional Kalman filter into a learning-based framework, utilizing differentiable Kalman filters in both event and image branches. This approach improves single-modality tracking and effectively solves the data association and fusion from asynchronous event and image data. We also introduce new synthetic and augmented datasets to better evaluate our model. Experimental results indicate that BlinkTrack significantly outperforms existing methods, exceeding 80 FPS with multi-modality data and 100 FPS with preprocessed event data.


## News

- **_News (2025-08-27)_**: We release the [pretrained weights](https://modelscope.cn/models/ColieShen/BlinkTrack)!
- **_News (2025-08-26)_**: We release the testing dataset, including [EC-syn, EC-occ, EDS-syn, EDS-occ](https://modelscope.cn/datasets/ColieShen/EC-occ_EDS-occ)!
- **_News (2025-08-24)_**: We release the [camera-ready version of the paper and supplementary materials](https://arxiv.org/abs/2409.17981)! We also release the training dataset, [MultiTrack](https://modelscope.cn/datasets/ColieShen/MultiTrack)!
- **_News (2025-07-19)_**: 🎉🎉🎉 We release the init version of BlinkTrack! Further updates will be available soon!

## Todo


- [x] Release the camera-ready version of the paper and supplementary materials
- [x] Release the pretrained weights
- [ ] Add complete usage guide for BlinkTrack
- [x] Release the MultiTrack dataset
- [x] Release the data generation code of MultiTrack
- [ ] Add the full documentation of MultiTrack
- [x] Release the EC-syn, EDS-syn, EC-occ and EDS-occ dataset
- [ ] Release the data generation code of EC-occ and EDS-occ
- [ ] Add the full documentation of EC-occ and EDS-occ


## Citation

If this work is helpful for your research, please consider citing the following BibTeX entry.

```
@misc{shen2024blinktrackfeaturetracking100,
      title={BlinkTrack: Feature Tracking over 100 FPS via Events and Images}, 
      author={Yichen Shen and Yijin Li and Shuo Chen and Guanglin Li and Zhaoyang Huang and Hujun Bao and Zhaopeng Cui and Guofeng Zhang},
      year={2024},
      eprint={2409.17981},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2409.17981}, 
}
```

## Content

[BlinkTrack](#blinktrack)

1. [Introduction](#introduction)

[MultiTrack](#multitrack)

1. [Introduction](#introduction-1)

# BlinkTrack

## Introduction
![](./figure/blinktrack_pipeline.png)
BlinkTrack is a Kalman-filter-based framework for feature tracking that effectively solves the data association and fusion
from asynchronous event data and image data.

# MultiTrack
## Introduction
![](./figure/multitrack_pipeline.png)
MultiTrack is a dataset with color images, events, occluded tracks, and visibility status.
![](./figure/multitrack_example.png)

---

## Acknowledgments

Our work stands on the shoulders of giants. We want to thank the following contributors for our code is based on:

- Deep-Ev-Tracker
https://github.com/uzh-rpg/deep_ev_tracker
- RAFT
https://github.com/princeton-vl/RAFT






