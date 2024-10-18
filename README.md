# <p align=center> [ECCV 2024] MonoWAD: Weather-Adaptive Diffusion Model for Robust Monocular 3D Object Detection</p>
Official Repository for "MonoWAD: Weather-Adaptive Diffusion Model for Robust Monocular 3D Object Detection".

<div align="center">

[![paper](https://img.shields.io/badge/MonoWAD-paper-green.svg)](https://arxiv.org/abs/2407.16448)

<img alt="img-name" src="https://github.com/VisualAIKHU/MonoWAD/assets/132932095/16871ca9-b57c-454f-895c-8d44bd835de1" width="900"> 

</div>

## Installation
**Create MonoWAD environment**
```
git clone https://github.com/VisualAIKHU/MonoWAD.git
cd MonoWAD

conda create -n monowad python=3.10
conda activate monowad
```

**Install pytorch**
```
# We adopt torch 2.0.1
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
```

## Data Preparation

Please download the official [KITTI dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d).
You can also download our Foggy KITTI dataset with different fog densities.

**Foggy KITTI dataset:**
* [Foggy 0.1](https://drive.google.com/file/d/1iOpoZ-QbJdU2ytRmd9wPxH0RNjZ6KNdQ/view?usp=sharing) (The main paper uses 0.1)
* [Foggy 0.05](https://drive.google.com/file/d/1BfWvrMqYkSA_8edxX3IfiM0Nog1w_p7w/view?usp=sharing)
* [Foggy 0.15](https://drive.google.com/file/d/1J37b12IpckWu38K8NSY-1yc8KD8h5F-R/view?usp=sharing)
* [Foggy 0.30](https://drive.google.com/file/d/1_fVHEssaCX58wE4fHh3fzexzBhrD4Zux/view?usp=sharing)
* [Foggy test](https://drive.google.com/file/d/1H5jQrueWlqfQy52ihsgxTySxxljM_4br/view?usp=sharing)

**Directory structure:**
```
#MonoWAD_ROOT
  |data/
    |KITTI/
      |object/			
        |training/
          |calib/
          |foggy_2/ #adverse weather images
          |origin_2/ #clear images
          |label_2/
          |velodyne/
        |testing/
          |calib/
          |image_2/ 
```

## Citation
If you use MonoWAD, please consider citing:

    @article{oh2024monowad,
      title={MonoWAD: Weather-Adaptive Diffusion Model for Robust Monocular 3D Object Detection},
      author={Oh, Youngmin and Kim, Hyung-Il and Kim, Seong Tae and Kim, Jung Uk},
      journal={arXiv preprint arXiv:2407.16448},
      year={2024}
    }
---

## Acknowlegment

Our codes benefits from the excellent [visualDet3D](https://github.com/Owen-Liuyuxuan/visualDet3D), [MonoDTR](https://github.com/KuanchihHuang/MonoDTR), [denoising-diffusion-pytorch
](https://github.com/lucidrains/denoising-diffusion-pytorch).