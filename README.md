[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://doi.org/10.48550/arXiv.2305.17654)
[![Weights](https://img.shields.io/badge/GoogleDrive-Weights-yellow)](https://drive.google.com/drive/folders/1ep6W4H3vNxshYjq71Tb3MzxrXGgaiM6C?usp=drive_link)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mixdehazenet-mix-structure-block-for-image/image-dehazing-on-sots-indoor)](https://paperswithcode.com/sota/image-dehazing-on-sots-indoor?p=mixdehazenet-mix-structure-block-for-image)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mixdehazenet-mix-structure-block-for-image/image-dehazing-on-sots-outdoor)](https://paperswithcode.com/sota/image-dehazing-on-sots-outdoor?p=mixdehazenet-mix-structure-block-for-image)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mixdehazenet-mix-structure-block-for-image/image-dehazing-on-reside-6k)](https://paperswithcode.com/sota/image-dehazing-on-reside-6k?p=mixdehazenet-mix-structure-block-for-image)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mixdehazenet-mix-structure-block-for-image/image-dehazing-on-haze4k)](https://paperswithcode.com/sota/image-dehazing-on-haze4k?p=mixdehazenet-mix-structure-block-for-image)

# MixDehazeNet : Mix Structure Block For Image Dehazing Network

>**Abstract:**
Image dehazing is a typical task in the low-level vision field. Previous studies verified the effectiveness of the large convolutional kernel and attention mechanism in dehazing. However, there are two drawbacks: the multi-scale properties of an image are readily ignored when a large convolutional kernel is introduced, and the standard series connection of an attention module does not sufficiently consider an uneven hazy distribution. In this paper, we propose a novel framework named Mix Structure Image Dehazing Network (MixDehazeNet), which solves two issues mentioned above. Specifically, it mainly consists of two parts: the multi-scale parallel large convolution kernel module and the enhanced parallel attention module. Compared with a single large kernel, parallel large kernels with multi-scale are more capable of taking partial texture into account during the dehazing phase. In addition, an enhanced parallel attention module is developed, in which parallel connections of attention perform better at dehazing uneven hazy distribution. Extensive experiments on three benchmarks demonstrate the effectiveness of our proposed methods. For example, compared with the previous state-of-the-art methods, MixDehazeNet achieves a significant improvement (42.62dB PSNR) on the SOTS indoor dataset.

## Framework:
![image](https://github.com/AmeryXiong/MixDehazeNet/assets/102467128/885f69da-ab72-4c9c-8223-1b7425e98d3a)

## Old Expriment Result:
![image](https://github.com/AmeryXiong/MixDehazeNet/assets/102467128/5d087804-0b5c-4232-8f73-8296de5b8374)

## Lastest Expriment Result (In new Vision but in process):
![image](https://github.com/AmeryXiong/MixDehazeNet/assets/102467128/e5df99e5-37f2-4a83-83bf-ca270a5d7e14)
![image](https://github.com/AmeryXiong/MixDehazeNet/assets/102467128/1e59ce32-75f8-4d3f-8f63-8766524af540)

## About dataset:
Since my code refers to [Dehazeformer](https://github.com/IDKiro/DehazeFormer#vision-transformers-for-single-image-dehazing), the dataset format is the same as that in Dehazeformer. In order to avoid errors when training the datasets, please download the datasets from [Dehazeformer](https://github.com/IDKiro/DehazeFormer#vision-transformers-for-single-image-dehazing) for training.

## This is pretrain weights (google driver):
https://drive.google.com/drive/folders/1ep6W4H3vNxshYjq71Tb3MzxrXGgaiM6C?usp=drive_link

## Help:
If you have any questions, you can send email to xiongqian2021@whut.edu.cn or xiongqian2019@outlook.com.

## Thanks
Special Thanks to my supervisor and @[IDKiro](https://github.com/IDKiro), they gave me selfless help in completing this work and answered my questions. Thank you very much.

## Citation
If you find this work useful for your research, please cite our paper:
```bibtex
@article{Mixdehazenet,
  title={MixDehazeNet: Mix Structure Block For Image Dehazing Network},
  author={Lu, LiPing and Xiong, Qian and Chu, DuanFeng and Xu, BingRong},
  journal={arXiv preprint arXiv:2305.17654},
  year={2023}
}
```

