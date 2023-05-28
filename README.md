#MixDehazeNet : Mix Structure Block For Image Dehazing Network

>**Abstract:**
Image dehazing is a typical task in the low-level vision field. Previous studies verified the effectiveness of the large convolutional kernel and attention mechanism in dehazing.However, there are two drawbacks: the multi-scale properties of an image are readily ignored when a large convolutional kernel is introduced, and the standard series connection of an attention module does not sufficiently consider an uneven hazy distribution. In this paper, we propose a novel framework named Mix Structure Image Dehazing Network (MixDehazeNet), which solves two issues mentioned above. Specifically, it mainly consists of two parts: the multi-scale parallel large convolution kernel module and the enhanced parallel attention module. Compared with a single large kernel, parallel large kernels with multi-scale are more capable of taking partial texture into account during the dehazing phase. In addition, an enhanced parallel attention module is developed, in which parallel connections of attention perform better at dehazing uneven hazy distribution. Extensive experiments on three benchmarks demonstrate the effectiveness of our proposed methods. For example, compared with the previous state-of-the-art methods, MixDehazeNet achieves a significant improvement (42.62dB PSNR) on the SOTS indoor dataset.

![image](https://github.com/AmeryXiong/MixDehazeNet/assets/102467128/6e0e3c9a-b137-4e14-b8fa-e3881f7fc20d)
![image](https://github.com/AmeryXiong/MixDehazeNet/assets/102467128/221707a7-db3b-4401-a8ed-8917f90b8561)
![image](https://github.com/AmeryXiong/MixDehazeNet/assets/102467128/c3e6073e-a78b-49bd-a6bd-875274925002)
![image](https://github.com/AmeryXiong/MixDehazeNet/assets/102467128/2b9b1124-fe63-436e-b67e-24960e99e14f)

