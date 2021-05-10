# PyTorch implementation of EWGS

This is the implementation of the paper "Network Quantization with Element-wise Gradient Scaling".

For more information, checkout the project site [[website](https://cvlab.yonsei.ac.kr/projects/EWGS/)] and the paper [[PDF](https://arxiv.org/pdf/2104.00903.pdf)].

## Requirements
* Python >= 3.6
* PyTorch >= 1.3.0

## Datasets
* CIFAR-10 (will be automatically downloaded when you run the code)
* ImageNet (ILSVRC-2012) available at [http://www.image-net.org](http://www.image-net.org/download)

## Code
Please refer to the ``run.sh`` files in the CIFAR10 and ImageNet folders.

## Bibtex
```
@inproceedings{lee2021network,
  title={Network Quantization with Element-wise Gradient Scaling},
  author={Lee, Junghyup and Kim, Dohyung and Ham, Bumsub},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2021}
}
```

## References
* ImageNet training code: [[PyTorch official example code](https://github.com/pytorch/examples/blob/master/imagenet/main.py)]
* ResNet-18/34 models: [[PyTorch official code](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py)]
* ResNet-20 model: [[ResNet on CIFAR10](https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py)] [[IRNet](https://github.com/XHPlus/IR-Net/blob/master/resnet-20-cifar10/1w1a/resnet.py)]
* Quantized modules: [[DSQ](https://github.com/ricky40403/DSQ/blob/master/DSQConv.py#L18)]
* Estimating Hessian trace: [[PyHessian](https://github.com/amirgholami/PyHessian/blob/master/pyhessian/hessian.py#L160)]