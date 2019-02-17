# satellite_image_segmentation using PyTorch
Fully Convolutional Network for Semantic Segmentation[1] implemented in PyTorch. The network architecture is not the same as used in the original work, but the basic idea is same: encoder using convolution, decoder using transposed convolution and skip connection. The purpose of the repository is to implement FCN with basic structure on satellite image. Few codes/ideas have been borrowed from [2], [3] and [4]

## Requirements
- matplotlib==3.0.2
- numpy==1.16.1
- Pillow==5.4.1
- prompt-toolkit==2.0.8
- torch==1.0.1.post2
- torchvision==0.2.1
- tqdm==4.31.1

## How to use
### To train:
```shell
python train.py --im_dir <input image folder> --seg_dir <gt folder> --save_cropped <parent folder path where cropped images will be saved> --crop_images True --  n_classes 2 --n_epoch 10 --model_path <path to save models> --output_path <path to save images during training time> --plot_path <path to save loss plot>
```
To crop the images and ground truth, set the flag --crop_images to True. The cropping works by taking the original image as input and sliding a cropping patch of 256x256 across the image.

### To test:

For testing, cropping of images is not required. 
```shell
python predict.py --im_dir <input image folder> --model_path <saved model path> --output_path <path to save output result>
```

## Reference
[1] [Long J, Shelhamer E, Darrell T. Fully convolutional networks for semantic segmentation[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2015: 3431-3440.](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)

[2] [https://github.com/pochih/FCN-pytorch](https://github.com/pochih/FCN-pytorch)

[3] [https://github.com/meetshah1995/pytorch-semseg](https://github.com/meetshah1995/pytorch-semseg)

[4] [https://github.com/daisenryaku/pytorch-fcn](https://github.com/daisenryaku/pytorch-fcn)
