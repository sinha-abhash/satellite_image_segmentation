from PIL import Image
import os
import numpy as np

from torch.utils.data import Dataset


class SegmentationData(Dataset):
    def __init__(self, image_path, segment_path, n_classes):
        self.root_path = image_path
        self.segment_path = segment_path
        self.image_names = [os.path.join(image_path, f) for f in os.listdir(image_path)]
        self.segment_names = [os.path.join(segment_path, f) for f in os.listdir(segment_path)]
        assert len(self.image_names) == len(self.segment_names), "number of images in image folder is not equal to " \
                                                                 "number of segment images: %d:%d" %(len(self.image_names),
                                                                                                         len(self.segment_names))
        self.palette = {0:0, 255:1}
        self.n_classes = n_classes

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        im_name = self.image_names[idx]
        im = Image.open(im_name)
        im = np.array(im)

        # set background to 255 where the alpha value is 0.
        im[im[:, :, 3] == 0] = 255

        # consider only three channels of the image, ignore alpha channel.
        im = im[:, :, :3]

        # channel first for satisfying pytorch requirements of image read.
        im = np.rollaxis(im, 2, 0)

        segment_name = self.segment_names[idx]
        segment_im = Image.open(segment_name)
        segment_im = segment_im.convert('L')
        segment_im = self.encode(segment_im)  # convert pixels with value 0 to 1 and 255 to 0
        segment_im = self.one_hot_encoding(segment_im)

        return im, segment_im

    def encode(self, seg_im):
        seg_im = np.array(seg_im)
        seg_im[seg_im == 0] = 1
        seg_im[seg_im == 255] = 0

        return seg_im

    def one_hot_encoding(self, label):
        layer1 = np.zeros_like(label, dtype=np.uint8)
        layer2 = np.zeros_like(label, dtype=np.uint8)

        # pixel with value 0 will have layer1 with value 1 and layer 2 will have value 0
        # pixel with value 1 will have layer2 with value 1 and layer 1 will have value 0
        layer1[label == np.array(1)] = 1
        layer2[label == np.array(0)] = 1

        # stack both the layers
        result = np.stack((layer1, layer2), axis=0)    # size = 2 x W x H

        return result