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
        segment_name = self.segment_names[idx]

        im = Image.open(im_name)
        r,g,b,a = im.split()
        im = np.stack((r,g,b), axis=0)

        segment_im = Image.open(segment_name)
        segment_im = segment_im.convert('L')
        segment_im = self.encode(segment_im)  # convert pixels with value 0 to 0 and 255 to 1
        segment_im = self.one_hot_encoding(segment_im)
        '''
        im_np = im.astype(np.uint8)
        im_pil = Image.fromarray(im_np)
        im_pil.save('/home/abhash/Documents/pix4d/MLExpert/output/test1.png')
        '''
        return im, segment_im

    def encode(self, seg_im):
        arr = np.zeros_like(seg_im, dtype=np.uint8)
        for c, i in self.palette.items():
            m = seg_im == np.array(c)
            arr[m] = i

        return arr

    def decode_segmap(self, temp, plot=False):
        label_colours = self.palette
        reverse_label_colours = {v:k for k,v in label_colours.items()}
        inference = np.zeros_like(temp, dtype=np.uint8)

        for l in range(0, self.n_classes):
            m = temp == np.array(l)
            inference[m] = reverse_label_colours[l]

        return inference


    def one_hot_encoding(self, label):
        layer1 = np.zeros_like(label, dtype=np.uint8)
        layer2 = np.zeros_like(label, dtype=np.uint8)

        # pixel with value 0 will have layer1 with value 1 and layer 2 will have value 0
        # pixel with value 1 will have layer2 with value 1 and layer 1 will have value 0
        layer1[label == np.array(0)] = 1
        layer2[label == np.array(1)] = 1

        # stack both the layers
        result = np.stack((layer1, layer2), axis=0)    # size = 2 x W x H

        return result
