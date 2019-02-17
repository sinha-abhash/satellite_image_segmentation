import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from PIL import Image
import numpy as np
import argparse
import os
import shutil

from utils import decode_segmap, crop_and_save, stitch_predicted
from data import SegmentationData


def main(args):
    # crop input image and save on disk
    cropped_input_images_path = os.path.join(args.save_cropped, 'input_images_test')
    crop_and_save(args, cropped_input_images_path)

    seg_dataset = SegmentationData(cropped_input_images_path, phase=args.phase)
    test_loader = DataLoader(seg_dataset, shuffle=False)

    # load model
    model = torch.load(args.model_path)

    # create temp folder for saving prediction for each cropped input images
    temp_name = 'temp'
    if not os.path.exists(os.path.join(args.output_path, temp_name)):
        os.makedirs(os.path.join(args.output_path, temp_name))

    for i, (image, im_name) in enumerate(test_loader):
        image = image.float()

        image = Variable(image.cuda())

        # predict
        output = model(image)
        pred = np.squeeze(output.data.max(1)[1].cpu().numpy(), axis=0)
        decoded_im = decode_segmap(pred)

        # save image
        output_name = im_name[0].split('/')[-1]

        output_name = os.path.join(args.output_path, temp_name, output_name)
        decoded_im = Image.fromarray(decoded_im)
        decoded_im.save(output_name)

    stitch_predicted(args)
    shutil.rmtree(os.path.join(args.output_path, 'temp'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Provide inputs for FCN models prediction")
    parser.add_argument('--im_dir', type=str, default='./images/images',
                        help='provide path of image files')
    parser.add_argument('--model_path', type=str, default='./saved_model/fcn.pt',
                        help='provide path of image files')
    parser.add_argument('--output_path', type=str, default='./inference/',
                        help='provide path for saving output image')
    parser.add_argument('--phase', type=str, default='test', choices=['train', 'test'], help='train or test')
    parser.add_argument('--save_cropped', type=str, default='./images/cropped',
                        help='provide path for saving cropped images')
    args = parser.parse_args()
    main(args)