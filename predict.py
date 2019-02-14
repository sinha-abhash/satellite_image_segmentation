import torch
from PIL import Image
import numpy as np
import argparse
import os


def decode_segmap(encoded_im):
    inference = np.ones_like(encoded_im, dtype=np.uint8) * 255

    inference[encoded_im == 0] = 0
    encoded_im[encoded_im == 1] = 255
    encoded_im = encoded_im.astype(np.uint8)

    return encoded_im

def main(args):
    # read images
    image_files = [os.path.join(args.im_dir, f) for f in os.listdir(args.im_dir)]

    # load model
    model = torch.load(args.model_path)

    for im_name in image_files:
        im = np.array(Image.open(im_name))

        # set background to 255 where the alpha value is 0.
        im[im[:, :, 3] == 0] = 255

        # consider only three channels of the image, ignore alpha channel.
        im = im[:, :, :3]

        # channel first for satisfying pytorch requirements of image read.
        im = np.rollaxis(im, 2, 0)

        im = np.expand_dims(im, axis=0)
        im = torch.from_numpy(im)
        im = im.float()

        # predict
        output = model(im)
        pred = np.squeeze(output.data.max(1)[1].cpu().numpy(), axis=0)
        decoded_im = decode_segmap(pred)

        # save image
        output_name = im_name.split('/')[-1]
        output_name = os.path.join(args.output_path, output_name)
        decoded_im = Image.fromarray(decoded_im)
        decoded_im.save(output_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Provide inputs for FCN models prediction")
    parser.add_argument('--im_dir', type=str, default='/home/abhash/Documents/pix4d/MLExpert/images/images',
                        help='provide path of image files')
    parser.add_argument('--model_path', type=str, default='/home/abhash/Documents/pix4d/MLExpert/saved_model/fcn.pt',
                        help='provide path of image files')
    parser.add_argument('--output_path', type=str, default='/home/abhash/Documents/pix4d/MLExpert/inference/',
                        help='provide path for saving output image')
    args = parser.parse_args()
    main(args)