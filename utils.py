from PIL import Image
import numpy as np
import os
from tqdm import tqdm


def crop_x(y1, y2, image):
    x2 = 0
    cropped_im_list = []
    cropped_x_y = []
    count_x = 0
    while x2 + 256 < image.size[0]:
        x1 = count_x * 256
        x2 = x1 + 256
        im = image.crop(box=(x1, y1, x2, y2))
        cropped_im_list.append(im)
        cropped_x_y.append((x1, y1))
        count_x += 1

    # residual portion on right
    if x2 + 256 > image.size[0]:
        x1 = image.size[0] - 256
        x2 = image.size[0]
        im = image.crop(box=(x1, y1, x2, y2))
        cropped_im_list.append(im)
        cropped_x_y.append((x1, y1))

    return cropped_im_list, cropped_x_y


def multiple_crop(image_path):
    image = Image.open(image_path)
    cropped_im_list, cropped_x_y_list = [], []
    x1, y1, x2, y2 = 0, 0, 0, 0
    count_y = 0
    while y2 + 256 < image.size[1]:
        y1 = count_y * 256
        y2 = y1 + 256

        cropped_x_list, cropped_x_y = crop_x(y1, y2, image)
        cropped_im_list.extend(cropped_x_list)
        cropped_x_y_list.extend(cropped_x_y)
        count_y += 1

    # residual portion at bottom
    if y2 + 256 > image.size[1]:
        y1 = image.size[1] - 256
        y2 = image.size[1]

        cropped_x_list, cropped_x_y = crop_x(y1, y2, image)
        cropped_im_list.extend(cropped_x_list)
        cropped_x_y_list.extend(cropped_x_y)

    return cropped_im_list, cropped_x_y_list


def crop_and_save(args, cropped_input_images_path, cropped_gt_images_path = None):
    image_names = [f for f in os.listdir(args.im_dir)]
    if args.phase == 'train':
        segment_names = [f for f in os.listdir(args.seg_dir)]
        assert len(image_names) == len(segment_names), "number of images in image folder is not equal to " \
                                                             "number of segment images: %d:%d" % (len(image_names),
                                                                                                  len(segment_names))

    if not os.path.exists(cropped_input_images_path):
        os.makedirs(cropped_input_images_path)

    if args.phase == 'train':
        if not os.path.exists(cropped_gt_images_path):
            os.makedirs(cropped_gt_images_path)

    for im_name in tqdm(image_names):
        cropped_input_images, cropped_x_y_list = multiple_crop(os.path.join(args.im_dir, im_name))
        if args.phase == 'train':
            cropped_gt_images, _ = multiple_crop(os.path.join(args.seg_dir, im_name))

            for idx, (input_im, gt_im) in tqdm(enumerate(zip(cropped_input_images, cropped_gt_images))):
                fname = im_name.split('.')[0] + '_' + str(idx) + '.png'
                input_image_fname = os.path.join(cropped_input_images_path, fname)
                gt_image_fname = os.path.join(cropped_gt_images_path, fname)
                input_im.save(input_image_fname)
                gt_im.save(gt_image_fname)
        elif args.phase == 'test':
            #cropped_test_input_images = multiple_crop(os.path.join(args.im_dir, im_name))
            for idx, (input_im, cropped_x_y) in tqdm(enumerate(zip(cropped_input_images, cropped_x_y_list))):
                fname = im_name.split('.')[0] + '_' + str(idx) + '_' + str(cropped_x_y[0]) + '_' + str(cropped_x_y[1]) + '.png'
                input_image_fname = os.path.join(cropped_input_images_path, fname)
                input_im.save(input_image_fname)


def decode_segmap(encoded_im):
    # value 0 in encoded_im corresponds to first layer in the target which is foreground
    # value 1 in encoded_im corresponds to second layer in the target which is background
    encoded_im[encoded_im == 1] = 255
    encoded_im = encoded_im.astype(np.uint8)

    return encoded_im


def new_stitch(files, in_fname):
    input_image = Image.open(in_fname)
    orig_w, orig_h = input_image.size
    stitched_im = Image.new('L', (orig_w, orig_h))

    for f in files:
        pred = Image.open(f)
        f = f.split('/')[-1].split('.')[0]
        splits = f.split('_')

        x, y = splits[2:]
        x = int(x)
        y = int(y)

        stitched_im.paste(pred, (x,y))

    return stitched_im


def stitch_predicted(args):
    output_path = os.path.join(args.output_path, 'temp')
    input_files = os.listdir(args.im_dir)
    total_output_files = [os.path.join(output_path, f) for f in os.listdir(output_path)]

    for im_name in input_files:
        f = im_name.split('.')[0]
        pred_files = [i for i in total_output_files if f in i]
        stitched = new_stitch(pred_files, os.path.join(args.im_dir, im_name))
        stitched.save(os.path.join(args.output_path, im_name))

