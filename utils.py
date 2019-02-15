from PIL import Image
import numpy as np


def crop_x(y1, y2, image):
    x2 = 0
    cropped_im_list = []
    count_x = 0
    while x2 + 256 < image.size[0]:
        x1 = count_x * 256
        x2 = x1 + 256
        im = image.crop(box=(x1, y1, x2, y2))
        cropped_im_list.append(im)
        count_x += 1

    # residual portion on right
    if x2 + 256 > image.size[0]:
        x1 = image.size[0] - 256
        x2 = image.size[0]
        im = image.crop(box=(x1, y1, x2, y2))
        cropped_im_list.append(im)

    return cropped_im_list

def multiple_crop(image_path):
    image = Image.open(image_path)
    cropped_im_list = []
    x1, y1, x2, y2 = 0, 0, 0, 0
    count_y = 0
    while y2 + 256 < image.size[1]:
        y1 = count_y * 256
        y2 = y1 + 256

        cropped_x_list = crop_x(y1, y2, image)
        cropped_im_list.extend(cropped_x_list)
        count_y += 1

    # residual portion at bottom
    if y2 + 256 > image.size[1]:
        y1 = image.size[1] - 256
        y2 = image.size[1]

        cropped_x_list = crop_x(y1, y2, image)
        cropped_im_list.extend(cropped_x_list)

    return cropped_im_list

def decode_segmap(encoded_im):
    # value 0 in encoded_im corresponds to first layer in the target which is foreground
    # value 1 in encoded_im corresponds to second layer in the target which is background
    encoded_im[encoded_im == 1] = 255
    encoded_im = encoded_im.astype(np.uint8)

    return encoded_im