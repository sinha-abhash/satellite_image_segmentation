import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

from fcn import FCN
from data import SegmentationData
from utils import crop_and_save, decode_segmap

from PIL import Image
from matplotlib import pyplot as plt
import os
import argparse


def plot(losses, args):
    plt.plot(losses, label='loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()

    if not os.path.exists(args.plot_path):
        os.makedirs(args.plot_path)
    plt.savefig(os.path.join(args.plot_path, 'loss.png'))


def main(args):
    torch.manual_seed(1)

    # crop input image and ground truth and save on disk
    cropped_input_images_path = os.path.join(args.save_cropped, 'input_images')
    cropped_gt_images_path = os.path.join(args.save_cropped, 'gt_images')

    if args.crop_images:
        crop_and_save(args, cropped_input_images_path, cropped_gt_images_path)

    seg_dataset = SegmentationData(cropped_input_images_path, cropped_gt_images_path, args.n_classes, args.phase)
    train_loader = DataLoader(seg_dataset, shuffle=True, num_workers=4, batch_size=args.batch_size)

    model = FCN(args.n_classes)
    use_gpu = torch.cuda.is_available()
    num_gpu = list(range(torch.cuda.device_count()))
    if use_gpu :
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=num_gpu)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    losses = []
    for epoch in range(args.n_epoch):
        for i, (image, segement_im) in enumerate(train_loader):
            image = image.float()
            images = Variable(image.cuda())
            labels = Variable(segement_im.cuda())

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            # add loss to a list for plotting it later
            if i == 0:
                losses.append(loss)
            print("epoch{} iteration {} loss: {}".format(epoch, i, loss.data.item()))

            if epoch%5 == 0:
                pred = outputs.data.max(1)[1].cpu().numpy()[0]

                decoded = decode_segmap(pred)
                decoded = Image.fromarray(decoded)

                path = os.path.join(args.output_path, 'output_%d_%d.png' % (epoch, i))

                decoded.save(path)

    # plot loss
    plot(losses, args)

    # save model
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    model_name = os.path.join(args.model_path, 'fcn.pt')
    torch.save(model, model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Provide inputs for FCN models parameters")
    parser.add_argument('--im_dir', type=str, default='./images/images',
                        help='provide path of image files')
    parser.add_argument('--seg_dir', type=str, default='./images/seg_images',
                        help='provide number of classes a pixel can have')
    parser.add_argument('--save_cropped', type=str, default='./images/cropped',
                        help='provide path for saving cropped images and ground truth')
    parser.add_argument('--n_classes', type=int, default=2, help='provide number of classes a pixel can have')
    parser.add_argument('--n_epoch', type=int, default=20, help='provide number of epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='number of images in a batch')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--lr_decay', type=float, default=0.95)
    parser.add_argument('--momentum', type=float, default=0.95)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--crop_images', type=bool, default=True,
                        help='True if images and ground truths needed to croped')
    parser.add_argument('--model_path', type=str, default='./saved_model',
                        help='provide path for saved models')
    parser.add_argument('--output_path', type=str, default='./images/output/',
                        help='provide path for saving output image')
    parser.add_argument('--plot_path', type=str, default='./plot/',
                        help='provide path for saving output image')
    parser.add_argument('--phase', type=str, default='train', choices=['train', 'test'], help='train or test')
    args = parser.parse_args()
    main(args)